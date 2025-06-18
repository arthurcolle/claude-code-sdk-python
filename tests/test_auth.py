"""Tests for authentication module."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, mock_open
import tempfile
import os

from claude_code_sdk import (
    ClaudeAuth,
    OAuthConfig,
    OAuthFlow,
    AuthToken,
    TokenStorage,
    AuthenticationError,
)
from claude_code_sdk.auth import LocalCallbackServer


class TestAuthToken:
    """Test AuthToken class."""
    
    def test_token_creation(self):
        """Test creating an auth token."""
        token = AuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token="refresh_token",
            scope="read write"
        )
        
        assert token.access_token == "test_token"
        assert token.token_type == "Bearer"
        assert token.refresh_token == "refresh_token"
        assert token.scope == "read write"
        assert not token.is_expired()
    
    def test_token_expiry(self):
        """Test token expiry checking."""
        # Expired token
        expired_token = AuthToken(
            access_token="test_token",
            expires_at=datetime.now() - timedelta(hours=1)
        )
        assert expired_token.is_expired()
        
        # Valid token
        valid_token = AuthToken(
            access_token="test_token",
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert not valid_token.is_expired()
        
        # No expiry
        no_expiry_token = AuthToken(access_token="test_token")
        assert not no_expiry_token.is_expired()
    
    def test_token_serialization(self):
        """Test token to/from dict conversion."""
        original = AuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token="refresh_token",
            scope="read write"
        )
        
        # Convert to dict
        data = original.to_dict()
        assert data["access_token"] == "test_token"
        assert data["token_type"] == "Bearer"
        assert data["refresh_token"] == "refresh_token"
        assert data["scope"] == "read write"
        assert "expires_at" in data
        
        # Convert back from dict
        restored = AuthToken.from_dict(data)
        assert restored.access_token == original.access_token
        assert restored.token_type == original.token_type
        assert restored.refresh_token == original.refresh_token
        assert restored.scope == original.scope
        assert restored.expires_at.isoformat() == original.expires_at.isoformat()


class TestOAuthConfig:
    """Test OAuth configuration."""
    
    def test_default_config(self):
        """Test default OAuth configuration."""
        config = OAuthConfig()
        
        assert config.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        assert config.redirect_uri == "http://localhost:54545/callback"
        assert config.authorize_url == "https://claude.ai/oauth/authorize"
        assert config.token_url == "https://claude.ai/oauth/token"
        assert config.client_secret is None
    
    def test_custom_config(self):
        """Test custom OAuth configuration."""
        config = OAuthConfig(
            client_id="custom_id",
            client_secret="custom_secret",
            redirect_uri="http://localhost:9000/callback",
            authorize_url="https://custom.auth/authorize",
            token_url="https://custom.auth/token"
        )
        
        assert config.client_id == "custom_id"
        assert config.client_secret == "custom_secret"
        assert config.redirect_uri == "http://localhost:9000/callback"
        assert config.authorize_url == "https://custom.auth/authorize"
        assert config.token_url == "https://custom.auth/token"
    
    @patch.dict("os.environ", {
        "CLAUDE_OAUTH_CLIENT_ID": "env_client_id",
        "CLAUDE_OAUTH_CLIENT_SECRET": "env_secret",
        "CLAUDE_OAUTH_REDIRECT_URI": "http://env.redirect",
        "CLAUDE_OAUTH_AUTHORIZE_URL": "https://env.auth/authorize",
        "CLAUDE_OAUTH_TOKEN_URL": "https://env.auth/token"
    })
    def test_env_config(self):
        """Test OAuth configuration from environment variables."""
        config = OAuthConfig()
        
        assert config.client_id == "env_client_id"
        assert config.client_secret == "env_secret"
        assert config.redirect_uri == "http://env.redirect"
        assert config.authorize_url == "https://env.auth/authorize"
        assert config.token_url == "https://env.auth/token"


class TestTokenStorage:
    """Test token storage."""
    
    def test_save_and_load_token(self):
        """Test saving and loading tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            storage = TokenStorage(storage_path)
            
            # Create and save token
            token = AuthToken(
                access_token="test_token",
                expires_at=datetime.now() + timedelta(hours=1),
                refresh_token="refresh_token"
            )
            storage.save_token(token)
            
            # Verify file exists with correct permissions
            assert storage_path.exists()
            assert oct(storage_path.stat().st_mode)[-3:] == "600"
            
            # Load token
            loaded_token = storage.load_token()
            assert loaded_token is not None
            assert loaded_token.access_token == "test_token"
            assert loaded_token.refresh_token == "refresh_token"
    
    def test_load_nonexistent_token(self):
        """Test loading when no token exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            storage = TokenStorage(storage_path)
            
            token = storage.load_token()
            assert token is None
    
    def test_delete_token(self):
        """Test deleting stored token."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            storage = TokenStorage(storage_path)
            
            # Save a token
            token = AuthToken(access_token="test_token")
            storage.save_token(token)
            assert storage_path.exists()
            
            # Delete it
            storage.delete_token()
            assert not storage_path.exists()


class TestOAuthFlow:
    """Test OAuth flow implementation."""
    
    @pytest.fixture
    def oauth_flow(self):
        """Create OAuth flow instance."""
        config = OAuthConfig()
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        return OAuthFlow(config, storage)
    
    def test_get_authorization_url(self, oauth_flow):
        """Test generating authorization URL."""
        url = oauth_flow.get_authorization_url()
        
        assert url.startswith(oauth_flow.config.authorize_url)
        assert f"client_id={oauth_flow.config.client_id}" in url
        assert "response_type=code" in url
        assert f"redirect_uri={oauth_flow.config.redirect_uri}" in url
        assert "scope=claude_code%3Aread+claude_code%3Awrite" in url
    
    def test_get_authorization_url_with_state(self, oauth_flow):
        """Test authorization URL with state parameter."""
        url = oauth_flow.get_authorization_url(state="test_state")
        assert "state=test_state" in url
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self, oauth_flow):
        """Test exchanging authorization code for token."""
        # Mock HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "new_refresh_token",
            "scope": "read write"
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with oauth_flow:
            oauth_flow._http_client = mock_client
            
            token = await oauth_flow.exchange_code_for_token("auth_code")
            
            assert token.access_token == "new_access_token"
            assert token.refresh_token == "new_refresh_token"
            assert token.scope == "read write"
            assert token.expires_at is not None
            
            # Verify token was saved
            saved_token = oauth_flow.storage.load_token()
            assert saved_token is not None
            assert saved_token.access_token == "new_access_token"
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, oauth_flow):
        """Test refreshing an expired token."""
        # Mock HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with oauth_flow:
            oauth_flow._http_client = mock_client
            
            token = await oauth_flow.refresh_token("old_refresh_token")
            
            assert token.access_token == "refreshed_token"
            assert token.expires_at is not None
            
            # Verify refresh token is preserved if not provided
            assert token.refresh_token == "old_refresh_token"


class TestClaudeAuth:
    """Test main authentication class."""
    
    @pytest.mark.asyncio
    async def test_api_key_auth(self):
        """Test API key authentication."""
        async with ClaudeAuth(use_oauth=False, api_key="test_key") as auth:
            headers = await auth.authenticate()
            assert headers == {"X-API-Key": "test_key"}
    
    @pytest.mark.asyncio
    async def test_api_key_from_env(self):
        """Test API key from environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env_key"}):
            async with ClaudeAuth(use_oauth=False) as auth:
                headers = await auth.authenticate()
                assert headers == {"X-API-Key": "env_key"}
    
    @pytest.mark.asyncio
    async def test_no_api_key_error(self):
        """Test error when no API key is provided."""
        async with ClaudeAuth(use_oauth=False, api_key=None) as auth:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(AuthenticationError, match="No API key provided"):
                    await auth.authenticate()
    
    def test_get_env_vars_api_key(self):
        """Test getting environment variables for API key auth."""
        auth = ClaudeAuth(use_oauth=False, api_key="test_key")
        env_vars = auth.get_env_vars()
        assert env_vars == {"ANTHROPIC_API_KEY": "test_key"}
    
    def test_get_env_vars_oauth(self):
        """Test getting environment variables for OAuth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TokenStorage(Path(tmpdir) / "tokens.json")
            
            # Save a valid token
            token = AuthToken(
                access_token="oauth_token",
                expires_at=datetime.now() + timedelta(hours=1)
            )
            storage.save_token(token)
            
            auth = ClaudeAuth(use_oauth=True, token_storage=storage)
            env_vars = auth.get_env_vars()
            
            # OAuth tokens are passed as Bearer tokens
            assert env_vars == {"ANTHROPIC_API_KEY": "Bearer oauth_token"}


class TestLocalCallbackServer:
    """Test OAuth callback server."""
    
    @pytest.mark.asyncio
    async def test_server_startup(self):
        """Test callback server starts correctly."""
        server = LocalCallbackServer(port=54545)
        await server.start()
        
        # Server should be running
        assert server._server_task is not None
        assert not server._server_task.done()
        
        # Clean up
        if server._server_task:
            server._server_task.cancel()
    
    @pytest.mark.asyncio
    async def test_server_handles_success_callback(self):
        """Test server handles successful OAuth callback."""
        server = LocalCallbackServer(port=54546)
        await server.start()
        
        try:
            # Simulate successful callback
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:54546/callback?code=test_code&state=test_state"
                ) as response:
                    assert response.status == 200
                    html = await response.text()
                    assert "Authentication successful!" in html
                    assert "success" in html
            
            # Server should have captured the code
            assert server.auth_code == "test_code"
            assert server.error is None
            
        finally:
            if server._server_task:
                server._server_task.cancel()
    
    @pytest.mark.asyncio
    async def test_server_handles_error_callback(self):
        """Test server handles OAuth error callback."""
        server = LocalCallbackServer(port=54547)
        await server.start()
        
        try:
            # Simulate error callback
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:54547/callback?error=access_denied&error_description=User%20denied%20access"
                ) as response:
                    assert response.status == 200
                    html = await response.text()
                    assert "Authentication failed" in html
                    assert "error" in html
            
            # Server should have captured the error
            assert server.auth_code is None
            assert server.error == "User denied access"
            
        finally:
            if server._server_task:
                server._server_task.cancel()
    
    @pytest.mark.asyncio
    async def test_wait_for_code_success(self):
        """Test waiting for authorization code."""
        server = LocalCallbackServer(port=54548)
        await server.start()
        
        try:
            # Set code after a short delay
            async def set_code():
                await asyncio.sleep(0.1)
                server.auth_code = "delayed_code"
            
            import asyncio
            asyncio.create_task(set_code())
            
            # Wait for code
            code = await server.wait_for_code(timeout=1)
            assert code == "delayed_code"
            
        finally:
            if server._server_task:
                server._server_task.cancel()
    
    @pytest.mark.asyncio
    async def test_wait_for_code_timeout(self):
        """Test timeout while waiting for code."""
        server = LocalCallbackServer(port=54549)
        await server.start()
        
        try:
            # Don't set any code
            with pytest.raises(AuthenticationError, match="Authentication timeout"):
                await server.wait_for_code(timeout=0.5)
                
        finally:
            if server._server_task:
                server._server_task.cancel()
    
    @pytest.mark.asyncio
    async def test_wait_for_code_error(self):
        """Test error while waiting for code."""
        server = LocalCallbackServer(port=54550)
        await server.start()
        
        try:
            # Set error instead of code
            server.error = "Test error"
            
            with pytest.raises(AuthenticationError, match="Authentication failed: Test error"):
                await server.wait_for_code(timeout=1)
                
        finally:
            if server._server_task:
                server._server_task.cancel()


class TestOAuthFlowAdvanced:
    """Advanced OAuth flow tests."""
    
    @pytest.mark.asyncio
    async def test_oauth_flow_context_manager(self):
        """Test OAuth flow as async context manager."""
        config = OAuthConfig()
        flow = OAuthFlow(config)
        
        # Should create HTTP client on entry
        assert flow._http_client is None
        
        async with flow:
            assert flow._http_client is not None
            assert flow.client == flow._http_client
        
        # Client should be closed on exit
        # (Can't easily test this without mocking)
    
    @pytest.mark.asyncio
    async def test_oauth_flow_without_context_manager(self):
        """Test error when using flow without context manager."""
        config = OAuthConfig()
        flow = OAuthFlow(config)
        
        with pytest.raises(RuntimeError, match="OAuthFlow must be used as async context manager"):
            _ = flow.client
    
    @pytest.mark.asyncio
    async def test_exchange_code_with_client_secret(self):
        """Test code exchange with client secret."""
        config = OAuthConfig(client_secret="test_secret")
        flow = OAuthFlow(config)
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "secret_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with flow:
            flow._http_client = mock_client
            
            token = await flow.exchange_code_for_token("test_code")
            
            # Verify client secret was included
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["client_secret"] == "test_secret"
    
    @pytest.mark.asyncio
    async def test_exchange_code_failure(self):
        """Test handling of code exchange failure."""
        config = OAuthConfig()
        flow = OAuthFlow(config)
        
        # Mock failed HTTP response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid code"
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with flow:
            flow._http_client = mock_client
            
            with pytest.raises(AuthenticationError, match="Token exchange failed: 400"):
                await flow.exchange_code_for_token("bad_code")
    
    @pytest.mark.asyncio
    async def test_refresh_token_preserves_refresh_token(self):
        """Test that refresh token is preserved if not in response."""
        config = OAuthConfig()
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        flow = OAuthFlow(config, storage)
        
        # Mock response without new refresh token
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with flow:
            flow._http_client = mock_client
            
            token = await flow.refresh_token("original_refresh_token")
            
            # Original refresh token should be preserved
            assert token.refresh_token == "original_refresh_token"
            
            # Token should be saved
            saved = storage.load_token()
            assert saved.refresh_token == "original_refresh_token"
    
    @pytest.mark.asyncio
    async def test_get_valid_token_refresh_expired(self):
        """Test getting valid token when current is expired."""
        config = OAuthConfig()
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        
        # Save expired token with refresh token
        expired_token = AuthToken(
            access_token="old_token",
            expires_at=datetime.now() - timedelta(hours=1),
            refresh_token="refresh_token"
        )
        storage.save_token(expired_token)
        
        flow = OAuthFlow(config, storage)
        
        # Mock successful refresh
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with flow:
            flow._http_client = mock_client
            
            token = await flow.get_valid_token()
            
            assert token is not None
            assert token.access_token == "refreshed_token"
            assert not token.is_expired()
    
    @pytest.mark.asyncio
    async def test_get_valid_token_refresh_fails(self):
        """Test handling when token refresh fails."""
        config = OAuthConfig()
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        
        # Save expired token
        expired_token = AuthToken(
            access_token="old_token",
            expires_at=datetime.now() - timedelta(hours=1),
            refresh_token="bad_refresh_token"
        )
        storage.save_token(expired_token)
        
        flow = OAuthFlow(config, storage)
        
        # Mock failed refresh
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid refresh token"
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        async with flow:
            flow._http_client = mock_client
            
            token = await flow.get_valid_token()
            
            # Should return None and clear storage
            assert token is None
            assert storage.load_token() is None


class TestClaudeAuthAdvanced:
    """Advanced ClaudeAuth tests."""
    
    @pytest.mark.asyncio
    async def test_oauth_authentication_with_existing_token(self):
        """Test OAuth auth when valid token exists."""
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        
        # Save valid token
        valid_token = AuthToken(
            access_token="existing_token",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1)
        )
        storage.save_token(valid_token)
        
        async with ClaudeAuth(use_oauth=True, token_storage=storage) as auth:
            headers = await auth.authenticate()
            
            assert headers == {"Authorization": "Bearer existing_token"}
    
    @pytest.mark.asyncio
    async def test_oauth_authentication_performs_flow(self):
        """Test OAuth auth performs full flow when no token."""
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        
        async with ClaudeAuth(use_oauth=True, token_storage=storage) as auth:
            # Mock the OAuth flow
            with patch.object(auth, 'perform_oauth_flow') as mock_flow:
                mock_token = AuthToken(
                    access_token="new_token",
                    token_type="Bearer"
                )
                mock_flow.return_value = mock_token
                
                headers = await auth.authenticate()
                
                assert headers == {"Authorization": "Bearer new_token"}
                mock_flow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_perform_oauth_flow(self):
        """Test full OAuth flow execution."""
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        
        async with ClaudeAuth(use_oauth=True, token_storage=storage) as auth:
            # We'll mock the key components
            with patch('webbrowser.open') as mock_browser, \
                 patch.object(LocalCallbackServer, 'start', new_callable=AsyncMock) as mock_start, \
                 patch.object(LocalCallbackServer, 'wait_for_code', new_callable=AsyncMock) as mock_wait:
                
                mock_wait.return_value = "auth_code_123"
                
                # Mock token exchange
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "access_token": "flow_token",
                    "token_type": "Bearer",
                    "expires_in": 3600
                }
                
                auth._oauth_flow._http_client = AsyncMock()
                auth._oauth_flow._http_client.post.return_value = mock_response
                
                token = await auth.perform_oauth_flow()
                
                assert token.access_token == "flow_token"
                mock_browser.assert_called_once()
                mock_start.assert_called_once()
                mock_wait.assert_called_once()
    
    def test_get_env_vars_expired_oauth_token(self):
        """Test env vars when OAuth token is expired."""
        storage = TokenStorage(Path(tempfile.mkdtemp()) / "tokens.json")
        
        # Save expired token
        expired_token = AuthToken(
            access_token="expired_token",
            expires_at=datetime.now() - timedelta(hours=1)
        )
        storage.save_token(expired_token)
        
        auth = ClaudeAuth(use_oauth=True, token_storage=storage)
        env_vars = auth.get_env_vars()
        
        # Should return empty dict for expired token
        assert env_vars == {}
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test proper cleanup in context manager."""
        auth = ClaudeAuth(use_oauth=True)
        
        # Enter context
        await auth.__aenter__()
        assert auth._oauth_flow is not None
        
        # Exit context
        await auth.__aexit__(None, None, None)
        # OAuth flow should be cleaned up (hard to test internals)
    
    @pytest.mark.asyncio
    async def test_authentication_without_context_manager(self):
        """Test error when using OAuth without context manager."""
        auth = ClaudeAuth(use_oauth=True)
        
        with pytest.raises(RuntimeError, match="ClaudeAuth must be used as async context manager"):
            await auth.authenticate()


class TestTokenStorageEdgeCases:
    """Test edge cases for token storage."""
    
    def test_storage_directory_creation(self):
        """Test storage creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested path that doesn't exist
            storage_path = Path(tmpdir) / "deep" / "nested" / "path" / "tokens.json"
            storage = TokenStorage(storage_path)
            
            # Parent directories should be created
            assert storage.storage_path.parent.exists()
            
            # Save a token to verify it works
            token = AuthToken(access_token="test")
            storage.save_token(token)
            assert storage_path.exists()
    
    def test_storage_handles_corrupted_file(self):
        """Test storage handles corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            storage = TokenStorage(storage_path)
            
            # Write corrupted JSON
            storage_path.write_text("{ corrupted json")
            
            # Should return None instead of crashing
            token = storage.load_token()
            assert token is None
    
    def test_storage_preserves_existing_data(self):
        """Test storage preserves other data in file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            storage = TokenStorage(storage_path)
            
            # Write existing data
            existing_data = {
                "other_key": "other_value",
                "nested": {"data": "preserved"}
            }
            storage_path.write_text(json.dumps(existing_data))
            
            # Save token
            token = AuthToken(access_token="new_token")
            storage.save_token(token)
            
            # Load and check
            data = json.loads(storage_path.read_text())
            assert data["other_key"] == "other_value"
            assert data["nested"]["data"] == "preserved"
            assert "token" in data
            assert "updated_at" in data
    
    def test_delete_nonexistent_token(self):
        """Test deleting when no token file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            storage = TokenStorage(storage_path)
            
            # Should not raise error
            storage.delete_token()
            assert not storage_path.exists()


class TestOAuthConfigEnvVarPrecedence:
    """Test environment variable precedence in OAuth config."""
    
    def test_explicit_values_override_env(self):
        """Test explicit values take precedence over env vars."""
        with patch.dict("os.environ", {
            "CLAUDE_OAUTH_CLIENT_ID": "env_id",
            "CLAUDE_OAUTH_CLIENT_SECRET": "env_secret",
        }):
            config = OAuthConfig(
                client_id="explicit_id",
                client_secret="explicit_secret"
            )
            
            assert config.client_id == "explicit_id"
            assert config.client_secret == "explicit_secret"
    
    def test_partial_env_override(self):
        """Test mixing env vars and explicit values."""
        with patch.dict("os.environ", {
            "CLAUDE_OAUTH_CLIENT_ID": "env_id",
            "CLAUDE_OAUTH_REDIRECT_URI": "http://env.redirect",
        }):
            config = OAuthConfig(
                client_secret="explicit_secret"
            )
            
            assert config.client_id == "env_id"  # From env
            assert config.client_secret == "explicit_secret"  # Explicit
            assert config.redirect_uri == "http://env.redirect"  # From env


# Convenience function tests
@pytest.mark.asyncio
async def test_login_function():
    """Test the login convenience function."""
    from claude_code_sdk.auth import login
    
    with patch('claude_code_sdk.auth.ClaudeAuth') as mock_auth_class:
        mock_auth = AsyncMock()
        mock_auth.__aenter__.return_value = mock_auth
        mock_auth.__aexit__.return_value = None
        mock_auth.perform_oauth_flow.return_value = AuthToken(access_token="test")
        mock_auth_class.return_value = mock_auth
        
        await login()
        
        mock_auth_class.assert_called_once_with(use_oauth=True)
        mock_auth.perform_oauth_flow.assert_called_once()


@pytest.mark.asyncio
async def test_logout_function():
    """Test the logout convenience function."""
    from claude_code_sdk.auth import logout
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a token file
        storage_path = Path(tmpdir) / ".claude_code" / "tokens.json"
        storage_path.parent.mkdir(parents=True)
        storage_path.write_text('{"token": {"access_token": "test"}}')
        
        with patch('claude_code_sdk.auth.TokenStorage') as mock_storage_class:
            mock_storage = Mock()
            mock_storage.delete_token = Mock()
            mock_storage_class.return_value = mock_storage
            
            await logout()
            
            mock_storage.delete_token.assert_called_once()


@pytest.mark.asyncio
async def test_get_auth_headers_function():
    """Test the get_auth_headers convenience function."""
    from claude_code_sdk.auth import get_auth_headers
    
    with patch('claude_code_sdk.auth.ClaudeAuth') as mock_auth_class:
        mock_auth = AsyncMock()
        mock_auth.__aenter__.return_value = mock_auth
        mock_auth.__aexit__.return_value = None
        mock_auth.authenticate.return_value = {"Authorization": "Bearer test"}
        mock_auth_class.return_value = mock_auth
        
        headers = await get_auth_headers()
        
        assert headers == {"Authorization": "Bearer test"}
        mock_auth.authenticate.assert_called_once()