"""Tests for authentication module."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, mock_open
import tempfile

from claude_code_sdk import (
    ClaudeAuth,
    OAuthConfig,
    OAuthFlow,
    AuthToken,
    TokenStorage,
    AuthenticationError,
)


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
        
        assert config.client_id == "claude-code-sdk"
        assert config.redirect_uri == "http://localhost:8089/callback"
        assert config.authorize_url == "https://console.anthropic.com/oauth/authorize"
        assert config.token_url == "https://api.anthropic.com/oauth/token"
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