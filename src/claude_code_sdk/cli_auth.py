"""CLI authentication commands for Claude Code SDK."""

import asyncio
import sys

from .auth import AuthenticationError, ClaudeAuth, OAuthConfig, TokenStorage


async def auth_login(
    client_id: str | None = None,
    client_secret: str | None = None,
) -> None:
    """
    Perform OAuth login for Claude Code Max plan.

    Args:
        client_id: OAuth client ID (optional)
        client_secret: OAuth client secret (optional)
    """
    try:
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
        )

        async with ClaudeAuth(use_oauth=True, oauth_config=config) as auth:
            print("Starting Claude Code OAuth login...")
            await auth.perform_oauth_flow()
            print(
                "\n✅ Login successful! You can now use Claude Code SDK without an API key."
            )

    except AuthenticationError as e:
        print(f"\n❌ Login failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nLogin cancelled.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


async def auth_logout() -> None:
    """Logout and remove stored authentication tokens."""
    try:
        storage = TokenStorage()
        storage.delete_token()
        print("✅ Logged out successfully.")
    except Exception as e:
        print(f"❌ Logout failed: {e}", file=sys.stderr)
        sys.exit(1)


async def auth_status() -> None:
    """Check authentication status."""
    try:
        storage = TokenStorage()
        token = storage.load_token()

        if not token:
            print("❌ Not authenticated. Run 'claude-auth login' to authenticate.")
            sys.exit(1)

        print("✅ Authenticated with Claude Code")

        if token.is_expired():
            print("⚠️  Token is expired. Will attempt to refresh on next use.")
        else:
            print("✅ Token is valid")
            if token.expires_at:
                print(
                    f"   Expires at: {token.expires_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        if token.scope:
            print(f"   Scopes: {token.scope}")

    except Exception as e:
        print(f"❌ Error checking status: {e}", file=sys.stderr)
        sys.exit(1)


async def auth_refresh() -> None:
    """Refresh authentication token."""
    try:
        storage = TokenStorage()
        token = storage.load_token()

        if not token:
            print("❌ Not authenticated. Run 'claude-auth login' to authenticate.")
            sys.exit(1)

        if not token.refresh_token:
            print("❌ No refresh token available. Please login again.")
            sys.exit(1)

        async with ClaudeAuth(use_oauth=True) as auth:
            if not auth._oauth_flow:
                print("❌ OAuth flow not initialized.")
                sys.exit(1)
            new_token = await auth._oauth_flow.refresh_token(token.refresh_token)
            print("✅ Token refreshed successfully!")
            if new_token.expires_at:
                print(
                    f"   New expiry: {new_token.expires_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )

    except AuthenticationError as e:
        print(f"❌ Token refresh failed: {e}", file=sys.stderr)
        print("Please run 'claude-auth login' to re-authenticate.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error refreshing token: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point for authentication commands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Claude Code SDK Authentication", prog="claude-auth"
    )

    subparsers = parser.add_subparsers(dest="command", help="Authentication commands")

    # Login command
    login_parser = subparsers.add_parser("login", help="Login with OAuth")
    login_parser.add_argument(
        "--client-id", help="OAuth client ID (defaults to environment variable)"
    )
    login_parser.add_argument(
        "--client-secret", help="OAuth client secret (defaults to environment variable)"
    )

    # Logout command
    subparsers.add_parser("logout", help="Logout and remove tokens")

    # Status command
    subparsers.add_parser("status", help="Check authentication status")

    # Refresh command
    subparsers.add_parser("refresh", help="Refresh authentication token")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate command
    if args.command == "login":
        asyncio.run(auth_login(args.client_id, args.client_secret))
    elif args.command == "logout":
        asyncio.run(auth_logout())
    elif args.command == "status":
        asyncio.run(auth_status())
    elif args.command == "refresh":
        asyncio.run(auth_refresh())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
