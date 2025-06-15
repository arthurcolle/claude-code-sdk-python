#!/usr/bin/env python3
"""
Example WebSocket server with inline HTML UI for Claude Code SDK.

This script demonstrates how to use the WebSocket server wrapper
to create a real-time web interface for interacting with Claude.

Usage:
    python websocket_ui_server.py

Then open http://localhost:8000 in your browser.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the SDK to the path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_code_sdk.websocket_server import ClaudeWebSocketServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run the WebSocket server with UI."""
    logger.info("Starting Claude Code SDK WebSocket server...")
    
    # Create and run the server
    server = ClaudeWebSocketServer()
    
    logger.info("Server running at http://localhost:8000")
    logger.info("Open this URL in your browser to access the UI")
    
    try:
        server.run(host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()