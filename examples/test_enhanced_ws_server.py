#!/usr/bin/env python3
"""Test script for the enhanced WebSocket server functionality."""

import asyncio
import json
import websockets


async def test_enhanced_server():
    """Test the enhanced WebSocket server features."""
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Wait for connection established
        msg = await websocket.recv()
        data = json.loads(msg)
        print(f"Connected: {data}")
        
        # Check capabilities
        capabilities = data['data']['capabilities']
        print(f"Server capabilities: {capabilities}")
        
        # Test 1: Send a simple query
        print("\n--- Test 1: Simple Query ---")
        await websocket.send(json.dumps({
            "type": "query",
            "prompt": "What is 2+2?",
            "options": {
                "allowed_tools": ["Read"],
                "permission_mode": "default"
            }
        }))
        
        # Collect responses
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            print(f"Received: {data['type']}")
            
            if data['type'] == 'query_end':
                break
        
        # Test 2: Define a new tool (if supported)
        if capabilities.get('tool_definition'):
            print("\n--- Test 2: Tool Definition ---")
            tool_def = {
                "name": "EchoTool",
                "description": "A simple tool that echoes input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo"
                        }
                    },
                    "required": ["message"]
                }
            }
            
            await websocket.send(json.dumps({
                "type": "define_tool",
                "tool": tool_def
            }))
            
            # Wait for response
            msg = await websocket.recv()
            data = json.loads(msg)
            print(f"Tool definition result: {data}")
        
        # Test 3: Query with interrupt (if supported)
        if capabilities.get('interrupt_query'):
            print("\n--- Test 3: Query with Interrupt ---")
            await websocket.send(json.dumps({
                "type": "query",
                "prompt": "Count from 1 to 100 slowly",
                "options": {
                    "allowed_tools": ["Task"],
                    "permission_mode": "default"
                }
            }))
            
            # Wait a bit then interrupt
            await asyncio.sleep(2)
            print("Sending interrupt...")
            await websocket.send(json.dumps({
                "type": "interrupt"
            }))
            
            # Collect responses
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"Received: {data['type']}")
                
                if data['type'] in ['query_end', 'query_interrupted']:
                    break
        
        print("\n--- Tests completed ---")


if __name__ == "__main__":
    print("Testing Enhanced WebSocket Server")
    print("Make sure the server is running on localhost:8000")
    print("Run with: python src/claude_code_sdk/websocket_server.py")
    
    asyncio.run(test_enhanced_server())