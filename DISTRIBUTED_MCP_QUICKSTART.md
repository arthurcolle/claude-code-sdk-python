# Distributed MCP Hub - Quick Start Guide

## 5-Minute Setup

Get up and running with the distributed MCP hub in 5 minutes.

### Step 1: Install (30 seconds)

```bash
# Clone repository
git clone https://github.com/your-org/claude-code-sdk-python
cd claude-code-sdk-python

# Install dependencies
pip install -e .
```

### Step 2: Run the Hub (30 seconds)

```bash
# Start the hub (runs in background)
python distributed_mcp_hub.py &

# Hub is now running with 5 default services and 15+ tools
```

### Step 3: Connect and Use (1 minute)

Create a file `my_first_client.py`:

```python
#!/usr/bin/env python3
import asyncio
from distributed_mcp_hub import SimpleMCPClient

async def main():
    # Connect to hub
    async with SimpleMCPClient() as client:

        # See what's available
        stats = await client.get_stats()
        print(f"Connected! {stats['total_tools']} tools available")

        # List tools
        for tool in client.list_tools()[:5]:
            print(f"  • {tool['name']}: {tool['description']}")

        # Use a tool
        result = await client.call("read_file", {
            "path": "/tmp/test.txt"
        })
        print(f"\nResult: {result}")

        # Use multiple tools in parallel
        results = await client.batch([
            ("http_get", {"url": "https://example.com", "headers": {}}),
            ("sql_query", {"query": "SELECT 1", "params": {}}),
            ("execute_python", {"code": "print('Hello!')", "timeout": 5})
        ])
        print(f"\nExecuted {len(results)} tools in parallel!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python my_first_client.py
```

Output:
```
Connected! 15 tools available
  • read_file: Read contents of a file
  • write_file: Write content to a file
  • sql_query: Execute SQL query
  • http_get: Make HTTP GET request
  • execute_python: Execute Python code

Result: {'status': 'success', 'tool': 'read_file', ...}

Executed 3 tools in parallel!
```

**That's it!** You now have a distributed tool ecosystem running.

---

## Common Use Cases

### Use Case 1: AI Agent with Tool Access

```python
from mcp_hub_claude_integration import ClaudeWithMCPHub

async def ai_agent_example():
    agent = ClaudeWithMCPHub()
    await agent.start()

    # Claude can now use all hub tools
    await agent.chat("Read the config file and query the database")
    await agent.chat("Get weather for San Francisco")
    await agent.chat("Generate a report from last week's analytics")

    await agent.stop()

asyncio.run(ai_agent_example())
```

### Use Case 2: Data Pipeline

```python
async def data_pipeline():
    async with SimpleMCPClient() as client:

        # Fetch data from multiple sources in parallel
        sources = await client.batch([
            ("http_get", {"url": "https://api1.com/data"}),
            ("sql_query", {"query": "SELECT * FROM raw_data"}),
            ("read_file", {"path": "/data/imports/file.csv"})
        ])

        # Process each source
        for data in sources:
            await client.call("execute_python", {
                "code": process_code,
                "timeout": 60
            })

        # Store results
        await client.call("sql_query", {
            "query": "INSERT INTO processed ...",
            "params": results
        })

asyncio.run(data_pipeline())
```

### Use Case 3: Custom Service

Add your own tools to the hub:

```python
from distributed_mcp_hub import ServiceNode, ToolSignature, ToolCategory

async def register_my_service():
    # Create service
    my_service = ServiceNode(
        service_id="my-service-001",
        name="My Custom Service",
        endpoint="internal://my-service",
        capabilities=["custom_feature"],
        tools=[
            ToolSignature(
                name="my_amazing_tool",
                description="Does something amazing",
                category=ToolCategory.CUSTOM,
                parameters={"input": "string"},
                returns={"output": "string"}
            )
        ]
    )

    # Register with hub
    from distributed_mcp_hub import MCPHub
    hub = MCPHub()
    await hub.start()
    await hub.registry.register_service(my_service)

    # Now ALL clients can use my_amazing_tool!

asyncio.run(register_my_service())
```

---

## Architecture Overview

```
                    ┌─────────────────┐
                    │   MCP Hub       │
                    │                 │
                    │  • 15+ tools    │
                    │  • 5 services   │
                    │  • Load balance │
                    │  • Auto-heal    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐     ┌──────────┐
    │ Client A │      │ Client B │     │ Claude   │
    │          │      │          │     │ Agent    │
    │ • Python │      │ • JS     │     │          │
    │ • CLI    │      │ • Web    │     │ • Query  │
    └──────────┘      └──────────┘     └──────────┘
```

Each client gets:
- ✓ Instant access to all tools
- ✓ Auto-discovery
- ✓ Load balancing
- ✓ Failover
- ✓ Caching
- ✓ Parallel execution

---

## Key Concepts

### 1. Hub
Central registry and coordinator. Manages:
- Service registration
- Tool discovery
- Execution routing
- Health monitoring

### 2. Services
Providers of functionality. Example:
- File System Service → read_file, write_file
- Database Service → sql_query, cache_get
- Web Service → http_get, scrape_page
- AI Service → claude_query, spawn_agent

### 3. Tools
Individual capabilities. Each tool has:
- Name (unique identifier)
- Description (what it does)
- Parameters (inputs)
- Returns (outputs)
- Category (organization)

### 4. Clients
Consumers of tools. Can be:
- Python scripts
- Web applications
- CLI tools
- AI agents (Claude, GPT, etc.)
- Other services

---

## Next Steps

### 1. **Explore Examples**

```bash
# Run all demos
python distributed_mcp_hub.py

# Run Claude integration demos
python mcp_hub_claude_integration.py
```

### 2. **Add Custom Tools**

See `examples/custom_service.py` for how to add your own tools.

### 3. **Deploy to Production**

See [DISTRIBUTED_MCP_HUB_README.md](DISTRIBUTED_MCP_HUB_README.md) for deployment guides.

### 4. **Read Full Documentation**

- [Full README](DISTRIBUTED_MCP_HUB_README.md)
- [Architecture Deep Dive](docs/architecture.md)
- [API Reference](docs/api.md)

---

## Troubleshooting

### "No module named 'distributed_mcp_hub'"

```bash
# Make sure you're in the right directory
cd claude-code-sdk-python

# Install in development mode
pip install -e .
```

### "Connection refused"

```bash
# Make sure hub is running
python distributed_mcp_hub.py &

# Check if it's listening
curl http://localhost:8000/health
```

### "Tool not found"

```python
# List available tools
async with SimpleMCPClient() as client:
    tools = client.list_tools()
    for tool in tools:
        print(tool['name'])
```

---

## FAQ

**Q: How many tools can I have?**
A: Unlimited. The hub scales horizontally.

**Q: Can I use this in production?**
A: Yes! See deployment guide for production setup with Redis, PostgreSQL, and load balancing.

**Q: Does this work with Claude Code SDK?**
A: Yes! See `mcp_hub_claude_integration.py` for examples.

**Q: Can I add my own services?**
A: Yes! Services can register themselves dynamically.

**Q: Is there authentication?**
A: Yes, see the Security section in the main README.

**Q: Can multiple hubs federate?**
A: Yes, hubs can connect to share tools across regions.

---

## Support

- GitHub Issues: https://github.com/your-org/claude-code-sdk-python/issues
- Discord: https://discord.gg/mcp-hub
- Email: support@mcp-hub.com

---

**Happy building! 🚀**
