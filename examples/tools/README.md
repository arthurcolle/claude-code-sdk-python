# Tool Management API Examples

This directory contains examples demonstrating the integration between Claude Code SDK and the Tool Management API.

## Prerequisites

1. Install the Claude Code SDK:
   ```bash
   pip install claude-code-sdk
   ```

2. Ensure you have access to the Tool Management API (default: https://arthurcolle--registry.modal.run)

## Examples

### 1. Basic Tool Usage (`basic_tool_usage.py`)

Demonstrates fundamental operations:
- Direct Tool API usage
- Searching for tools
- Executing tools
- Automatic tool execution with `query_with_tools`
- Tool discovery and availability checking

```bash
python examples/tools/basic_tool_usage.py
```

### 2. Tool Search and Suggestions (`tool_search_and_suggest.py`)

Shows advanced search capabilities:
- Search by capability description
- Find similar tools using vector similarity
- Get tool suggestions for specific tasks
- List tools by namespace/category

```bash
python examples/tools/tool_search_and_suggest.py
```

### 3. Custom Tool Creation (`custom_tool_creation.py`)

Demonstrates creating and managing custom tools:
- Creating Python-based tools
- Creating HTTP-based tools
- Batch tool creation
- Updating existing tools

```bash
python examples/tools/custom_tool_creation.py
```

### 4. Tool Execution Hooks (`tool_execution_hooks.py`)

Advanced examples using execution hooks:
- Monitoring tool usage
- Remote tool execution
- Tool filtering and access control
- Collecting usage metrics

```bash
python examples/tools/tool_execution_hooks.py
```

## Key Concepts

### Tool Management Client

The `ToolManagementClient` provides direct access to the Tool Management API:

```python
from claude_code_sdk.tools import ToolManagementClient

async with ToolManagementClient() as client:
    # Search for tools
    tools = await client.search_tools("calculator")
    
    # Execute a tool
    result = await client.execute_tool(
        tool_id=tools[0].id,
        input_data={"expression": "2 + 2"}
    )
```

### Enhanced Query with Tools

The `query_with_tools` function extends the standard Claude Code query with tool management:

```python
from claude_code_sdk.tools import query_with_tools

async for message in query_with_tools(
    prompt="Calculate 10 * 25",
    use_remote_tools=True
):
    print(message)
```

### Tool Execution Hooks

Hooks allow you to intercept and control tool execution:

```python
from claude_code_sdk.tools import ToolExecutionHooks

hooks = ToolExecutionHooks(
    on_tool_use=lambda t: print(f"Using tool: {t.name}"),
    on_tool_result=lambda r: print(f"Result: {r.content}"),
    use_remote_tools=True
)
```

### Tool Discovery

The `ToolDiscovery` class provides utilities for finding and suggesting tools:

```python
from claude_code_sdk.tools import ToolDiscovery

discovery = ToolDiscovery(client)
suggestions = await discovery.suggest_tools_for_task(
    "analyze sentiment in text"
)
```

## API Endpoints

The Tool Management API provides these main endpoints:

- `GET /tools` - List all tools
- `POST /tools` - Create a new tool
- `GET /tools/{id}` - Get specific tool
- `PUT /tools/{id}` - Update tool
- `DELETE /tools/{id}` - Delete tool
- `POST /tools/search` - Search tools
- `POST /tools/retrieve` - Vector similarity search
- `POST /execute_tool` - Execute single tool
- `POST /execute_tools_sequential` - Execute tools in sequence
- `POST /execute_tools_parallel` - Execute tools in parallel

## Tool Types

Tools can have different action types:

1. **Python** - Execute Python code
2. **JavaScript** - Execute JavaScript code
3. **HTTP** - Make HTTP requests
4. **Service** - Call registered services

## Best Practices

1. **Error Handling**: Always wrap tool execution in try-except blocks
2. **Caching**: Use `ToolRegistry` for efficient tool lookups
3. **Monitoring**: Use hooks to track tool usage and performance
4. **Security**: Validate tool inputs and outputs
5. **Namespaces**: Organize tools by namespace for better management

## Troubleshooting

### Connection Issues

If you can't connect to the Tool Management API:
1. Check your network connection
2. Verify the API URL is correct
3. Ensure you have proper authentication (if required)

### Tool Not Found

If tools aren't found:
1. Check the tool name spelling
2. Try searching with different keywords
3. Verify the tool exists in the registry

### Execution Errors

If tool execution fails:
1. Check the input data format matches the tool's schema
2. Verify the tool's action configuration is correct
3. Check the API logs for detailed error messages