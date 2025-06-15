# Tool Management API Integration

This document describes the integration between Claude Code SDK and the Tool Management API.

## Overview

The Tool Management API integration allows Claude Code to:
- Discover and search for tools in a registry
- Execute tools remotely via API
- Create and manage custom tools
- Monitor tool usage with hooks

## Architecture

### Components

1. **Tool API Client** (`tools.client`)
   - HTTP client for Tool Management API
   - Supports all CRUD operations for tools
   - Tool execution (single, sequential, parallel)
   - Vector similarity search

2. **Tool Execution Hooks** (`tools.hooks`)
   - Intercept tool usage in Claude conversations
   - Execute tools remotely when enabled
   - Custom callbacks for monitoring

3. **Enhanced Client** (`tools.enhanced_client`)
   - Extends base Claude client with tool capabilities
   - Automatic remote tool execution
   - Message interception and modification

4. **Tool Discovery** (`tools.discovery`)
   - Search tools by capability
   - Find similar tools
   - Suggest tools for tasks
   - Organize by namespace/category

5. **Tool Registry** (`tools.discovery`)
   - Local caching of tools
   - Tool availability checking
   - Batch registration

## Usage

### Basic Tool Search

```python
import asyncio
from claude_code_sdk.tools import ToolManagementClient

async def main():
    async with ToolManagementClient() as client:
        # Search for calculator tools
        tools = await client.search_tools("calculator", limit=5)
        for tool in tools:
            print(f"{tool.name}: {tool.description}")

asyncio.run(main())
```

### Remote Tool Execution

```python
from claude_code_sdk.tools import query_with_tools

async def main():
    async for message in query_with_tools(
        prompt="Calculate 123 * 456",
        use_remote_tools=True
    ):
        print(message)

asyncio.run(main())
```

### Custom Tool Creation

```python
from claude_code_sdk.tools import (
    ToolManagementClient,
    Tool,
    ToolAction,
    ToolActionType,
    PythonActionConfig,
)

async def main():
    async with ToolManagementClient() as client:
        tool = Tool(
            name="string_reverser",
            description="Reverses a string",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "reversed": {"type": "string"}
                }
            },
            action=ToolAction(
                type=ToolActionType.PYTHON,
                python=PythonActionConfig(
                    code="def reverse(input_data): return {'reversed': input_data['text'][::-1]}",
                    function_name="reverse"
                )
            )
        )
        
        created = await client.create_tool(tool)
        print(f"Created tool: {created.id}")

asyncio.run(main())
```

### Tool Usage Monitoring

```python
from claude_code_sdk.tools import ToolExecutionHooks, EnhancedClient, ToolRegistryOptions

def on_tool_use(tool_use):
    print(f"Tool used: {tool_use.name}")

def on_tool_result(tool_result):
    print(f"Tool result: {tool_result.content}")

async def main():
    hooks = ToolExecutionHooks(
        on_tool_use=on_tool_use,
        on_tool_result=on_tool_result
    )
    
    options = ToolRegistryOptions(allowed_tools=["Read", "calculator"])
    client = EnhancedClient(tool_hooks=hooks)
    
    async for message in client.process_query("Read README.md", options):
        # Process messages
        pass

asyncio.run(main())
```

## API Endpoints

The Tool Management API provides:

- `GET /tools` - List all tools
- `POST /tools` - Create new tool
- `GET /tools/{id}` - Get specific tool
- `PUT /tools/{id}` - Update tool
- `DELETE /tools/{id}` - Delete tool
- `POST /tools/search` - Search tools
- `POST /tools/retrieve` - Vector similarity search
- `GET /tools/embeddings` - Get tool embeddings
- `POST /execute_tool` - Execute single tool
- `POST /execute_tools_sequential` - Sequential execution
- `POST /execute_tools_parallel` - Parallel execution

## Tool Types

### Action Types
- **HTTP**: Make HTTP requests
- **Python**: Execute Python code
- **JavaScript**: Execute JavaScript code
- **Service**: Call registered services

### Output Types
- **JSON**: Structured data
- **Text**: Plain text
- **Binary**: Binary data
- **AI**: AI-generated content

## Configuration

### Environment Variables
- `TOOL_API_URL`: Override default API URL

### Options
```python
from claude_code_sdk.tools import ToolRegistryOptions

options = ToolRegistryOptions(
    tool_api_url="https://custom.api",
    use_remote_tools=True,
    tool_namespace="my-tools",
    allowed_tools=["Read", "Write", "calculator"]
)
```

## Security Considerations

1. **Tool Validation**: Always validate tool inputs/outputs
2. **API Authentication**: Add auth headers if required
3. **Sandboxing**: Remote execution is sandboxed
4. **Rate Limiting**: Respect API rate limits
5. **Error Handling**: Handle network/execution errors

## Performance

1. **Caching**: Use ToolRegistry for local caching
2. **Batch Operations**: Use batch endpoints when possible
3. **Parallel Execution**: Execute independent tools in parallel
4. **Connection Pooling**: Client reuses HTTP connections

## Troubleshooting

### Common Issues

1. **Tool Not Found**
   - Check tool name spelling
   - Verify tool exists in registry
   - Check namespace if specified

2. **Execution Errors**
   - Validate input matches schema
   - Check tool action configuration
   - Review API error messages

3. **Connection Issues**
   - Verify API URL
   - Check network connectivity
   - Ensure proper authentication

## Future Enhancements

1. **WebSocket Support**: Real-time tool updates
2. **Tool Versioning**: Support multiple tool versions
3. **Dependency Management**: Tool dependency resolution
4. **Workflow Engine**: Complex tool workflows
5. **Analytics**: Detailed usage analytics