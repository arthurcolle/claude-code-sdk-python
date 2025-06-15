"""Example of creating and registering custom tools."""

import asyncio
from claude_code_sdk.tools import (
    ToolManagementClient,
    Tool,
    ToolAction,
    ToolActionType,
    ToolOutput,
    ToolOutputType,
    PythonActionConfig,
    HTTPActionConfig,
)


async def main():
    """Demonstrate custom tool creation."""
    
    async with ToolManagementClient() as client:
        # Example 1: Create a Python-based tool
        print("=== Creating Python-based tool ===")
        
        python_tool = Tool(
            name="string_reverser",
            namespace="examples",
            description="Reverses a given string",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to reverse"
                    }
                },
                "required": ["text"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "reversed": {
                        "type": "string",
                        "description": "The reversed text"
                    }
                }
            },
            action=ToolAction(
                type=ToolActionType.PYTHON,
                python=PythonActionConfig(
                    code="""
def reverse_string(input_data):
    text = input_data.get('text', '')
    return {'reversed': text[::-1]}
""",
                    function_name="reverse_string"
                )
            ),
            output=ToolOutput(
                type=ToolOutputType.JSON
            )
        )
        
        try:
            created_tool = await client.create_tool(python_tool)
            print(f"Created tool: {created_tool.name} (ID: {created_tool.id})")
            
            # Test the tool
            result = await client.execute_tool(
                tool_id=created_tool.id,
                input_data={"text": "Hello, World!"}
            )
            print(f"Execution result: {result.output_data}")
            
        except Exception as e:
            print(f"Error creating Python tool: {e}")
        
        # Example 2: Create an HTTP-based tool
        print("\n=== Creating HTTP-based tool ===")
        
        http_tool = Tool(
            name="weather_checker",
            namespace="examples",
            description="Gets current weather for a city",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "description": {"type": "string"},
                    "humidity": {"type": "number"}
                }
            },
            action=ToolAction(
                type=ToolActionType.HTTP,
                http=HTTPActionConfig(
                    method="GET",
                    url="https://api.example.com/weather/{city}",
                    headers={
                        "Accept": "application/json"
                    }
                )
            ),
            output=ToolOutput(
                type=ToolOutputType.JSON
            )
        )
        
        try:
            created_http_tool = await client.create_tool(http_tool)
            print(f"Created tool: {created_http_tool.name} (ID: {created_http_tool.id})")
        except Exception as e:
            print(f"Error creating HTTP tool: {e}")
        
        # Example 3: Create a batch of tools
        print("\n=== Creating batch of tools ===")
        
        batch_tools = [
            Tool(
                name="text_length_counter",
                namespace="examples",
                description="Counts the length of text",
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
                        "length": {"type": "integer"}
                    }
                }
            ),
            Tool(
                name="word_counter",
                namespace="examples",
                description="Counts words in text",
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
                        "word_count": {"type": "integer"}
                    }
                }
            )
        ]
        
        try:
            created_batch = await client.create_tools_batch(batch_tools)
            print(f"Created {len(created_batch)} tools in batch:")
            for tool in created_batch:
                print(f"  - {tool.name} (ID: {tool.id})")
        except Exception as e:
            print(f"Error creating batch tools: {e}")
        
        # Example 4: Update an existing tool
        print("\n=== Updating a tool ===")
        
        if created_batch:
            tool_to_update = created_batch[0]
            tool_to_update.description = "Counts the number of characters in text (updated)"
            
            try:
                updated_tool = await client.update_tool(
                    tool_to_update.id,
                    Tool(
                        name=tool_to_update.name,
                        namespace=tool_to_update.namespace,
                        description=tool_to_update.description,
                        input_schema=tool_to_update.input_schema,
                        output_schema=tool_to_update.output_schema
                    )
                )
                print(f"Updated tool: {updated_tool.name}")
                print(f"New description: {updated_tool.description}")
            except Exception as e:
                print(f"Error updating tool: {e}")


if __name__ == "__main__":
    asyncio.run(main())