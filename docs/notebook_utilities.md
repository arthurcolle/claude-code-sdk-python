# Notebook Utilities for Claude Code SDK

The Claude Code SDK includes special utilities for enhanced display of Claude's responses in Jupyter notebooks, including beautiful markdown rendering, syntax highlighting, and interactive displays.

## Features

- **Markdown Rendering**: Automatically convert Claude's markdown responses to styled HTML
- **Syntax Highlighting**: Code blocks are highlighted for better readability
- **Tool Visualization**: See when Claude uses tools with clear, colored formatting
- **Streaming Support**: Display responses as they arrive in real-time
- **Smart Detection**: Automatically detects if running in a Jupyter notebook

## Installation

The notebook utilities are included with the Claude Code SDK:

```bash
pip install claude-code-sdk
```

If you're installing from source:
```bash
pip install -e .
```

## Quick Start

```python
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.notebook_utils import display_claude_response, display_claude_stream

# Simple display of a response
async for message in query(prompt="Explain Python decorators"):
    display_claude_response(message)
```

## Basic Usage

### Rendering Markdown Text

```python
from claude_code_sdk.notebook_utils import render_markdown

markdown_text = """
# Welcome to Claude Code SDK!

This is **bold** and *italic* text.

## Features
- Bullet points
- `Inline code`
- [Links](https://github.com)

```python
def hello():
    print("Hello, World!")
```
"""

render_markdown(markdown_text)
```

### Displaying Claude Messages

```python
from claude_code_sdk.notebook_utils import NotebookDisplay
from claude_code_sdk import query

# Create a display helper with custom options
display = NotebookDisplay(
    show_tool_use=True,      # Show when Claude uses tools
    show_tool_results=True,  # Show tool execution results
    render_markdown=True,    # Convert markdown to HTML
    syntax_highlight=True,   # Highlight code syntax
    max_text_length=500     # Truncate long responses
)

async for message in query(prompt="Create a Python function"):
    if isinstance(message, AssistantMessage):
        display.display_message(message)
```

### Streaming Responses

For longer conversations, display Claude's responses as they stream:

```python
from claude_code_sdk.notebook_utils import display_claude_stream

# This will display messages as they arrive
messages = await display_claude_stream(
    query(prompt="Write a tutorial on async Python", options=options),
    render_markdown=True,
    show_tool_use=True
)
```

## Advanced Features

### Custom Display Configuration

You can customize how messages are displayed:

```python
from claude_code_sdk.notebook_utils import NotebookDisplay

display = NotebookDisplay(
    show_tool_use=True,        # Show tool invocations
    show_tool_results=True,    # Show tool results
    render_markdown=True,      # Render markdown as HTML
    syntax_highlight=True,     # Enable syntax highlighting
    max_text_length=1000      # Truncate at 1000 chars
)
```

### Tool Use Visualization

When Claude uses tools, they're displayed with special formatting:

- **Tool Use**: Blue background with tool name and parameters
- **Tool Success**: Green background with results
- **Tool Error**: Red background with error message

### Notebook Detection

The utilities automatically detect if running in Jupyter:

```python
from claude_code_sdk.notebook_utils import is_notebook

if is_notebook():
    print("Running in Jupyter!")
else:
    print("Running as a script")
```

## Example Notebook

See `examples/notebook_markdown_demo.ipynb` for a complete demonstration of all features.

## Comparison: With vs Without Notebook Utilities

### Without (Plain Text):
```
## Hello World

This is **bold** and *italic* text.

```python
def greet(name):
    return f"Hello, {name}!"
```
```

### With Notebook Utilities:
- Headers are styled and sized appropriately
- **Bold** and *italic* text are properly formatted
- Code blocks have syntax highlighting and a nice border
- Links are clickable and styled
- Lists are properly indented with bullets

## Tips and Best Practices

1. **Use in Jupyter Only**: These utilities are designed for Jupyter notebooks. In regular Python scripts, they gracefully fall back to plain text output.

2. **Streaming for Long Responses**: For queries that might generate long responses, use `display_claude_stream` to see results as they arrive.

3. **Custom Styling**: You can modify the HTML styles by creating your own display functions based on the provided utilities.

4. **Error Handling**: The utilities handle cases where IPython is not available, making your code portable.

## Troubleshooting

### ImportError for IPython
If you see import errors for IPython, install it with:
```bash
pip install ipython
```

### Markdown Not Rendering
Ensure you're running in a Jupyter notebook, not a regular Python script or IPython terminal.

### Styles Not Applying
Some Jupyter themes might override the inline styles. The utilities use inline CSS for maximum compatibility.

## API Reference

### Functions

- `render_markdown(text: str, as_html: bool = True)`: Render markdown text
- `display_claude_response(message: Message, **kwargs)`: Display a single message
- `display_claude_stream(message_stream, **kwargs)`: Display streaming messages
- `is_notebook() -> bool`: Check if running in Jupyter

### Classes

- `NotebookDisplay`: Main class for customized message display
  - `display_message(message: Message)`: Display a message with formatting

### Helper Functions

- `markdown_to_html(text: str) -> str`: Convert markdown to HTML
- `format_code_block(code: str, language: str) -> str`: Format code with syntax highlighting