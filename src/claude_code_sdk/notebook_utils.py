"""Jupyter notebook utilities for Claude Code SDK.

This module provides utilities for enhanced display of Claude responses in Jupyter notebooks,
including markdown rendering, syntax highlighting, and interactive displays.
"""

import re
from typing import Any, Union
from dataclasses import dataclass

try:
    from IPython.display import display, HTML, Markdown, Code
    from IPython import get_ipython
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

from .types import (
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    Message
)


def is_notebook() -> bool:
    """Check if code is running in a Jupyter notebook.
    
    Returns:
        bool: True if running in Jupyter notebook, False otherwise
    """
    if not IPYTHON_AVAILABLE:
        return False
    
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Probably standard Python interpreter


def render_markdown(text: str, as_html: bool = True) -> None:
    """Render markdown text in a Jupyter notebook.
    
    Args:
        text: Markdown text to render
        as_html: If True, converts markdown to HTML for better styling
    """
    if not IPYTHON_AVAILABLE or not is_notebook():
        print(text)
        return
    
    if as_html:
        # Convert markdown to HTML with enhanced styling
        display(HTML(f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    line-height: 1.6; color: #333; padding: 10px;">
            {markdown_to_html(text)}
        </div>
        """))
    else:
        display(Markdown(text))


def markdown_to_html(text: str) -> str:
    """Convert markdown to HTML with basic styling.
    
    This is a simple converter for common markdown elements.
    For full markdown support, consider using a library like markdown2.
    
    Args:
        text: Markdown text
        
    Returns:
        str: HTML representation
    """
    html = text
    
    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
    
    # Italic
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
    
    # Code blocks
    html = re.sub(r'```(\w+)?\n(.*?)\n```', lambda m: format_code_block(m.group(2), m.group(1)), html, flags=re.DOTALL)
    
    # Inline code
    html = re.sub(r'`(.+?)`', r'<code style="background: #f4f4f4; padding: 2px 4px; border-radius: 3px;">\1</code>', html)
    
    # Links
    html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" style="color: #0066cc; text-decoration: none;">\1</a>', html)
    
    # Lists
    html = re.sub(r'^\* (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^\- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    
    # Wrap consecutive list items
    html = re.sub(r'(<li>.*?</li>\n)+', lambda m: '<ul style="margin: 10px 0; padding-left: 20px;">\n' + m.group(0) + '</ul>\n', html, flags=re.DOTALL)
    
    # Paragraphs
    html = re.sub(r'\n\n', '</p><p>', html)
    html = f'<p>{html}</p>'
    
    # Clean up empty paragraphs
    html = re.sub(r'<p>\s*</p>', '', html)
    
    return html


def format_code_block(code: str, language: str = None) -> str:
    """Format a code block with syntax highlighting.
    
    Args:
        code: Code content
        language: Programming language for syntax highlighting
        
    Returns:
        str: Formatted HTML code block
    """
    # Basic syntax highlighting colors for common languages
    style = """
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
    overflow-x: auto;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.4;
    """
    
    if language == 'python':
        # Escape HTML entities first
        code = code.replace('&', '&amp;')
        code = code.replace('<', '&lt;')
        code = code.replace('>', '&gt;')
        
        # Simple Python syntax highlighting
        code = re.sub(r'\b(def|class|import|from|return|if|else|elif|for|while|try|except|with|as|None|True|False)\b', 
                     r'<span style="color: #d73a49; font-weight: bold;">\1</span>', code)
        code = re.sub(r'#.*$', r'<span style="color: #6a737d; font-style: italic;">\g<0></span>', code, flags=re.MULTILINE)
        code = re.sub(r'"([^"]*)"', r'<span style="color: #032f62;">&quot;\1&quot;</span>', code)
        code = re.sub(r"'([^']*)'", r'<span style="color: #032f62;">&#39;\1&#39;</span>', code)
    
    return f'<pre style="{style}"><code>{code}</code></pre>'


@dataclass
class NotebookDisplay:
    """Enhanced display options for Claude messages in notebooks."""
    
    show_tool_use: bool = True
    show_tool_results: bool = True
    render_markdown: bool = True
    syntax_highlight: bool = True
    max_text_length: int = None
    
    def display_message(self, message: Message) -> None:
        """Display a Claude message with enhanced formatting.
        
        Args:
            message: The message to display
        """
        if not is_notebook():
            print(message)
            return
        
        if isinstance(message, AssistantMessage):
            self._display_assistant_message(message)
        else:
            # For other message types, use default string representation
            print(f"{type(message).__name__}: {message}")
    
    def _display_assistant_message(self, message: AssistantMessage) -> None:
        """Display an assistant message with rich formatting."""
        html_parts = []
        
        for block in message.content:
            if isinstance(block, TextBlock):
                text = block.text
                if self.max_text_length and len(text) > self.max_text_length:
                    text = text[:self.max_text_length] + "..."
                
                if self.render_markdown:
                    html_parts.append(markdown_to_html(text))
                else:
                    html_parts.append(f"<p>{text}</p>")
            
            elif isinstance(block, ToolUseBlock) and self.show_tool_use:
                html_parts.append(f"""
                <div style="background: #e8f4fd; border-left: 4px solid #0066cc; padding: 10px; margin: 10px 0;">
                    <strong>🔧 Tool Use:</strong> {block.name}<br>
                    <small style="color: #666;">ID: {block.id}</small>
                    {self._format_tool_input(block.input)}
                </div>
                """)
            
            elif isinstance(block, ToolResultBlock) and self.show_tool_results:
                status = "❌ Error" if block.is_error else "✅ Success"
                html_parts.append(f"""
                <div style="background: {'#fee' if block.is_error else '#efe'}; 
                            border-left: 4px solid {'#c00' if block.is_error else '#0c0'}; 
                            padding: 10px; margin: 10px 0;">
                    <strong>{status}</strong><br>
                    <pre style="margin-top: 5px; white-space: pre-wrap;">{block.content}</pre>
                </div>
                """)
        
        if html_parts:
            display(HTML('<div>' + ''.join(html_parts) + '</div>'))
    
    def _format_tool_input(self, input_data: dict) -> str:
        """Format tool input data for display."""
        if not input_data:
            return ""
        
        formatted_items = []
        for key, value in input_data.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            formatted_items.append(f"<li><strong>{key}:</strong> {value}</li>")
        
        return f'<ul style="margin-top: 5px; font-size: 0.9em;">{" ".join(formatted_items)}</ul>'


# Convenience functions
def display_claude_response(message: Union[Message, str], **kwargs) -> None:
    """Display a Claude response with enhanced formatting in Jupyter notebooks.
    
    Args:
        message: The message to display (can be a Message object or string)
        **kwargs: Additional options passed to NotebookDisplay
    """
    if isinstance(message, str):
        render_markdown(message)
    else:
        display = NotebookDisplay(**kwargs)
        display.display_message(message)


async def display_claude_stream(message_stream, **display_kwargs) -> list[Message]:
    """Display Claude messages as they stream in a Jupyter notebook.
    
    Args:
        message_stream: Async iterator of messages
        **display_kwargs: Options passed to NotebookDisplay
        
    Returns:
        list: All messages received
    """
    display = NotebookDisplay(**display_kwargs)
    messages = []
    
    async for message in message_stream:
        messages.append(message)
        if isinstance(message, AssistantMessage):
            display.display_message(message)
    
    return messages