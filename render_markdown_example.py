"""
Quick code to render Markdown files in Jupyter Lab
"""

from IPython.display import display, Markdown, HTML
import os

def render_markdown_file(filepath):
    """Render a markdown file in Jupyter Lab"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    display(Markdown(content))

def render_markdown_string(markdown_text):
    """Render markdown string in Jupyter Lab"""
    display(Markdown(markdown_text))

def render_markdown_with_style(markdown_text):
    """Render markdown with custom CSS styling"""
    html = f"""
    <style>
        .markdown-body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }}
        .markdown-body h1, .markdown-body h2 {{
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }}
        .markdown-body code {{
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }}
        .markdown-body pre {{
            background-color: #f6f8fa;
            padding: 16px;
            overflow: auto;
            border-radius: 6px;
        }}
    </style>
    <div class="markdown-body">
    """
    display(HTML(html))
    display(Markdown(markdown_text))
    display(HTML("</div>"))

# Example usage in Jupyter:
if __name__ == "__main__":
    # Example 1: Render a markdown file
    # render_markdown_file("README.md")
    
    # Example 2: Render markdown string
    example_markdown = """
# Hello Jupyter!

This is **bold** and this is *italic*.

## Code Example
```python
def hello():
    print("Hello, World!")
```

### Lists
- Item 1
- Item 2
  - Nested item

1. First
2. Second
3. Third
"""
    
    print("Use these functions in Jupyter Lab:")
    print("1. render_markdown_file('path/to/file.md')")
    print("2. render_markdown_string(markdown_text)")
    print("3. render_markdown_with_style(markdown_text)")