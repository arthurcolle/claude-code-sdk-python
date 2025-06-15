# Contributing to Claude Code SDK for Python

Thank you for your interest in contributing to the Claude Code SDK! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js (for Claude Code CLI)
- Git

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/claude-code-sdk-python.git
cd claude-code-sdk-python
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
make install-dev
# Or manually:
pip install -e ".[dev,websocket]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Install Claude Code CLI:
```bash
npm install -g @anthropic-ai/claude-code
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
python -m pytest tests/test_client.py -v

# Run WebSocket tests
make test-websocket
```

### Code Quality

```bash
# Run all checks
make check-all

# Run linting
make lint

# Format code
make format

# Type checking
make type-check
```

### Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes, following the coding standards

3. Add tests for new functionality

4. Run tests and quality checks:
```bash
make check-all
```

5. Commit your changes:
```bash
git add .
git commit -m "feat: add new feature"
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names

### Type Hints

```python
from typing import Optional, List, Dict, Any

async def process_data(
    data: Dict[str, Any],
    options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process data with optional filters.
    
    Args:
        data: Input data to process
        options: Optional list of processing options
        
    Returns:
        Processed data dictionary
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    More detailed description if needed. Can span
    multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
    """
    ...
```

### Error Handling

```python
from claude_code_sdk import ClaudeSDKError

try:
    result = await risky_operation()
except SpecificError as e:
    # Handle specific error
    logger.error(f"Operation failed: {e}")
    raise ClaudeSDKError("User-friendly message") from e
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock

class TestFeature:
    """Test suite for specific feature."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        return AsyncMock()
    
    async def test_happy_path(self, mock_client):
        """Test normal operation."""
        # Arrange
        mock_client.query.return_value = "expected"
        
        # Act
        result = await function_under_test(mock_client)
        
        # Assert
        assert result == "expected"
        mock_client.query.assert_called_once()
    
    async def test_error_handling(self, mock_client):
        """Test error scenarios."""
        # Test specific error cases
        ...
```

### Test Coverage

- Aim for >80% code coverage
- Test both happy paths and error cases
- Include integration tests for complex features

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include usage examples in docstrings
- Document exceptions that can be raised

### Feature Documentation

When adding new features:

1. Update the README.md with usage examples
2. Add detailed documentation in `docs/`
3. Update the API reference if needed

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Build process or auxiliary tool changes

Examples:
```
feat: add WebSocket server for real-time communication
fix: handle timeout errors in subprocess transport
docs: update README with WebSocket examples
```

## Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the CHANGELOG.md if applicable
5. Submit PR with clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Project Structure

```
claude-code-sdk-python/
├── src/
│   └── claude_code_sdk/
│       ├── __init__.py          # Public API
│       ├── types.py             # Type definitions
│       ├── websocket_server.py  # WebSocket server
│       └── _internal/           # Internal implementation
├── tests/                       # Test files
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── agent_system/               # Agent system (separate package)
└── Makefile                    # Development commands
```

## Getting Help

- Check existing issues and discussions
- Ask questions in discussions
- Join our community chat (if available)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Follow the project's code of conduct

Thank you for contributing!