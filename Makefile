.PHONY: help install install-dev test test-cov lint format type-check clean build docs serve-docs websocket-demo agent-demo

# Default target
help:
	@echo "Available commands:"
	@echo "  make install         Install the package"
	@echo "  make install-dev     Install with development dependencies"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make lint            Run linting"
	@echo "  make format          Format code"
	@echo "  make type-check      Run type checking"
	@echo "  make clean           Clean build artifacts"
	@echo "  make build           Build distribution packages"
	@echo "  make docs            Build documentation"
	@echo "  make serve-docs      Serve documentation locally"
	@echo "  make websocket-demo  Run WebSocket server demo"
	@echo "  make agent-demo      Run agent system demo"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,websocket]"

install-all:
	pip install -e ".[dev,websocket]"
	cd agent_system && pip install -e ".[dev]"

# Testing targets
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=claude_code_sdk --cov-report=xml --cov-report=html --cov-report=term

test-websocket:
	python -m pytest tests/test_websocket_server.py -v

test-integration:
	python -m pytest tests/ -v -m integration

# Code quality targets
lint:
	ruff check src/ tests/
	@echo "Linting agent system..."
	cd agent_system && ruff check agents/ || true

format:
	ruff format src/ tests/
	@echo "Formatting agent system..."
	cd agent_system && ruff format agents/ || true

format-check:
	ruff format --check src/ tests/

type-check:
	mypy src/claude_code_sdk

type-check-strict:
	mypy src/claude_code_sdk --strict

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

# Documentation targets
docs:
	@echo "Building documentation..."
	@mkdir -p docs/api
	@echo "Documentation building not yet configured"

serve-docs:
	@echo "Serving documentation..."
	python -m http.server 8080 --directory docs/

# Demo targets
websocket-demo:
	@echo "Starting WebSocket server demo..."
	python examples/websocket_ui_server.py

agent-demo:
	@echo "Starting agent system demo..."
	cd agent_system && python main.py

websocket-client-demo:
	@echo "Starting WebSocket client demo..."
	python examples/websocket_client.py

# Development workflow targets
dev-setup: install-all
	@echo "Setting up development environment..."
	@echo "Installing pre-commit hooks..."
	pre-commit install || echo "pre-commit not installed, skipping..."
	@echo "Development environment ready!"

check-all: lint type-check test
	@echo "All checks passed!"

fix: format
	@echo "Code formatted!"

# Agent system targets
agent-install:
	cd agent_system && pip install -e .

agent-test:
	cd agent_system && python -m pytest tests/ -v

agent-run:
	cd agent_system && python main.py

# Release targets
version:
	@grep version pyproject.toml | head -1

bump-patch:
	@echo "Bumping patch version..."
	@echo "Please update version in pyproject.toml manually"

bump-minor:
	@echo "Bumping minor version..."
	@echo "Please update version in pyproject.toml manually"

bump-major:
	@echo "Bumping major version..."
	@echo "Please update version in pyproject.toml manually"

release: check-all build
	@echo "Ready to release!"
	@echo "1. Update version in pyproject.toml"
	@echo "2. Commit changes"
	@echo "3. Tag the release: git tag -a v0.x.x -m 'Release v0.x.x'"
	@echo "4. Push tags: git push origin --tags"
	@echo "5. Run: make upload"

# Docker targets (future)
docker-build:
	@echo "Docker support not yet implemented"

docker-run:
	@echo "Docker support not yet implemented"

# Continuous Integration helpers
ci-test:
	python -m pytest tests/ -v --cov=claude_code_sdk --cov-report=xml

ci-lint:
	ruff check src/ tests/ --output-format=github

ci-type-check:
	mypy src/claude_code_sdk --junit-xml mypy-report.xml

# Utility targets
watch-test:
	@echo "Watching for changes and running tests..."
	@echo "Install watchdog: pip install watchdog"
	watchmedo auto-restart --patterns="*.py" --recursive -- python -m pytest tests/ -v

profile:
	@echo "Running profiler..."
	python -m cProfile -o profile.stats examples/quick_start.py
	python -m pstats profile.stats

# Environment info
info:
	@echo "=== Environment Info ==="
	@echo "Python version:"
	@python --version
	@echo "\nPip version:"
	@pip --version
	@echo "\nInstalled packages:"
	@pip list | grep -E "(claude|anyio|fastapi|uvicorn)"
	@echo "\nNode version:"
	@node --version || echo "Node.js not installed"
	@echo "\nClaude Code CLI:"
	@which claude || echo "Claude Code CLI not found"