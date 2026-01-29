.PHONY: install install-dev install-train install-all clean test lint format check serve help

PYTHON := python3
PIP := $(PYTHON) -m pip

# Default target
help:
	@echo "SwarmCode - Local-first coding agent training and runtime"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install core dependencies"
	@echo "  make install-dev    Install with development tools"
	@echo "  make install-train  Install with training dependencies"
	@echo "  make install-all    Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter"
	@echo "  make format         Format code"
	@echo "  make check          Run all checks"
	@echo ""
	@echo "Usage:"
	@echo "  make serve          Start the API server"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Quickstart:"
	@echo "  make install"
	@echo "  swarmcode ingest ./your-repo"
	@echo "  swarmcode synth ./data/snapshots/your-repo"
	@echo "  swarmcode train"
	@echo "  swarmcode serve"

# Installation targets
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-train:
	$(PIP) install -e ".[train]"

install-parser:
	$(PIP) install -e ".[parser]"

install-all:
	$(PIP) install -e ".[all]"

# Development targets
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=swarmcode_cli --cov=swarmcode_agent --cov=swarmcode_trainer --cov=swarmcode_registry --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy swarmcode-cli swarmcode-agent swarmcode-trainer swarmcode-registry

check: lint typecheck test

# Usage targets
serve:
	swarmcode serve

serve-dev:
	swarmcode serve --port 8000 -v

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-data:
	rm -rf data/
	rm -rf artifacts/
	rm -rf workspace/

# Example workflow
example-workflow:
	@echo "Running example workflow..."
	mkdir -p data artifacts workspace
	swarmcode ingest https://github.com/tiangolo/fastapi.git -o ./data/snapshots
	swarmcode synth ./data/snapshots/fastapi -o ./data/training.jsonl
	@echo "Dataset generated at ./data/training.jsonl"
	@echo "To train: swarmcode train --config configs/train.yaml"
