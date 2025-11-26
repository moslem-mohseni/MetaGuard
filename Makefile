# MetaGuard Makefile
# Author: Moslem Mohseni
# Repository: https://github.com/moslem-mohseni/MetaGuard

.PHONY: help install install-dev test lint format type-check clean build docs serve

# Default target
help:
	@echo "MetaGuard Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package in production mode"
	@echo "  make install-dev    Install package in development mode"
	@echo ""
	@echo "Quality:"
	@echo "  make test           Run tests with coverage"
	@echo "  make lint           Run linting (ruff)"
	@echo "  make format         Format code (ruff format)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "Build:"
	@echo "  make build          Build package"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"
	@echo ""
	@echo "Training:"
	@echo "  make generate-data  Generate training data"
	@echo "  make train          Train the model"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src/metaguard --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Linting and Formatting
lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

format-check:
	ruff format src/ tests/ --check

# Type Checking
type-check:
	mypy src/metaguard

# All quality checks
check: lint format-check type-check test

# Security
security:
	bandit -r src/ -c pyproject.toml
	safety check

# Build
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

docs-clean:
	cd docs && make clean

# Training
generate-data:
	python scripts/generate_data.py

train:
	python scripts/train.py

# Development
dev-setup: install-dev
	@echo "Development environment ready!"

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Version
version:
	@python -c "import metaguard; print(metaguard.__version__)"

# Docker
docker-build:
	docker build -t metaguard:latest .

docker-run:
	docker run -p 8000:8000 --rm metaguard:latest

docker-dev:
	docker-compose --profile dev up --build

docker-prod:
	docker-compose up --build -d

docker-stop:
	docker-compose down

# API Server
serve:
	uvicorn metaguard.api.rest:app --reload --host 0.0.0.0 --port 8000

# Benchmarking
benchmark:
	python scripts/benchmark.py

# Release
release-patch:
	bump2version patch
	git push && git push --tags

release-minor:
	bump2version minor
	git push && git push --tags

release-major:
	bump2version major
	git push && git push --tags
