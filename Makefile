.PHONY: help install install-dev install-docs lint format test test-unit test-integration \
       train convert evaluate deploy docs docs-serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────

install: ## Install the package
	pip install .

install-dev: ## Install with dev + docs extras
	pip install -e ".[dev,docs]"

install-docs: ## Install docs extras only
	pip install -e ".[docs]"

# ── Quality ────────────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check birdnet_stm32/ tests/

format: ## Run ruff formatter
	ruff format birdnet_stm32/ tests/

format-check: ## Check formatting without changing files
	ruff format --check birdnet_stm32/ tests/

typecheck: ## Run mypy type checking
	mypy birdnet_stm32/

# ── Tests ──────────────────────────────────────────────────────────────────────

test: ## Run all tests
	python -m pytest tests/ -v

test-unit: ## Run unit tests only
	python -m pytest tests/ -v -m "not integration"

test-integration: ## Run integration tests only
	python -m pytest tests/ -v -m "integration"

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --cov=birdnet_stm32 --cov-report=term-missing

# ── Pipeline ───────────────────────────────────────────────────────────────────

train: ## Train a model (pass ARGS="..." for extra arguments)
	python train.py $(ARGS)

convert: ## Convert model to TFLite (pass ARGS="..." for extra arguments)
	python convert.py $(ARGS)

evaluate: ## Evaluate a model (pass ARGS="..." for extra arguments)
	python test.py $(ARGS)

deploy: ## Deploy to STM32N6570-DK
	bash deploy.sh

# ── Documentation ──────────────────────────────────────────────────────────────

docs: ## Build documentation
	mkdocs build --strict

docs-serve: ## Serve documentation locally
	mkdocs serve

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info site/ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
