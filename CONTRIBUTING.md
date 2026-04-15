# Contributing to BirdNet-STM32

Thank you for your interest in contributing! This document explains how to get
started and what we expect from contributions.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. Create a feature branch from `master`:

   ```bash
   git checkout -b my-feature
   ```

## Development Workflow

### Language

All code, comments, documentation, and commit messages must be in **American
English**.

### Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Run `ruff check .` and `ruff format --check .` before committing.
- Pre-commit hooks enforce this automatically.

### Documentation

- Add docstrings to all public functions and classes.
- Update relevant docs under `docs/` when behavior changes.
- Keep the README concise; detailed content goes in MkDocs pages.

### Testing

- Add tests for new functionality in `tests/`.
- Run the test suite: `pytest`
- Aim for unit tests on all public APIs and integration tests for
  end-to-end workflows (train, convert, evaluate).

### Commits

- **One semantic unit per commit.** Do not mix unrelated changes.
- **One-line commit messages** in imperative mood:
  - Good: `Add lme pooling to evaluation pipeline`
  - Good: `Fix channel alignment in hybrid frontend`
  - Bad: `Updated some stuff`
  - Bad: `WIP`

### Pull Requests

1. Ensure all tests pass and linting is clean.
2. Write a clear PR description explaining *what* and *why*.
3. Reference any related issues (e.g., `Closes #42`).
4. Keep PRs focused; split large changes into smaller PRs.

## Reporting Issues

- Use GitHub Issues for bugs, feature requests, and questions.
- Include reproduction steps, expected vs. actual behavior, and environment
  details (OS, Python version, TF version, hardware).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please
read it before participating.
