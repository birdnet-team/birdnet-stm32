# Contributing

For the full contribution guide, see [CONTRIBUTING.md](https://github.com/birdnet-team/birdnet-stm32/blob/master/CONTRIBUTING.md)
in the repository root.

This page covers developer-specific workflow details.

## Development setup

```bash
git clone https://github.com/birdnet-team/birdnet-stm32.git
cd birdnet-stm32
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Branch workflow

1. Create a feature branch from `master`:
   ```bash
   git checkout -b my-feature
   ```
2. Make focused changes — one semantic unit per commit.
3. Run tests and linting:
   ```bash
   pytest
   ruff check . && ruff format --check .
   ```
4. Open a PR against `master`.

## Commit messages

One-line, imperative mood:

- `Add lme pooling to evaluation pipeline`
- `Fix channel alignment in hybrid frontend`

Do not use multi-line messages, WIP commits, or vague descriptions.

## Code style

- All code, comments, documentation, and commit messages in **American English**.
- Ruff handles linting and formatting. Pre-commit hooks enforce this automatically.
- Add docstrings to all public functions and classes.

## AI-assisted contributions

AI tools (Copilot, ChatGPT, etc.) are welcome. Rules:

- **Small, focused PRs.** Do not submit large generated dumps.
- **Review every line.** You are responsible for the code you submit.
- **Test everything.** AI-generated code must pass the test suite.
- **Attribute honestly.** Mention AI assistance in the PR description.

## Building docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview.
