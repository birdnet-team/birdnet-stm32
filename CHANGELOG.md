# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffolding: CODE_OF_CONDUCT, CONTRIBUTING, CITATION.cff, SECURITY, CHANGELOG
- `pyproject.toml` with dev/docs dependency groups
- Pre-commit hooks (ruff, yaml, whitespace)
- `birdnet_stm32/` Python package structure
- Test framework with pytest fixtures and synthetic audio data
- `config.example.json` replacing hardcoded paths

### Changed

- Refactored flat scripts into `birdnet_stm32/` package modules
- Replaced `deploy.sh` hardcoded paths with config resolution (env vars, config file, CLI args)
- Updated `.gitignore` for new project structure

### Removed

- Hardcoded personal paths from `deploy.sh`, `config.json`, `config_n6l.json`
