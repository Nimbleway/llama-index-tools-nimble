.PHONY: test lint format build help

TEST_FILE ?= tests/

test:
	uv run --group test pytest -v $(TEST_FILE)

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .
	uv run --group typing mypy llama_index/

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --select I --fix .

build:
	uv build

help:
	@echo 'Commands:'
	@echo '  make test    - run unit tests'
	@echo '  make lint    - ruff check + format check + mypy'
	@echo '  make format  - auto-format and fix imports'
	@echo '  make build   - build sdist + wheel'
