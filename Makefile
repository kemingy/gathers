format-rust:
	@cargo +nightly fmt
	@cd python
	@cargo +nightly fmt

format-python:
	@ruff check --fix python

format: format-rust format-python

lint-rust:
	@cargo +nightly fmt --check
	@cargo clippy -- -D warnings
	@cd python
	@cargo +nightly fmt --check
	@cargo clippy -- -D warnings

lint-python:
	@ruff check python

lint: lint-rust lint-python

test-rust:
	@cargo test --verbose

install-python:
	@cd python && pip install -e .

test-python: install-python
	@pytest -v python/tests

test: test-rust test-python
