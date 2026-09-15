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

lint-ast:
	@ast-grep scan $(AST_GREP_SCAN_ARGS)

test-ast:
	@for rule in ast-grep/rules/*.yml; do \
		name="$$(basename "$$rule" .yml)"; \
		test_file="ast-grep/tests/$$(basename "$$rule")"; \
		test -f "$$test_file" || { echo "missing ast-grep test: $$test_file"; exit 1; }; \
		rule_id="$$(sed -n 's/^id: //p' "$$rule")"; \
		test_id="$$(sed -n 's/^id: //p' "$$test_file")"; \
		test "$$rule_id" = "$$name" || { echo "ast-grep rule ID does not match filename: $$rule"; exit 1; }; \
		test "$$test_id" = "$$name" || { echo "ast-grep test ID does not match filename: $$test_file"; exit 1; }; \
	done
	@for test_file in ast-grep/tests/*.yml; do \
		rule="ast-grep/rules/$$(basename "$$test_file")"; \
		test -f "$$rule" || { echo "missing ast-grep rule: $$rule"; exit 1; }; \
	done
	@ast-grep test --skip-snapshot-tests

lint-python:
	@ruff check python

lint: lint-rust lint-python

test-rust:
	@cargo test --verbose

install-python:
	@cd python && pip install -e .

test-python: install-python
	@pytest -v -s python/tests

test: test-rust test-python
