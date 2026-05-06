.PHONY: install verify lint type-check test notebooks-run notebooks-clean clean help

help:
	@echo "Targets:"
	@echo "  install         - install package with all extras"
	@echo "  verify          - run lint + type-check + test + notebooks"
	@echo "  lint            - ruff check src/ tests/ + notebooks"
	@echo "  type-check      - mypy src/"
	@echo "  test            - pytest tests/"
	@echo "  notebooks-run   - papermill execute all notebooks"
	@echo "  notebooks-clean - strip outputs from notebooks"
	@echo "  clean           - remove caches and executed notebooks"

install:
	pip install -e ".[all]"

verify: lint type-check test notebooks-run

lint:
	ruff check src/ tests/
	@if command -v nbqa >/dev/null 2>&1; then nbqa ruff notebooks/; else echo "nbqa not installed, skipping notebook lint"; fi

type-check:
	mypy src/

test:
	pytest tests/

notebooks-run:
	mkdir -p notebooks_executed
	@for nb in notebooks/[0-9]*.ipynb notebooks/[a-z]*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "Running $$nb..."; \
			papermill "$$nb" "notebooks_executed/$$(basename $$nb)" \
				--kernel python3 || exit 1; \
		fi \
	done

notebooks-clean:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf notebooks_executed/
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
