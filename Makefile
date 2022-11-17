SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

install:  # Install the app locally
	poetry install
.PHONY: install

ci: typecheck lint test ## Run all checks (test, lint, typecheck)
.PHONY: ci

test:  ## Run tests
	poetry run pytest .
.PHONY: test

lint:  ## Run linting
	poetry run black --check apollo
	poetry run isort -c apollo
	poetry run flake8 apollo
	# poetry run pydocstyle apollo
.PHONY: lint

lint-fix:  ## Run autoformatters
	poetry run black apollo
	poetry run isort apollo
.PHONY: lint-fix

typecheck:  ## Run typechecking
	poetry run mypy --show-error-codes --pretty apollo
.PHONY: typecheck

.DEFAULT_GOAL := help
help: Makefile
	@grep -E '(^[a-zA-Z_-]+:.*?##.*$$)|(^##)' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}' | sed -e 's/\[32m##/[33m/'