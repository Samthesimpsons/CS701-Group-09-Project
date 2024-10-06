# Usage:
# Run `make lint` to lint the codebase using flake8 and ruff.
# Run `make typing` to check type annotations using mypy.
# Run `make test` to execute tests and generate a coverage report.
# Run `make check-all` to perform linting, typing checks, and testing with coverage.

.PHONY: lint typing check-all test

lint:
	poetry run flake8 --ignore=E203,E266,W503,E501 --max-line-length=88 src tests
	poetry run ruff check src tests

typing:
	poetry run mypy src tests

test:
	poetry run pytest --cov=src --cov-report=term-missing

check-all: lint typing test
