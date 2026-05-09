.PHONY: install test lint format clean

install:
	pip install -r requirements.txt

test:
	pytest --cov=shm --cov-report=term-missing tests/

lint:
	ruff check .

format:
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
