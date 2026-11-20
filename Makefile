.PHONY: install install-dev test cov lint format clean

install:
	pip install -r requirements.txt

# 开发/测试环境：含 pytest / pytest-asyncio / rdflib / ruff 与 ONNX 快路径
install-dev:
	pip install -e ".[test,onnx]"

test:
	pytest -q tests/

cov:
	pytest -q tests/ --cov=api --cov=core --cov=retrieval --cov=graph --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
