# SHM — Self-evolving Hypergraph Memory
# 多阶段构建: 最小化最终镜像体积

FROM python:3.11-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir setuptools wheel

COPY pyproject.toml .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -e .

FROM python:3.11-slim

WORKDIR /app

# 运行时依赖（只安装必要的系统库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制 wheel
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# 复制源码
COPY . .

# 数据目录
VOLUME /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["python3", "run_server.py"]
