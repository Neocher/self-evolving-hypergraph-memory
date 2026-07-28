# 使用 Aliyun 镜像加速 (Docker Hub 被 GFW 阻断)
FROM docker.m.daocloud.io/library/python:3.11-slim-bookworm AS base

LABEL description="Hermes D+F (Deep Research + Feishu) Agent Runtime"

# ===== 系统依赖（使用阿里云镜像加速）=====
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    gcc libc6-dev \
    curl git sudo procps \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ===== Go 1.25.9（国内阿里云镜像）=====
ARG GO_VERSION=1.25.9
ARG GO_TAR=go${GO_VERSION}.linux-amd64.tar.gz
RUN curl -sL --max-time 120 \
    "https://mirrors.aliyun.com/golang/${GO_TAR}" \
    -o /tmp/go.tar.gz \
    && tar xzf /tmp/go.tar.gz -C /usr/local/ \
    && rm /tmp/go.tar.gz \
    && /usr/local/go/bin/go version
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOROOT=/usr/local/go

# ===== Python Dependencies（国内镜像）=====
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir \
    hermes-agent \
    python-docx lxml \
    requests urllib3 \
    pyyaml \
    -i https://mirrors.aliyun.com/pypi/simple/

# ===== Hermes Runtime Setup =====
RUN mkdir -p /home/hermes/.hermes /home/hermes/.local/bin \
    && chmod -R 755 /home/hermes
ENV HERMES_HOME=/home/hermes/.hermes
ENV HOME=/home/hermes

# ===== 研究工具 =====
RUN python3 -c "import docx; print(f'python-docx {docx.__version__} ✅')" && \
    python3 -c "import lxml; print(f'lxml {lxml.__version__} ✅')" && \
    python3 -c "import requests; print(f'requests ✅')" && \
    /usr/local/go/bin/go version

# ===== 启动脚本 =====
RUN echo '#!/bin/bash\n\
echo "=== Hermes D+F Container ==="\n\
echo "Go: $(/usr/local/go/bin/go version 2>/dev/null)"\n\
echo "LibreOffice: $(libreoffice --version 2>/dev/null | head -1)"\n\
echo "Python: $(python3 --version 2>/dev/null)"\n\
echo "Proxy: socks5://overseas-gateway:1080"\n\
echo ""\n\
# Test proxy connectivity to overseas-gateway\n\
if nc -z overseas-gateway 1080 2>/dev/null; then\n\
    echo "✅ Proxy reachable: overseas-gateway:1080"\n\
else\n\
    host_ip=$(ip route | grep default | awk "{print \$3}")\n\
    echo "⚠️  Direct proxy check failed, trying host..."\n\
    if nc -z "$host_ip" 1081 2>/dev/null; then\n\
        echo "✅ Proxy reachable via host: $host_ip:1081"\n\
        export SOCKS5_PROXY="$host_ip:1081"\n\
    fi\n\
fi\n\
echo ""\n\
echo "Container ready. Hermes agent will be started by the entrypoint."\n\
' > /usr/local/bin/healthcheck.sh && chmod +x /usr/local/bin/healthcheck.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" && \
    libreoffice --headless --version > /dev/null 2>&1 || exit 1

WORKDIR /home/hermes
ENTRYPOINT ["tail", "-f", "/dev/null"]
