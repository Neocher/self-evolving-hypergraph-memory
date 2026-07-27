# 境外网关 Docker 沙箱容器 — 全栈版
# 所有翻墙通道全部容器内运行，宿主机零依赖
# 包含：WARP客户端 + gost + GTProxy + 通道管理

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg lsb-release ca-certificates python3 netcat \
    openssh-client git \
    && rm -rf /var/lib/apt/lists/*

# 安装 cloudflare-warp 客户端
RUN curl -sL https://pkg.cloudflareclient.com/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/cloudflare-warp.list && \
    apt-get update && apt-get install -y --no-install-recommends cloudflare-warp && \
    rm -rf /var/lib/apt/lists/*

# 安装 gost
COPY gost /usr/local/bin/gost
RUN chmod +x /usr/local/bin/gost

# 安装 gh (GitHub CLI)
COPY gh /usr/local/bin/gh
RUN chmod +x /usr/local/bin/gh

# 通道模块
COPY tunnels/ /tunnels/
RUN chmod +x /tunnels/*.sh 2>/dev/null || true

# GTProxy Python脚本
COPY gtproxy.py /opt/gtproxy.py

# 启动入口
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

EXPOSE 1080 8080 40000 8084 9099

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD /healthcheck.sh

ENTRYPOINT ["/entrypoint.sh"]
