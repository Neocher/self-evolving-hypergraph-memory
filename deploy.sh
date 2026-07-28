#!/usr/bin/env bash
# SHM v5.14 一键部署
# 用法: curl -fsSL https://raw.githubusercontent.com/Neocher/self-evolving-hypergraph-memory/main/deploy.sh | bash
set -euo pipefail

# ─── 颜色 ───
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }

# ─── 前置检查 ───
info "检查系统依赖..."

if ! command -v docker &>/dev/null; then
    error "Docker 未安装。请先安装 Docker: https://docs.docker.com/engine/install/"
    exit 1
fi
ok "Docker $(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)"

if ! docker compose version &>/dev/null; then
    error "Docker Compose v2 未安装。请升级 Docker: https://docs.docker.com/compose/install/"
    exit 1
fi
COMPOSE_VER=$(docker compose version --short 2>/dev/null || echo "v2")
ok "Docker Compose $COMPOSE_VER"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── .env ───
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn "已从 .env.example 生成 .env"
        echo -e "${YELLOW}  → 请编辑 .env 填入 DEEPSEEK_API_KEY 后重新运行本脚本${NC}"
        echo -e "${YELLOW}  → 或直接设置环境变量覆盖: DEEPSEEK_API_KEY=xxx bash deploy.sh${NC}"
        echo
        exit 0
    else
        warn "未找到 .env 或 .env.example，将使用环境变量（如有）"
    fi
else
    ok ".env 已存在"
fi

# ─── 构建 ───
info "构建 Docker 镜像..."
docker compose build --pull
ok "镜像构建完成"

# ─── 启动 ───
info "启动服务..."
docker compose up -d
ok "容器已启动"

# ─── 健康检查轮询 ───
info "等待服务就绪..."
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo
        ok "服务健康检查通过"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo
        error "服务未能按时启动，请检查日志: docker compose logs shm"
        exit 1
    fi
    printf "."
    sleep 2
done

# ─── 完成 ───
echo
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  SHM v5.14 部署成功！${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "  ${CYAN}API:${NC}       http://localhost:8000"
echo -e "  ${CYAN}Health:${NC}    http://localhost:8000/health"
echo -e "  ${CYAN}Docs:${NC}      http://localhost:8000/docs"
echo -e "  ${CYAN}MCP:${NC}       http://localhost:8222"
echo -e "${GREEN}────────────────────────────────────────────────${NC}"
echo -e "  查看日志: ${CYAN}docker compose logs -f shm${NC}"
echo -e "  停止服务: ${CYAN}docker compose down${NC}"
echo -e "${GREEN}────────────────────────────────────────────────${NC}"
