#!/bin/bash
# Hermes D+F Docker 环境搭建向导
set -e

echo "=========================================="
echo "  Hermes D+F Docker 部署"
echo "  Deep Research + Feishu Delivery"
echo "=========================================="
echo ""

# 检查 Docker
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker 未运行或不可访问"
    exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. 创建 Docker 网络
echo "📦 步骤1: 创建 hermes-df 网络..."
docker network create hermes-df 2>/dev/null || echo "  网络已存在，跳过"

# 2. 将 overseas-gateway 连接到新网络
echo "🔗 步骤2: 连接 overseas-gateway 到 hermes-df 网络..."
if docker ps --format '{{.Names}}' | grep -q overseas-gateway; then
    docker network connect hermes-df overseas-gateway 2>/dev/null || \
        echo "  overseas-gateway 已在网络中，跳过"
    echo "  ✅ overseas-gateway 已连接到 hermes-df"
else
    echo "  ⚠️  overseas-gateway 未运行，先启动它"
    exit 1
fi

# 3. 构建镜像
echo "🔨 步骤3: 构建 hermes-df 镜像..."
echo "  (首次构建约需 5-15 分钟，取决于网络速度)"
docker compose build 2>&1 | tail -5

# 4. 验证
echo ""
echo "🔍 步骤4: 验证..."
echo "  镜像: $(docker images hermes-df --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null)"
echo ""

# 5. 测试容器
echo "🧪 步骤5: 启动测试容器..."
docker compose up -d 2>&1 | tail -3

echo ""
echo "✅ 部署完成！"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  容器状态:"
docker ps --filter name=hermes-df --format "  {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📖 使用方式:"
echo "  进入容器:  docker exec -it hermes-df bash"
echo "  查看日志:  docker logs hermes-df"
echo "  停止容器:  docker compose down"
echo "  重新构建:  docker compose build --no-cache"
echo ""
echo "🔒 安全提示:"
echo "  - 凭证文件以 ro 模式挂载（只读）"
echo "  - 容器以 no-new-privileges 运行"
echo "  - 已 drop ALL capabilities，仅加 NET_BIND_SERVICE"
echo "  - 内存限制: 8GB，CPU限制: 4核"
echo ""
