#!/bin/bash
# 境外网关全栈启动脚本
# 容器内启动 WARP + gost + GTProxy

set -e

WARP_LICENSE="${WARP_LICENSE:-}"
WARP_SOCKS_PORT="${WARP_SOCKS_PORT:-40000}"

echo "=========================================="
echo "  🌐 境外网关 (全容器版)"
echo "  所有通道在容器内，宿主机零依赖"
echo "=========================================="
mkdir -p /tmp/gateway /var/run

# ===== 1. 启动 WARP =====
echo ""
echo "[gateway] 🟢 启动 WARP 客户端..."

# 如果 warp-svc 还没运行
if ! pgrep warp-svc > /dev/null 2>&1; then
    echo "[gateway] 启动 warp-svc 守护进程..."
    /usr/bin/warp-svc &
    WARP_PID=$!
    echo "[gateway] warp-svc PID: $WARP_PID"
    sleep 3
else
    echo "[gateway] warp-svc 已在运行"
fi

# 等待 WARP 服务就绪
for i in $(seq 1 15); do
    if warp-cli --accept-tos status > /dev/null 2>&1; then
        echo "[gateway] ✅ WARP 服务就绪 (${i}s)"
        break
    fi
    sleep 2
done

# 注册/连接
WARP_STATUS=$(warp-cli --accept-tos status 2>&1 | grep -o "Connected\|Disconnected\|Connecting")
echo "[gateway] WARP 状态: $WARP_STATUS"

if [ "$WARP_STATUS" != "Connected" ]; then
    echo "[gateway] 注册 WARP (License: ${WARP_LICENSE:0:12}...)"
    warp-cli --accept-tos registration new > /dev/null 2>&1 || true
    if [ -n "$WARP_LICENSE" ]; then
        warp-cli --accept-tos registration license "$WARP_LICENSE" > /dev/null 2>&1 || true
    fi
    echo "[gateway] 设置代理模式..."
    warp-cli --accept-tos mode proxy > /dev/null 2>&1 || true
    warp-cli --accept-tos proxy port "$WARP_SOCKS_PORT" > /dev/null 2>&1 || true
    echo "[gateway] 连接..."
    warp-cli --accept-tos connect > /dev/null 2>&1 || true
    sleep 5
fi

# 最终验证
if warp-cli --accept-tos status 2>&1 | grep -q "Connected"; then
    echo "[gateway] ✅ WARP 已连接，SOCKS5 端口: $WARP_SOCKS_PORT"
else
    echo "[gateway] ⚠️ WARP 未连接，状态:"
    warp-cli --accept-tos status 2>&1
fi

# ===== 2. 启动 gost（统一代理入口） =====
echo ""
echo "[gateway] 🔵 启动 gost..."
gost \
  -L "socks5://:1080" \
  -L "http://:8080" \
  -F "socks5://127.0.0.1:${WARP_SOCKS_PORT}" &
GOST_PID=$!
echo "[gateway] gost PID: $GOST_PID"
sleep 2

# ===== 3. 启动 GTProxy（BBC/AJ/NHK穿透） =====
echo ""
echo "[gateway] 🟢 启动 GTProxy..."
python3 /opt/gtproxy.py &
GTPID=$!
echo "[gateway] GTProxy PID: $GTPID"

# ===== 4. 状态服务器 =====
echo ""
echo "[gateway] 📊 启动状态服务器 :9099..."
python3 << 'PYEOF' &
import http.server, json, os, subprocess

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            warp_ok = False
            try:
                r = subprocess.run(['warp-cli','--accept-tos','status'],
                    capture_output=True, text=True, timeout=5)
                warp_ok = 'Connected' in r.stdout
            except: pass
            self.wfile.write(json.dumps({
                "status": "ok" if warp_ok else "degraded",
                "warp": warp_ok,
                "version": "all-in-docker",
                "gtproxy": os.path.exists('/tmp/gtproxy.pid')
            }).encode())
        elif self.path == '/channels':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            out = {"warp": "active", "gtproxy": "active", "github": "pending"}
            self.wfile.write(json.dumps(out).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Gateway OK</h1></body></html>")

http.server.HTTPServer(('0.0.0.0', 9099), H).serve_forever()
PYEOF

echo ""
echo "[gateway] ════════════════════════════════════"
echo "[gateway] ✅ 境外网关全栈启动完成"
echo "[gateway]    WARP    :${WARP_SOCKS_PORT}"
echo "[gateway]    HTTP    :8080"
echo "[gateway]    SOCKS5  :1080"
echo "[gateway]    GTProxy :8084 (bbc/aj/nhk)"
echo "[gateway]    状态    :9099"
echo "[gateway] ════════════════════════════════════"

wait $GOST_PID
