# 海外网关 (Overseas Gateway)

## 架构

```
Cloudflare WARP (宿主机 warp-svc)
  └── SOCKS5 :40000
       └── gost (容器 overseas-gateway:full)
            ├── HTTP  :8083 (Docker/系统代理)
            └── SOCKS5 :1081 (通用 SOCKS5)
```

## 快速恢复

1. `sudo systemctl start warp-svc`
2. `warp-cli --accept-tos registration new && warp-cli --accept-tos mode proxy && warp-cli --accept-tos proxy port 40000 && warp-cli --accept-tos connect`
3. `docker build -t overseas-gateway:full .`
4. `docker run -d --name overseas-gateway --network host --entrypoint gost overseas-gateway:full -L "http://:8083" -L "socks5://:1081" -F "socks5://127.0.0.1:40000"`
5. 配置 Docker systemd proxy → `:8083`

详细步骤见 Hermes skill: `overseas-gateway-recovery`
