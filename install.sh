#!/bin/bash
# Install SHM as a systemd service
# Usage:
#   System-level (root, multi-user server): sudo bash install.sh
#   User-level   (no root, desktop):        bash install.sh --user
#
# User-level mode installs to ~/.config/systemd/user/ with:
#   - Restart=always (crash auto-restart)
#   - linger enabled (start at boot without login)
#   - GraphLite SDK env vars (defaults to $HOME/GraphLite)
#   - GPU embedding via SHM_EMBEDDING__DEVICE=cuda (override with env)
#
# Environment overrides:
#   INSTALL_DIR   - repo location (default: script dir)
#   SERVICE_NAME  - unit name (default: shm / shm-server)
#   GRAPHLITE_BINDINGS / GRAPHLITE_SDK - GraphLite SDK paths
#   SHM_EMBEDDING__DEVICE - cpu|cuda (default: cuda for user mode, cpu for system mode)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"

USER_MODE=0
if [ "${1:-}" = "--user" ]; then
    USER_MODE=1
fi

if [ "$USER_MODE" = "1" ]; then
    SERVICE_NAME="${SERVICE_NAME:-shm-server}"
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    ENV_DEVICE="${SHM_EMBEDDING__DEVICE:-cuda}"
    GL_BINDINGS="${GRAPHLITE_BINDINGS:-$HOME/GraphLite/bindings/python}"
    GL_SDK="${GRAPHLITE_SDK:-$HOME/GraphLite/sdk-python/src}"

    echo "=== Installing SHM systemd USER service ==="
    echo "  Unit dir: $UNIT_DIR"
    mkdir -p "$UNIT_DIR"

    cat > "$UNIT_DIR/$SERVICE_NAME.service" <<EOF
[Unit]
Description=SHM — Self-evolving Hypergraph Memory (GraphLite engine)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 run_server.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10

# GraphLite SDK paths
Environment=GRAPHLITE_BINDINGS=$GL_BINDINGS
Environment=GRAPHLITE_SDK=$GL_SDK
# Local dev: skip Bearer token auth (set false in production)
Environment=DEV_MODE=true
# Embedding device: cpu for no-GPU, cuda for GPU acceleration
Environment=SHM_EMBEDDING__DEVICE=$ENV_DEVICE

# Resource limits
MemoryHigh=2G
MemoryMax=3G
CPUQuota=300%

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME"
    systemctl --user start "$SERVICE_NAME"

    # Enable linger so the service survives logout / starts at boot
    if ! loginctl show-user "$(id -un)" | grep -q "Linger=yes"; then
        echo "  Enabling linger for $(id -un)..."
        loginctl enable-linger "$(id -un)" 2>/dev/null || echo "  (needs: sudo loginctl enable-linger $(id -un))"
    fi

    echo "=== Done ==="
    echo "  systemctl --user status $SERVICE_NAME"
    echo "  journalctl --user -u $SERVICE_NAME -f"
else
    SERVICE_NAME="${SERVICE_NAME:-shm}"
    if [ "$(id -u)" != "0" ]; then
        echo "System mode requires root: sudo bash install.sh"
        echo "Or use user mode: bash install.sh --user"
        exit 1
    fi

    echo "=== Installing SHM systemd system service ==="

    # Copy service file (模板路径 /opt/shm 替换为实际 INSTALL_DIR)
    sed "s|/opt/shm|${INSTALL_DIR}|g" "${INSTALL_DIR}/shm.service" \
        > "/etc/systemd/system/${SERVICE_NAME}.service"

    # Ensure GraphLite SDK env vars are set in the unit
    GL_BINDINGS="${GRAPHLITE_BINDINGS:-${INSTALL_DIR}/GraphLite/bindings/python}"
    GL_SDK="${GRAPHLITE_SDK:-${INSTALL_DIR}/GraphLite/sdk-python/src}"
    if ! grep -q "GRAPHLITE_BINDINGS" "/etc/systemd/system/${SERVICE_NAME}.service"; then
        sed -i "/EnvironmentFile/a Environment=GRAPHLITE_BINDINGS=${GL_BINDINGS}\nEnvironment=GRAPHLITE_SDK=${GL_SDK}" \
            "/etc/systemd/system/${SERVICE_NAME}.service"
    fi

    # Create .env if not exists
    if [ ! -f "${INSTALL_DIR}/.env" ]; then
        echo "# SHM environment variables" > "${INSTALL_DIR}/.env"
        echo "# Add: DEEPSEEK_API_KEY=sk-..." >> "${INSTALL_DIR}/.env"
        echo "Created ${INSTALL_DIR}/.env (add API keys there)"
    fi

    # Reload, enable, start
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"
    systemctl start "${SERVICE_NAME}"

    echo "=== Done ==="
    echo "  systemctl status ${SERVICE_NAME}"
    echo "  journalctl -u ${SERVICE_NAME} -f"
fi
