#!/bin/bash
# Install SHM as a systemd service
# Usage: sudo bash install.sh

set -e

SERVICE_NAME="shm"
INSTALL_DIR="/home/admin/shm"

if [ "$(id -u)" != "0" ]; then
    echo "Please run as root: sudo bash install.sh"
    exit 1
fi

echo "=== Installing SHM systemd service ==="

# Copy service file
cp "${INSTALL_DIR}/shm.service" /etc/systemd/system/${SERVICE_NAME}.service

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
