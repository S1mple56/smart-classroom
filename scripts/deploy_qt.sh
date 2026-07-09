#!/bin/bash
# 智瞳课堂 - 一键部署 Qt 触摸屏自启动
# 在板子上运行此脚本

set -e

PROJECT_DIR="/home/elf/qiansai"
SERVICE_NAME="zhitong-classroom"

echo "========================================"
echo "  智瞳课堂 - Qt 触摸屏部署"
echo "========================================"

# 1. 安装依赖
echo "[1/4] 安装依赖..."
pip3 install PyQt5 opencv-python requests 2>/dev/null || pip install PyQt5 opencv-python requests

# 2. 设置脚本权限
echo "[2/4] 设置权限..."
chmod +x "$PROJECT_DIR/start_qt.sh"
chmod +x "$PROJECT_DIR/start_board.sh" 2>/dev/null

# 3. 安装 systemd 服务
echo "[3/4] 安装自启动服务..."
cp "$PROJECT_DIR/zhitong-classroom.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
echo "  服务已安装并启用"

# 4. 启动服务
echo "[4/4] 启动服务..."
systemctl start "$SERVICE_NAME"
echo "  服务已启动"

echo ""
echo "========================================"
echo "  部署完成!"
echo "========================================"
echo ""
echo "  查看状态: systemctl status $SERVICE_NAME"
echo "  查看日志: tail -f $PROJECT_DIR/log/stdout.log"
echo "  停止服务: systemctl stop $SERVICE_NAME"
echo "  禁用自启: systemctl disable $SERVICE_NAME"
echo ""
