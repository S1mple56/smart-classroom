#!/bin/bash
# ============================================================
# RK3588 开发板 — 一键部署脚本
# 用法: sudo ./deploy_board.sh
# ============================================================
set -e

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
CURRENT_USER=$(whoami)

echo "========================================"
echo "  RK3588 课堂智能监测系统 — 一键部署"
echo "========================================"
echo "  项目目录: ${PROJECT_DIR}"
echo ""

# ---- 1. 系统依赖 ----
echo "[1/5] 安装系统依赖..."
apt update -qq
apt install -y python3 python3-pip python3-opencv \
    v4l-utils fonts-wqy-zenhei libgl1-mesa-glx libglib2.0-0 \
    > /dev/null 2>&1
echo "  ✓ 系统依赖"

# ---- 2. Python 依赖 ----
echo "[2/5] 安装 Python 依赖..."
pip3 install flask flask-cors numpy pillow --quiet \
    -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null
echo "  ✓ Flask + 基础依赖"

pip3 install ultralytics --quiet \
    -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
    echo "  ⚠ ultralytics 安装失败，YOLO 功能不可用"
echo "  ✓ YOLO"

# ---- 3. 创建目录 ----
echo "[3/5] 创建数据目录..."
mkdir -p "${PROJECT_DIR}"/{data/upload,data/eval/students,data/temp_recognize,data/captures,log,model,rknn_convert}
chmod -R 755 "${PROJECT_DIR}"/data "${PROJECT_DIR}"/log
echo "  ✓ 目录创建完成"

# ---- 4. systemd 服务 ----
echo "[4/5] 配置开机自启..."
cat > /etc/systemd/system/face-board.service << EOF
[Unit]
Description=Campus Smart Monitor (RK3588)
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/python3 ${PROJECT_DIR}/board_app.py
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=5
Environment="PORT=5050"
StandardOutput=append:${PROJECT_DIR}/log/stdout.log
StandardError=append:${PROJECT_DIR}/log/stderr.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable face-board.service
echo "  ✓ 服务已注册，开机自启"

# ---- 5. 快捷脚本 ----
echo "[5/5] 创建快捷脚本..."
chmod +x "${PROJECT_DIR}"/start_board.sh

# 完成
BOARD_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "========================================"
echo "  部署完成！"
echo "========================================"
echo ""
echo "  启动:  ./start_board.sh"
echo "  停止:  systemctl stop face-board"
echo "  状态:  systemctl status face-board"
echo "  日志:  tail -f log/stdout.log"
echo ""
echo "  访问:  http://${BOARD_IP}:5050"
echo "  账号:  admin / admin123"
echo ""
echo "  开机自启已配置，重启板子后自动运行"
echo "========================================"
