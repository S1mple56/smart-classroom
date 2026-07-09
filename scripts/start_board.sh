#!/bin/bash
# ============================================================
# RK3588 开发板 — 课堂智能监测系统 启动脚本
# 用法: ./start_board.sh [--real] [--port 5050]
#   --real   启用真实 GPIO 控制（默认模拟模式）
#   --port   指定端口（默认 5050）
# ============================================================

cd "$(dirname "$0")"

# 默认参数
PORT=5050
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --real)  EXTRA_ARGS="$EXTRA_ARGS --real"; shift ;;
        --port)  PORT="$2"; shift 2 ;;
        *)       echo "未知参数: $1"; exit 1 ;;
    esac
done

# 获取本机 IP
BOARD_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$BOARD_IP" ] && BOARD_IP="localhost"

echo "========================================"
echo "  课堂智能监测系统 (RK3588)"
echo "========================================"
echo "  地址: http://${BOARD_IP}:${PORT}"
echo "  账号: admin  密码: admin123"
echo "  模式: $([ -n "$EXTRA_ARGS" ] && echo '真实GPIO' || echo '模拟模式')"
echo "========================================"
echo "  按 Ctrl+C 停止服务"
echo "========================================"
echo ""

# 设置环境变量
export PORT=$PORT

# 启动服务
python3 board_app.py $EXTRA_ARGS
