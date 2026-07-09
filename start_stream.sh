#!/bin/bash
# 启动板子摄像头视频流服务 (mjpg-streamer)
# 使用方法: ./start_stream.sh

echo "正在启动摄像头视频流服务..."

# 停止之前的视频流服务
pkill mjpg_streamer 2>/dev/null || true
sleep 1

# 检查摄像头设备是否存在
if [ ! -e /dev/video0 ]; then
    echo "错误: 未检测到摄像头设备 /dev/video0"
    echo "请检查摄像头是否已连接"
    exit 1
fi

# 检查 mjpg-streamer 是否已安装
if ! command -v mjpg_streamer &> /dev/null; then
    echo "mjpg-streamer 未安装，正在安装..."
    sudo apt update
    sudo apt install -y mjpg-streamer libv4l-dev
fi

# 获取板子 IP 地址
BOARD_IP=$(hostname -I | awk '{print $1}')

# 启动 mjpg-streamer
# -d /dev/video0: 使用 video0 设备
# -r 640x480: 分辨率
# -f 30: 帧率
# -p 8080: HTTP 端口
mjpg_streamer -i "input_uvc.so -d /dev/video0 -r 640x480 -f 30" -o "output_http.so -p 8080 -w /usr/share/mjpg-streamer/www" &

sleep 2

# 检查是否启动成功
if pgrep mjpg_streamer > /dev/null; then
    echo "✓ 视频流服务已启动"
    echo "  视频流地址: http://${BOARD_IP}:8080/?action=stream"
    echo "  单帧地址:   http://${BOARD_IP}:8080/?action=snapshot"
    echo ""
    echo "在网页考勤页面中点击'开启摄像头'即可看到板子摄像头画面"
else
    echo "✗ 视频流服务启动失败"
    echo "请检查:"
    echo "  1. 摄像头是否已连接: ls -l /dev/video*"
    echo "  2. mjpg-streamer 是否已安装: apt install mjpg-streamer"
    echo "  3. 端口 8080 是否被占用: netstat -tlnp | grep 8080"
fi