#!/bin/bash
# 停止学生行为识别系统服务

echo "正在停止服务..."

# 使用多种方式确保进程被停止
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "gunicorn.*app:app" 2>/dev/null || true

# 等待进程结束
sleep 1

# 检查是否还有进程在运行
if pgrep -f "python.*app.py" > /dev/null 2>&1; then
    echo "警告: 进程可能还在运行，强制终止..."
    pkill -9 -f "python.*app.py" 2>/dev/null || true
fi

echo "✓ 服务已停止"
