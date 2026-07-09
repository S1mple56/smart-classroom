#!/bin/bash
# 智瞳课堂 - Qt 触摸屏自启动脚本
# 等待网络连接后启动 Flask 后端 + Qt 界面

PROJECT_DIR="/home/elf/qiansai"
LOG_DIR="$PROJECT_DIR/log"
mkdir -p "$LOG_DIR"

# 等待网络就绪（最多等 60 秒）
echo "[$(date)] 等待网络连接..."
for i in $(seq 1 60); do
    if ping -c 1 -W 1 baidu.com &>/dev/null; then
        echo "[$(date)] 网络已连接"
        break
    fi
    sleep 1
done

# 设置环境变量
export SILICONFLOW_API_KEY="sk-dmhuhgiqzcdngjyhmuajbrrrwzintniushyjzuduwaokvrqx"
export DISPLAY=:0
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms

# 杀掉旧进程
pkill -f "python3.*app.py" 2>/dev/null
pkill -f qt_touch_app.py 2>/dev/null
sleep 2

# 启动 Flask 后端
echo "[$(date)] 启动 Flask 后端..."
cd "$PROJECT_DIR"
python3 app.py > "$LOG_DIR/stdout.log" 2> "$LOG_DIR/stderr.log" &
FLASK_PID=$!
echo "[$(date)] Flask PID: $FLASK_PID"

# 等待后端就绪
sleep 6

# 启动 Qt 界面
echo "[$(date)] 启动 Qt 界面..."
python3 qt_touch_app.py > "$LOG_DIR/qt.log" 2>&1 &
QT_PID=$!
echo "[$(date)] Qt PID: $QT_PID"

# 保存 PID
echo "$FLASK_PID" > "$LOG_DIR/flask.pid"
echo "$QT_PID" > "$LOG_DIR/qt.pid"

echo "[$(date)] 启动完成"
