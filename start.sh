     #!/bin/bash
# 启动学生行为识别系统服务

cd "$(dirname "$0")"

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在，请先运行 ./deploy.sh"
    exit 1
fi

# 激活虚拟环境并启动
source venv/bin/activate

echo "正在启动学生行为识别系统..."
echo "服务地址: http://0.0.0.0:5000"
echo "按 Ctrl+C 停止服务"
echo ""

python app.py
