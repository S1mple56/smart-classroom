#!/bin/bash
#==============================================================================
# ELF2 RK3588 部署脚本 - 学生行为识别系统
# 使用方法: chmod +x deploy.sh && sudo ./deploy.sh
#==============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目目录（自动检测）
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)

# 用户名（自动检测当前用户）
CURRENT_USER=$(whoami)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ELF2 RK3588 一键部署脚本${NC}"
echo -e "${GREEN}  学生行为识别系统${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  项目目录: ${PROJECT_DIR}"
echo -e "  当前用户: ${CURRENT_USER}"
echo ""

# 检测是否为 root 用户
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}警告: 建议使用 sudo 运行此脚本${NC}"
    echo -e "${YELLOW}如果遇到权限问题，请使用: sudo ./deploy.sh${NC}"
    echo ""
fi

#==============================================================================
# 1. 系统依赖安装
#==============================================================================
echo -e "${GREEN}[1/6] 安装系统依赖...${NC}"
echo "----------------------------------------"

apt update

# Python 和基础工具
apt install -y python3 python3-pip python3-dev python3-venv

# 图像处理和多媒体依赖
apt install -y libjpeg-dev zlib1g-dev libpng-dev

# 摄像头支持
apt install -y v4l-utils ffmpeg

# 网络工具
apt install -y net-tools curl

echo -e "${GREEN}✓ 系统依赖安装完成${NC}"
echo ""

#==============================================================================
# 2. 创建虚拟环境
#==============================================================================
echo -e "${GREEN}[2/6] 创建 Python 虚拟环境...${NC}"
echo "----------------------------------------"

cd "$PROJECT_DIR"

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

echo -e "${GREEN}✓ 虚拟环境创建完成${NC}"
echo ""

#==============================================================================
# 3. 安装 Python 依赖
#==============================================================================
echo -e "${GREEN}[3/6] 安装 Python 依赖...${NC}"
echo "----------------------------------------"

# 安装依赖（使用国内镜像加速）
pip install -r requirements_rk3588.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo -e "${GREEN}✓ Python 依赖安装完成${NC}"
echo ""

#==============================================================================
# 4. 创建数据目录
#==============================================================================
echo -e "${GREEN}[4/6] 创建数据目录...${NC}"
echo "----------------------------------------"

mkdir -p data/upload
mkdir -p data/eval/students
mkdir -p data/temp_recognize
mkdir -p data/zipped-eval
mkdir -p data/zipped-train
mkdir -p model
mkdir -p log

# 设置权限
chmod -R 755 data
chmod -R 755 model
chmod -R 755 log

echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

#==============================================================================
# 5. 配置 systemd 服务
#==============================================================================
echo -e "${GREEN}[5/6] 配置系统服务...${NC}"
echo "----------------------------------------"

# 创建服务文件（使用当前用户和项目目录）
cat > /etc/systemd/system/face-attendance.service << EOF
[Unit]
Description=Student Face Attendance System
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=${PROJECT_DIR}/venv/bin:/usr/bin:/bin"
ExecStart=${PROJECT_DIR}/venv/bin/python app.py
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=5
StandardOutput=append:${PROJECT_DIR}/log/stdout.log
StandardError=append:${PROJECT_DIR}/log/stderr.log

[Install]
WantedBy=multi-user.target
EOF

# 重新加载 systemd
systemctl daemon-reload

# 设置开机自启
systemctl enable face-attendance.service

echo -e "${GREEN}✓ 系统服务配置完成${NC}"
echo ""

#==============================================================================
# 6. 创建快捷命令脚本
#==============================================================================
echo -e "${GREEN}[6/7] 创建快捷命令...${NC}"
echo "----------------------------------------"

# 创建 start.sh
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在，请先运行 ./deploy.sh"
    exit 1
fi
source venv/bin/activate
echo "正在启动学生行为识别系统..."
echo "服务地址: http://0.0.0.0:5000"
echo "按 Ctrl+C 停止服务"
python app.py
EOF
chmod +x start.sh

# 创建 stop.sh
cat > stop.sh << 'EOF'
#!/bin/bash
pkill -f "python app.py" || true
pkill -f "python3 app.py" || true
echo "服务已停止"
EOF
chmod +x stop.sh

# 创建 restart.sh
cat > restart.sh << 'EOF'
#!/bin/bash
./stop.sh
sleep 2
./start.sh
EOF
chmod +x restart.sh

# 创建 status.sh
cat > status.sh << 'EOF'
#!/bin/bash
if pgrep -f "python.*app.py" > /dev/null; then
    echo "✓ 服务运行中"
else
    echo "✗ 服务未运行"
fi
if pgrep mjpg_streamer > /dev/null; then
    echo "✓ 视频流服务运行中"
else
    echo "✗ 视频流服务未运行"
fi
EOF
chmod +x status.sh

# 确保 start_stream.sh 有执行权限
chmod +x start_stream.sh

echo -e "${GREEN}✓ 快捷命令创建完成${NC}"
echo ""

#==============================================================================
# 完成
#==============================================================================
BOARD_IP=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "使用方法："
echo "  ./start.sh          - 启动网页服务（含摄像头视频流）"
echo "  ./stop.sh           - 停止网页服务"
echo "  ./restart.sh        - 重启网页服务"
echo "  ./status.sh         - 查看服务状态"
echo ""
echo "或使用 systemd："
echo "  systemctl start face-attendance    - 启动服务"
echo "  systemctl stop face-attendance     - 停止服务"
echo ""
echo "服务地址："
echo "  网页: http://${BOARD_IP}:5000"
echo "  视频流: http://${BOARD_IP}:5000/video_feed"
echo ""