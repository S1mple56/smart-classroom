#!/bin/bash
# ============================================================
# 课堂行为检测系统 — 开发板环境一键安装
# 在 Rockchip 开发板 (Ubuntu/Debian) 上运行
# ============================================================
set -e

echo "========================================"
echo "  课堂行为检测 — 板子环境安装"
echo "========================================"

# 1. 基础包
echo "[1/5] 安装系统依赖..."
sudo apt update
sudo apt install -y python3 python3-pip python3-opencv \
    fonts-wqy-zenhei libgl1-mesa-glx libglib2.0-0

# 2. Python 包
echo "[2/5] 安装 Python 依赖..."
pip3 install flask numpy pillow --quiet

# 3. YOLO (ultralytics + PyTorch)
echo "[3/5] 安装 YOLO/PyTorch (可能需要几分钟)..."
pip3 install torch torchvision --quiet --index-url https://download.pytorch.org/whl/cpu
pip3 install ultralytics --quiet

# 4. RKNN Lite (NPU 推理)
echo "[4/5] 安装 RKNN Lite2..."
if pip3 list 2>/dev/null | grep -q rknn-toolkit-lite2; then
    echo "  RKNN Lite2 已安装，跳过"
else
    # 从 GitHub 下载 aarch64 版本
    ARCH=$(uname -m)
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
    WHL="rknn_toolkit_lite2-2.2.0-cp${PY_VER}-cp${PY_VER}-linux_aarch64.whl"
    URL="https://github.com/airockchip/rknn-toolkit2/releases/download/v2.2.0/${WHL}"
    wget -q "$URL" -O /tmp/${WHL} || {
        echo "  下载失败，请手动安装: pip3 install rknn-toolkit-lite2"
    }
    pip3 install /tmp/${WHL} --quiet 2>/dev/null || true
    rm -f /tmp/${WHL}
fi

# 5. 验证
echo "[5/5] 验证安装..."
echo ""
python3 -c "
import cv2;      print(f'  OpenCV      {cv2.__version__}')
import numpy;    print(f'  NumPy       {numpy.__version__}')
import flask;    print(f'  Flask       {flask.__version__}')
import PIL;      print(f'  Pillow      {PIL.__version__}')
try:
    from rknnlite.api import RKNNLite; print(f'  RKNN Lite   已安装')
except:          print(f'  RKNN Lite   未安装 (不影响主体功能)')
try:
    import torch; print(f'  PyTorch     {torch.__version__}')
except:          print(f'  PyTorch     未安装')
try:
    import ultralytics; print(f'  Ultralytics 已安装')
except:          print(f'  Ultralytics 未安装')
"
echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
