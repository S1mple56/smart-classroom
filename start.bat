@echo off
chcp 65001 >nul
echo ========================================
echo   校园智能监控系统
echo   集成: 人脸识别考勤 + YOLO 行为检测
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] 检查 Python 环境...
D:\anaconda\envs\py39\python -c "from ultralytics import YOLO; import cv2; import onnxruntime; print('依赖检查通过')" 2>nul
if errorlevel 1 (
    echo 错误: py39 环境缺少依赖，请先安装
    pause
    exit /b 1
)

set SILICONFLOW_API_KEY=sk-dmhuhgiqzcdngjyhmuajbrrrwzintniushyjzuduwaokvrqx

echo [2/2] 启动服务...
echo.
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.

D:\anaconda\envs\py39\python app.py
pause
