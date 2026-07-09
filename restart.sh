#!/bin/bash
# 重启学生行为识别系统服务

echo "正在重启服务..."
./stop.sh
sleep 2
./start.sh
