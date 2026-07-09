#!/bin/bash
# 查看服务状态

echo "========================================"
echo "  学生行为识别系统 - 服务状态"
echo "========================================"
echo ""

# 检查进程
if pgrep -f "python.*app.py" > /dev/null 2>&1; then
    echo "✓ Python 服务运行中"
    echo ""
    echo "进程信息:"
    ps aux | grep "python.*app.py" | grep -v grep
else
    echo "✗ Python 服务未运行"
fi

echo ""
echo "端口监听:"
netstat -tlnp 2>/dev/null | grep ":5000" || ss -tlnp 2>/dev/null | grep ":5000" || echo "端口 5000 未监听"

echo ""
echo "最近日志 (最后 10 行):"
if [ -f "log/stdout.log" ]; then
    tail -n 10 log/stdout.log 2>/dev/null || echo "无日志记录"
else
    echo "无日志文件"
fi
