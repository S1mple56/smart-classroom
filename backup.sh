#!/bin/bash
#==============================================================================
# 项目备份脚本 - 排除训练数据和缓存
# 使用方法: ./backup.sh [目标路径]
#==============================================================================

set -e

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="face-attendance-backup-$(date +%Y%m%d-%H%M%S)"

# 备份目标路径
if [ -z "$1" ]; then
    BACKUP_DIR="/home/root/backups"
else
    BACKUP_DIR="$1"
fi

# 创建备份目录
mkdir -p "$BACKUP_DIR"

echo "========================================"
echo "  项目备份脚本"
echo "========================================"
echo ""
echo "项目: $PROJECT_NAME"
echo "目标: $BACKUP_DIR"
echo ""

# 创建临时目录
TEMP_DIR="/tmp/$PROJECT_NAME"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# 复制文件（排除大文件和训练数据）
echo "正在复制文件..."

# 需要保留的目录和文件
rsync -av \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='venv' \
    --exclude='.git' \
    --exclude='data/upload' \
    --exclude='data/eval/students' \
    --exclude='data/temp_recognize' \
    --exclude='data/zipped-eval' \
    --exclude='data/zipped-train' \
    --exclude='log' \
    --exclude='*.log' \
    --exclude='__pycache__' \
    "$PROJECT_DIR/" "$TEMP_DIR/"

# 打包
echo "正在打包..."
cd /tmp
tar -czf "$BACKUP_DIR/$PROJECT_NAME.tar.gz" "$PROJECT_NAME"

# 清理临时目录
rm -rf "$TEMP_DIR"

# 计算大小
SIZE=$(du -h "$BACKUP_DIR/$PROJECT_NAME.tar.gz" | cut -f1)

echo ""
echo "✓ 备份完成!"
echo ""
echo "备份文件: $BACKUP_DIR/$PROJECT_NAME.tar.gz"
echo "文件大小: $SIZE"
echo ""

# 列出备份目录中的所有备份
echo "现有备份:"
ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "无其他备份"
