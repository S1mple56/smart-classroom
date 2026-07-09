#!/usr/bin/env python3
"""打包 bvn 数据集为 zip，用于上传到 Google Colab"""
import os, zipfile
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, 'datasets', 'bvn')
OUT = os.path.join(BASE, 'bvn.zip')

print(f"打包: {SRC}")
print(f"输出: {OUT}")

count = 0
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SRC):
        for f in files:
            filepath = os.path.join(root, f)
            arcname = os.path.relpath(filepath, SRC)
            zf.write(filepath, arcname)
            count += 1
            if count % 100 == 0:
                print(f"  已打包 {count} 个文件...")

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"\n完成! 共 {count} 个文件, {size_mb:.1f} MB")
print(f"上传到 Colab: {OUT}")
