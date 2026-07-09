#!/usr/bin/env python3
"""
优化版 YOLO 训练脚本 — 行为检测模型
改进点：
  1. 模型：yolo11n → yolo11s（small，精度更高，速度仍然快）
  2. 图片尺寸：640 → 800（更清晰的特征）
  3. 轮数：150 → 300，patience 30 → 50（给更多收敛时间）
  4. 增强数据增强：加入 mixup、copy_paste
  5. 使用 cos_lr 学习率调度

使用方法：
  python train_optimized.py
"""
import os, sys
from ultralytics import YOLO

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(BASE, 'dataset', 'bvn.yaml')

    if not os.path.isfile(DATA_YAML):
        print(f'ERROR: 数据集配置文件不存在: {DATA_YAML}')
        sys.exit(1)

    # 使用 yolo11s（small）— 比 nano 精度高 3-5%，速度仍然快
    model = YOLO('yolo11s.pt')

    results = model.train(
        data=DATA_YAML,
        epochs=300,              # 更多轮数
        imgsz=800,               # 更大图片尺寸
        batch=8,                 # 图片大了，batch 减小（显存不够可改回 16 + imgsz=640）
        device='cuda:0',
        workers=4,
        project=os.path.join(BASE, 'runs', 'detect'),
        name='train-optimized',
        exist_ok=True,
        patience=50,             # 更宽容的早停
        lr0=0.001,
        lrf=0.0001,
        cos_lr=True,
        close_mosaic=15,         # 最后15轮关闭mosaic，让模型适应真实分布
        # 增强
        mixup=0.1,               # 混合增强
        copy_paste=0.1,          # 复制粘贴增强（小目标）
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        erasing=0.4,
    )

    # 训练完成后自动复制最优模型到 models/
    import shutil
    src = os.path.join(BASE, 'runs', 'detect', 'train-optimized', 'weights', 'best.pt')
    dst = os.path.join(BASE, 'models', 'class_behavior_best.pt')
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f'\n✅ 最优模型已复制到: {dst}')
    else:
        print(f'\n⚠️ 未找到训练结果: {src}')
