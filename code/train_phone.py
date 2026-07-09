#!/usr/bin/env python3
"""
重新训练 YOLO11 行为检测模型
- 使用 yolo11n.pt 从头训练（不使用之前损坏的模型）
- 输出到 runs/detect/train-final

使用方法（在 yolov8_env 环境下运行）：
  python train_phone.py
"""
import os, sys
from ultralytics import YOLO


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))

    # 数据集 yaml
    DATA_YAML = os.path.join(BASE, 'dataset', 'bvn.yaml')

    if not os.path.isfile(DATA_YAML):
        print(f'ERROR: 数据集配置文件不存在: {DATA_YAML}')
        sys.exit(1)

    # ===== 重要：使用全新的 yolo11n.pt，不从损坏的模型微调 =====
    model = YOLO('yolo11n.pt')

    results = model.train(
        data=DATA_YAML,
        epochs=150,              # 足够轮数确保分类头收敛
        imgsz=640,
        batch=16,                # 根据显存调整
        device='cuda:0',
        workers=4,
        project=os.path.join(BASE, 'runs', 'detect'),
        name='train-final',
        exist_ok=True,
        patience=30,             # 30 轮无提升早停
        lr0=0.001,
        lrf=0.0001,
        cos_lr=True,
        close_mosaic=10,
    )
