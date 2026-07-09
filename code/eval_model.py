#!/usr/bin/env python3
"""
评估 YOLO 行为检测模型的 mAP 指标
使用方法：
  python eval_model.py                          # 评估当前使用的模型
  python eval_model.py runs/detect/train-optimized/weights/best.pt  # 指定模型
"""
import os, sys
from ultralytics import YOLO

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(BASE, 'dataset', 'bvn.yaml')

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(BASE, 'models', 'class_behavior_best.pt')

    if not os.path.isfile(model_path):
        print(f'ERROR: 模型不存在: {model_path}')
        sys.exit(1)

    print(f'评估模型: {model_path}')
    print(f'数据集:   {DATA_YAML}\n')

    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, device='cuda:0')

    print(f'\n===== 评估结果 =====')
    print(f'mAP50    : {metrics.box.map50:.4f}')
    print(f'mAP50-95 : {metrics.box.map:.4f}')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall   : {metrics.box.mr:.4f}')

    # 每类指标
    names = metrics.names
    for i, (p, r, ap50, ap) in enumerate(
        zip(metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap)
    ):
        print(f'  [{i}] {names[i]:12s}  P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}')
