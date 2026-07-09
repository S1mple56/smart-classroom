#!/usr/bin/env python3
"""
板端 NPU 行为检测诊断脚本
用法: python3 diagnose_npu.py [图片路径]
如果没有传图片路径，会尝试打开摄像头拍一张
"""
import os, sys, time
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'rknn_convert'))

# 1. 加载 NPU YOLO 模型
from yolo_rknn import init_model, detect, CLASS_NAMES, CONF_THRESHOLD
rknn_path = os.path.join(BASE_DIR, 'rknn_convert', 'yolo_best.rknn')
print(f"[1] 加载模型: {rknn_path}")
print(f"    文件大小: {os.path.getsize(rknn_path)/1024:.0f} KB")
if not init_model(rknn_path):
    print("    加载失败!")
    sys.exit(1)
print(f"    加载成功, CONF_THRESHOLD={CONF_THRESHOLD}")

# 2. 获取图片
img_path = sys.argv[1] if len(sys.argv) > 1 else None
if img_path and os.path.isfile(img_path):
    img = Image.open(img_path).convert('RGB')
    print(f"\n[2] 测试图片: {img_path} ({img.size[0]}x{img.size[1]})")
else:
    print("\n[2] 尝试打开摄像头拍摄...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("    摄像头不可用")
            sys.exit(1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"    拍摄成功: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        print(f"    摄像头失败: {e}")
        print("    使用纯灰色测试图")
        img = Image.new('RGB', (640, 480), (128, 128, 128))

# 3. 推理
print(f"\n[3] 开始 NPU 推理...")
t0 = time.time()
detections = detect(img)
elapsed = (time.time() - t0) * 1000
print(f"    耗时: {elapsed:.1f} ms")
print(f"    检测数: {len(detections)}")

# 4. 按类别分组显示
if detections:
    from collections import Counter
    class_counts = Counter(d['class_name'] for d in detections)
    print(f"\n[4] 类别统计:")
    for cls_name in CLASS_NAMES:
        cnt = class_counts.get(cls_name, 0)
        bar = '#' * min(cnt, 50)
        print(f"    {cls_name:>10}: {cnt:>3} {bar}")
    
    print(f"\n[5] 检测详情 (conf>={CONF_THRESHOLD}):")
    for i, d in enumerate(sorted(detections, key=lambda x: x['conf'], reverse=True)):
        box = d['xyxy']
        print(f"    #{i+1}: {d['class_name']:>10}  conf={d['conf']:.3f}  "
              f"box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")
else:
    print(f"\n[4] 未检测到任何目标 (conf={CONF_THRESHOLD})")
    print(f"    当前画面中没有举手/睡觉/玩手机/异常行为")

print(f"\n{'='*50}")
print(f"诊断完成。如果 hand_up > 0 但画面中没人举手，")
print(f"说明模型本身有问题，需重新训练。")
