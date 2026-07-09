"""测试模型对各类别的检测置信度"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO

model_path = os.path.join(BASE, 'runs', 'detect', 'train-3', 'weights', 'best.pt')
model = YOLO(model_path)

# 测试几个验证集图片（有不同类别的）
val_images = os.path.join(BASE, 'datasets', 'bvn', 'images', 'val')
val_labels = os.path.join(BASE, 'datasets', 'bvn', 'labels', 'val')

test_files = []
for fn in os.listdir(val_labels):
    if fn.endswith('.txt'):
        path = os.path.join(val_labels, fn)
        with open(path) as f:
            classes = set()
            for line in f:
                cid = line.strip().split()[0]
                classes.add(int(cid))
        test_files.append((fn.replace('.txt', '.jpg'), classes))

# 每类取最多3个样本测试
names = {0:'hand_up', 1:'abnormal', 2:'sleep', 3:'phone'}
from collections import defaultdict
samples = defaultdict(list)
for img_fn, cls_set in test_files:
    for c in cls_set:
        if c > 0 and len(samples[c]) < 3:  # 只取非hand_up的类
            samples[c].append(img_fn)
            break

print("=" * 60)
print("模型对少数类检测测试 (conf阈值=0)")
print("=" * 60)

for cid, img_list in sorted(samples.items()):
    print(f"\n--- 类别 {cid} ({names[cid]}) ---")
    for img_fn in img_list:
        img_path = os.path.join(val_images, img_fn)
        if not os.path.isfile(img_path):
            print(f"  {img_fn}: 图片不存在，跳过")
            continue
        results = model.predict(img_path, imgsz=640, conf=0, verbose=False, device='cpu')
        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                print(f"  {img_fn}: 无检测结果")
                continue
            confs = boxes.conf.tolist()
            clsids = [int(x) for x in boxes.cls.tolist()]
            for cls_id, conf in zip(clsids, confs):
                name = names.get(cls_id, str(cls_id))
                marker = " ***" if cls_id == cid else ""
                print(f"  {img_fn}: {name} conf={conf:.3f}{marker}")
