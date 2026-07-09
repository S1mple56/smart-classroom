#!/usr/bin/env python3
"""
修复 YOLO 训练数据集：
1. 从验证集中移除 class 4 和 5 的标注
2. 将 class 3 (phone) 的 60% 数据从 val 移到 train
"""
import os, sys, shutil, random

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'datasets', 'bvn')
TRAIN_LABELS = os.path.join(DATASET, 'labels', 'train')
TRAIN_IMAGES = os.path.join(DATASET, 'images', 'train')
VAL_LABELS = os.path.join(DATASET, 'labels', 'val')
VAL_IMAGES = os.path.join(DATASET, 'images', 'val')

random.seed(42)
corrupted_files = []


def safe_read_lines(fpath):
    """安全读取文件，损坏则跳过"""
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f.readlines() if l.strip()]
    except OSError:
        corrupted_files.append(fpath)
        return []


def safe_write_lines(fpath, lines):
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except OSError:
        corrupted_files.append(fpath)


# ========== 步骤 1: 清理无效标注 (class 4/5) ==========
def clean_invalid_classes(label_dir, dir_name):
    """从标注目录中移除 class 4 和 5"""
    r4, r5 = 0, 0
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(label_dir, fname)
        lines = safe_read_lines(fpath)
        if not lines:
            continue
        new_lines = []
        for line in lines:
            try:
                cls_id = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if cls_id >= 4:  # class 4 或更高
                if cls_id == 4:
                    r4 += 1
                elif cls_id == 5:
                    r5 += 1
                continue
            new_lines.append(line + '\n')
        safe_write_lines(fpath, new_lines)
    print(f"  {dir_name}: 移除 class 4={r4}, class 5={r5}")
    return r4, r5

print("=" * 50)
print("步骤 1: 清理 class 4/5 无效标注")
t4, t5 = clean_invalid_classes(TRAIN_LABELS, "train")
v4, v5 = clean_invalid_classes(VAL_LABELS, "val")
print(f"  合计移除: class 4={t4+v4}, class 5={t5+v5}")

# ========== 步骤 2: 把 phone 数据从 val 移到 train ==========
print("\n步骤 2: 将 class 3 (phone) 的 60% 从 val 移到 train")

phone_files = []
for fname in os.listdir(VAL_LABELS):
    if not fname.endswith('.txt'):
        continue
    fpath = os.path.join(VAL_LABELS, fname)
    lines = safe_read_lines(fpath)
    has_phone = any(int(l.split()[0]) == 3 for l in lines)
    if has_phone:
        phone_files.append(fname)

print(f"  验证集中含 phone 的文件: {len(phone_files)} 个")

random.shuffle(phone_files)
split_idx = int(len(phone_files) * 0.6)
move_files = phone_files[:split_idx]
keep_files = phone_files[split_idx:]
print(f"  移到 train: {len(move_files)} 个, 留在 val: {len(keep_files)} 个")

# 移动 label 和 image（跳过损坏文件）
moved_count = 0
for fname in move_files:
    label_src = os.path.join(VAL_LABELS, fname)
    label_dst = os.path.join(TRAIN_LABELS, fname)
    try:
        shutil.move(label_src, label_dst)
    except (OSError, shutil.Error):
        corrupted_files.append(label_src)
        continue

    stem = os.path.splitext(fname)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        img_src = os.path.join(VAL_IMAGES, stem + ext)
        img_dst = os.path.join(TRAIN_IMAGES, stem + ext)
        if os.path.isfile(img_src):
            try:
                shutil.move(img_src, img_dst)
            except (OSError, shutil.Error):
                corrupted_files.append(img_src)
            break
    moved_count += 1

print(f"  实际移动成功: {moved_count} 个")

# ========== 步骤 3: 统计最终分布 ==========
print("\n" + "=" * 50)
print("步骤 3: 最终数据分布")

def count_classes(label_dir):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    files = 0
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(label_dir, fname)
        lines = safe_read_lines(fpath)
        if not lines:
            continue
        files += 1
        for line in lines:
            try:
                cls_id = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if cls_id in counts:
                counts[cls_id] += 1
    return files, counts

train_files, train_counts = count_classes(TRAIN_LABELS)
val_files, val_counts = count_classes(VAL_LABELS)

CLASS_NAMES = {0: 'hand_up', 1: 'abnormal', 2: 'sleep', 3: 'phone'}

print(f"\n训练集 ({train_files} 文件):")
total_train = 0
for cls_id, name in CLASS_NAMES.items():
    c = train_counts[cls_id]
    total_train += c
    bar = '#' * (c // 20) if c >= 20 else ''
    print(f"  {name}: {c:>5d} {bar}")
print(f"  总计: {total_train}")

print(f"\n验证集 ({val_files} 文件):")
total_val = 0
for cls_id, name in CLASS_NAMES.items():
    c = val_counts[cls_id]
    total_val += c
    bar = '#' * (c // 20) if c >= 20 else ''
    print(f"  {name}: {c:>5d} {bar}")
print(f"  总计: {total_val}")

# ========== 结果判断 ==========
print("\n" + "=" * 50)
all_ok = True
for cls_id, name in CLASS_NAMES.items():
    if train_counts[cls_id] == 0:
        print(f"!! 警告: {name} 训练集样本为 0，模型无法学习！")
        all_ok = False

if all_ok:
    print("数据检查通过，所有类别都有训练样本。")
else:
    print("请补充缺失类别的训练数据后再训练。")

if corrupted_files:
    print(f"\n!! {len(corrupted_files)} 个文件损坏已跳过:")
    for cf in corrupted_files[:10]:
        print(f"   {cf}")
    if len(corrupted_files) > 10:
        print(f"   ... 还有 {len(corrupted_files) - 10} 个")

print("\n可以运行训练了：")
print("  conda activate yolov8_env")
print("  python train_phone.py")
