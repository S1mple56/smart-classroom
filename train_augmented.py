#!/usr/bin/env python3
"""
数据增强 + YOLO 训练脚本
参考 Student-Attention-Analysis-YOLOv8 的增强策略
使用 albumentations 对少数类做增强，平衡数据集后训练

使用方法：
  pip install albumentations
  python train_augmented.py
"""
import os, sys, shutil, random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ============================================================================
# 配置
# ============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_SRC = os.path.join(BASE, 'datasets', 'bvn')           # 原始数据集
DATASET_BALANCED = os.path.join(BASE, 'datasets', 'bvn_balanced')  # 增强后
DATA_YAML = os.path.join(BASE, 'dataset', 'bvn_augmented.yaml')

# 类别（与现有模型一致）
CLASS_NAMES = ['hand-raising', 'reading', 'writing', 'using phone', 'bowing the head', 'leaning over the table']
NC = len(CLASS_NAMES)

# 训练参数
EPOCHS = 200
BATCH = 8
IMGSZ = 640
MODEL = 'yolo11s.pt'  # 比 nano 更大，精度更高
PATIENCE = 40

# ============================================================================
# 数据增强
# ============================================================================
def get_augmentation_pipeline():
    """构建增强管道（参考 notebook 的策略）"""
    try:
        import albumentations as A
    except ImportError:
        print("请先安装 albumentations: pip install albumentations")
        sys.exit(1)

    return A.Compose([
        # 光照变化
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=1.0),
        ], p=0.8),

        # 噪声和模糊（模拟摄像头）
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
        ], p=0.3),

        # 轻微变换（模拟不同角度）
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=3,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),

        # 压缩模拟
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.4,
        min_area=0.01
    ))


def augment_image(image_path, label_path, output_img, output_lbl, pipeline):
    """对单张图片做增强"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    coords = list(map(float, parts[1:5]))
                    bboxes.append(coords)
                    class_labels.append(cid)

        if not bboxes:
            return False

        transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['class_labels']

        if not aug_bboxes:
            return False

        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_img), aug_img_bgr)

        with open(output_lbl, 'w') as f:
            for bbox, cid in zip(aug_bboxes, aug_labels):
                f.write(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        return True
    except Exception as e:
        print(f"增强失败 {image_path.name}: {e}")
        return False


def balance_dataset(src, dst, target_ratio=0.6):
    """平衡数据集：对少数类做增强"""
    import albumentations as A

    src_images = Path(src) / 'images' / 'train'
    src_labels = Path(src) / 'labels' / 'train'

    # 复制整个数据集到目标
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    dst_images = Path(dst) / 'images' / 'train'
    dst_labels = Path(dst) / 'labels' / 'train'

    # 统计每类的图片数
    class_files = defaultdict(list)
    for lbl_file in src_labels.glob('*.txt'):
        with open(lbl_file, 'r') as f:
            classes = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        if 0 <= cid < NC:
                            classes.add(cid)
                    except:
                        pass
            for cid in classes:
                class_files[cid].append(lbl_file.stem)

    print("\n原始分布:")
    for cid in range(NC):
        print(f"  {CLASS_NAMES[cid]:25s}: {len(class_files[cid]):5d} 张")

    max_count = max(len(v) for v in class_files.values())
    target = int(max_count * target_ratio)
    print(f"\n目标: 每类至少 {target} 张")

    pipeline = get_augmentation_pipeline()
    total_aug = 0

    for cid in range(NC):
        files = class_files[cid]
        shortage = target - len(files)
        if shortage <= 0:
            print(f"  {CLASS_NAMES[cid]:25s}: 已足够")
            continue

        print(f"  {CLASS_NAMES[cid]:25s}: 需增强 {shortage} 张...")
        generated = 0
        attempts = 0

        while generated < shortage and attempts < shortage * 3:
            src_file = random.choice(files)
            src_img = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = src_images / f'{src_file}{ext}'
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                attempts += 1
                continue

            src_lbl = src_labels / f'{src_file}.txt'
            if not src_lbl.exists():
                attempts += 1
                continue

            aug_name = f"{src_file}_aug_{generated}"
            out_img = dst_images / f"{aug_name}{src_img.suffix}"
            out_lbl = dst_labels / f"{aug_name}.txt"

            if augment_image(src_img, src_lbl, out_img, out_lbl, pipeline):
                generated += 1
                total_aug += 1
            attempts += 1

        print(f"    生成 {generated} 张")

    print(f"\n共增强 {total_aug} 张图片")

    # 显示增强后的分布
    print("\n增强后分布:")
    class_files_new = defaultdict(list)
    for lbl_file in dst_labels.glob('*.txt'):
        with open(lbl_file, 'r') as f:
            classes = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        if 0 <= cid < NC:
                            classes.add(cid)
                    except:
                        pass
            for cid in classes:
                class_files_new[cid].append(lbl_file.stem)

    for cid in range(NC):
        print(f"  {CLASS_NAMES[cid]:25s}: {len(class_files_new[cid]):5d} 张")

    return total_aug


def create_data_yaml():
    """创建 data.yaml"""
    yaml_content = f"""path: {DATASET_BALANCED}
train: images/train
val: images/val
nc: {NC}
names: {CLASS_NAMES}
"""
    os.makedirs(os.path.dirname(DATA_YAML), exist_ok=True)
    with open(DATA_YAML, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"data.yaml 已创建: {DATA_YAML}")


def train():
    """训练模型"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"  模型: {MODEL}")
    print(f"  轮数: {EPOCHS}")
    print(f"  图片尺寸: {IMGSZ}")
    print(f"  Batch: {BATCH}")
    print(f"{'='*60}\n")

    model = YOLO(MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device='cuda:0',
        workers=4,
        project=os.path.join(BASE, 'runs', 'detect'),
        name='train-augmented',
        exist_ok=True,
        patience=PATIENCE,
        lr0=0.001,
        lrf=0.0001,
        cos_lr=True,
        close_mosaic=15,
        optimizer='AdamW',
        seed=42,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # 复制最优模型
    src = os.path.join(BASE, 'runs', 'detect', 'train-augmented', 'weights', 'best.pt')
    dst = os.path.join(BASE, 'models', 'class_behavior_best.pt')
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"\n最优模型已复制到: {dst}")

    return results


if __name__ == '__main__':
    # 1. 平衡数据集
    print("="*60)
    print("步骤 1: 数据增强 + 平衡")
    print("="*60)
    total_aug = balance_dataset(DATASET_SRC, DATASET_BALANCED, target_ratio=0.6)

    # 2. 创建 data.yaml
    print("\n" + "="*60)
    print("步骤 2: 创建 data.yaml")
    print("="*60)
    create_data_yaml()

    # 3. 训练
    print("\n" + "="*60)
    print("步骤 3: 训练模型")
    print("="*60)
    train()
