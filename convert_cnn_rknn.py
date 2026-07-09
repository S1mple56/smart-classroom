#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN 模型 RKNN 转换脚本
将 model/cnn_model.onnx 转换为 RK3588 NPU 可用的 .rknn 模型

使用方法：
    python convert_cnn_rknn.py

依赖：
    pip install rknn-toolkit2

注意：此脚本需在 PC（Windows/Linux x86）上运行，生成 .rknn 后传到 RK3588 板子使用。
"""

import os
import sys
import random

# ============ 配置 ============
ONNX_MODEL_PATH = 'model/cnn_model.onnx'
RKNN_MODEL_PATH = 'model/cnn_model.rknn'
QUANT_DATASET_DIR = 'model/quant_dataset'  # 量化校准图片目录
NUM_CALIB_IMAGES = 50  # 量化校准图片数量


def prepare_quant_dataset():
    """
    准备量化校准数据集。
    CNN 模型输入是 64x64，从训练数据集中随机采样若干张做校准。
    """
    import numpy as np
    from PIL import Image

    os.makedirs(QUANT_DATASET_DIR, exist_ok=True)

    # 查找训练数据集中的图片
    dataset_paths = [
        'datasets/bvn/images/train',
        'data/upload',
        'data/eval/students',
    ]
    all_images = []
    for dp in dataset_paths:
        if not os.path.isdir(dp):
            continue
        for root, _, files in os.walk(dp):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_images.append(os.path.join(root, f))

    if len(all_images) < NUM_CALIB_IMAGES:
        print(f"警告：找到 {len(all_images)} 张图片，不足 {NUM_CALIB_IMAGES} 张")
        selected = all_images
    else:
        selected = random.sample(all_images, NUM_CALIB_IMAGES)

    # 生成量化校准所需的 txt 列表文件
    txt_path = 'model/cnn_quant_dataset.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        for img_path in selected:
            # 保存为模型输入尺寸 (64x64) 的numpy数组，供rknn读取
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64), Image.BILINEAR)
            img_np = np.array(img).astype(np.float32) / 255.0
            # HWC -> CHW
            img_np = img_np.transpose((2, 0, 1))
            # 保存为 npy
            npy_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.npy'
            npy_path = os.path.join(QUANT_DATASET_DIR, npy_name)
            np.save(npy_path, img_np)
            f.write(npy_path + '\n')

    print(f"✓ 已生成 {len(selected)} 张量化校准图片")
    return txt_path


def convert():
    try:
        from rknn.api import RKNN
    except ImportError:
        print("错误：未安装 rknn-toolkit2")
        print("  pip install rknn-toolkit2")
        sys.exit(1)

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误：ONNX 模型不存在: {ONNX_MODEL_PATH}")
        print("  请先运行: python convert_to_onnx.py")
        sys.exit(1)

    print("=" * 60)
    print("CNN 模型 RKNN 转换")
    print("=" * 60)

    # 1. 准备量化校准数据
    print("\n[1/5] 准备量化校准数据集...")
    dataset_txt = prepare_quant_dataset()

    # 2. 创建 RKNN 对象
    print("\n[2/5] 初始化 RKNN...")
    rknn = RKNN(verbose=True)

    # 3. 配置 RKNN
    print("\n[3/5] 配置 RKNN (target: rk3588)...")
    rknn.config(
        target_platform='rk3588',
        # 量化配置：使用 int8 量化以最大化 NPU 性能
        quantization=True,
        # 优化级别
        optimization_level=3,
    )

    # 4. 加载 ONNX 模型
    print(f"\n[4/5] 加载 ONNX 模型: {ONNX_MODEL_PATH}")
    ret = rknn.load_onnx(
        model=ONNX_MODEL_PATH,
        input_size_list=[[1, 3, 64, 64]],  # 与模型输入一致
    )
    if ret != 0:
        print("✗ 加载 ONNX 失败")
        sys.exit(1)
    print("✓ 加载成功")

    # 5. 构建 RKNN 模型
    print("\n[5/5] 构建 RKNN 模型（量化中，可能需要几分钟）...")
    ret = rknn.build(
        do_quantization=True,
        dataset=dataset_txt,
    )
    if ret != 0:
        print("✗ 构建失败")
        sys.exit(1)
    print("✓ 构建成功")

    # 6. 导出 RKNN 模型
    print(f"\n导出 RKNN 模型: {RKNN_MODEL_PATH}")
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("✗ 导出失败")
        sys.exit(1)
    print("✓ 导出成功")

    # 7. 释放资源
    rknn.release()

    # 8. 验证文件
    if os.path.exists(RKNN_MODEL_PATH):
        size_mb = os.path.getsize(RKNN_MODEL_PATH) / 1024 / 1024
        print(f"\n{'=' * 60}")
        print(f"转换完成！")
        print(f"  RKNN 模型: {RKNN_MODEL_PATH}")
        print(f"  文件大小: {size_mb:.2f} MB")
        print(f"\n下一步：")
        print(f"  1. 将 {RKNN_MODEL_PATH} 复制到 RK3588 板子的 model/ 目录")
        print(f"  2. 在板子上安装: pip install rknnlite2")
        print(f"  3. 运行 app.py（已适配 RKNN 推理）")
        print(f"{'=' * 60}")
    else:
        print("✗ 文件未生成")


if __name__ == '__main__':
    convert()
