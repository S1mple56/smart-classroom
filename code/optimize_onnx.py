#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型优化脚本 - 针对 RK3588 NPU 优化
使用方法: python optimize_onnx.py

注意: 需要在 PC 上运行，生成优化后的模型再传到板子上
"""

import os
import sys

def optimize_onnx_model():
    """优化 ONNX 模型以获得更好的推理性能"""
    import onnx
    from onnx import optimizer, shape_inference

    model_path = 'model/cnn_model.onnx'
    optimized_path = 'model/cnn_model_rk3588.onnx'

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return False

    print("=" * 50)
    print("ONNX 模型优化工具 - RK3588 NPU 优化")
    print("=" * 50)
    print()

    # 加载模型
    print(f"加载模型: {model_path}")
    model = onnx.load(model_path)

    # 检查模型
    print("模型信息:")
    print(f"  IR 版本: {model.ir_version}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Opset 版本: {model.opset_import[0].version}")
    print()

    # 应用优化 passes
    print("应用优化 passes...")

    # 常用优化
    optimization_passes = [
        'eliminate_identity',
        'eliminate_nop_pad',
        'eliminate_unused_initializer',
        'fuse_add_bias_into_conv',
        'fuse_bn_into_conv',
        'fuse_consecutive_squeezes',
        'fuse_dequantize_linear',
        'fuse_matmul_add_bias',
        'fuse_pad_into_conv',
        'fuse_quantize_linear',
        'fuse_resize_pad',
        'eliminate_deadend',
    ]

    try:
        model = optimizer.optimize(model, optimization_passes)
        print("  ✓ 基础优化完成")
    except Exception as e:
        print(f"  ⚠ 部分优化失败: {e}")

    # 应用形状推理
    try:
        model = shape_inference.infer_shapes(model)
        print("  ✓ 形状推理完成")
    except Exception as e:
        print(f"  ⚠ 形状推理失败: {e}")

    # 保存优化后的模型
    print()
    print(f"保存优化后的模型: {optimized_path}")
    onnx.save(model, optimized_path)

    # 验证模型
    print()
    print("验证模型...")
    try:
        onnx.checker.check_model(model)
        print("  ✓ 模型验证通过")
    except Exception as e:
        print(f"  ⚠ 模型验证警告: {e}")

    # 获取模型大小
    original_size = os.path.getsize(model_path) / 1024 / 1024
    optimized_size = os.path.getsize(optimized_path) / 1024 / 1024

    print()
    print("=" * 50)
    print("优化完成!")
    print(f"  原始大小: {original_size:.2f} MB")
    print(f"  优化后大小: {optimized_size:.2f} MB")
    print(f"  压缩比: {original_size/optimized_size:.2f}x")
    print("=" * 50)
    print()
    print("下一步:")
    print(f"  1. 将 {optimized_path} 复制到 RK3588 板子")
    print("  2. 在板子上将模型重命名为 cnn_model.onnx")
    print("  3. 或修改 app.py 中的模型路径")

    return True

def create_rknn_model():
    """创建 RKNN 模型（需要 RKNN toolkit）"""
    print()
    print("=" * 50)
    print("RKNN 模型转换")
    print("=" * 50)
    print()
    print("如需使用 RK3588 NPU 加速，请使用 RKNN Toolkit:")
    print()
    print("  pip install rknn-toolkit")
    print()
    print("转换示例:")
    print("""
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='model/cnn_model.onnx', input_size_list=[[3, 64, 64]])
rknn.build(do_quantization=True)
rknn.export_rknn(model='model/cnn_model_rk3588.rknn')
    """)
    print()
    print("注意: RKNN 转换需要在 PC 上完成，生成 .rknn 文件后再传到板子")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--rknn':
        create_rknn_model()
    else:
        optimize_onnx_model()
