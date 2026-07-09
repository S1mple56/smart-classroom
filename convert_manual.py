#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动方式转换 PaddlePaddle 模型为 ONNX 格式
"""

import paddle
import numpy as np
import os
import subprocess
from cnn_model import CNN

def convert_model():
    # 1. 创建模型实例
    model = CNN()
    
    # 2. 加载预训练权重
    try:
        layer_state_dict = paddle.load('model/cnn_layer.pdparams')
        model.set_state_dict(layer_state_dict)
        print("✓ 成功加载预训练权重")
    except Exception as e:
        print(f"✗ 加载权重失败: {e}")
        return
    
    # 3. 设置模型为评估模式
    model.eval()
    
    # 4. 先保存为 paddle 完整模型格式
    paddle.save(model.state_dict(), 'model/cnn_full.pdparams')
    
    # 5. 使用 paddle2onnx 命令行工具转换
    cmd = [
        'paddle2onnx',
        '--model_dir', 'model/',
        '--model_filename', 'cnn_full.pdparams',
        '--save_file', 'model/cnn_model.onnx',
        '--input_shape', '{"x":[1,3,64,64]}',
        '--opset_version', '11'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 成功导出 ONNX 模型到 model/cnn_model.onnx")
        else:
            print(f"✗ 导出失败: {result.stderr}")
            return
    except Exception as e:
        print(f"✗ 执行命令失败: {e}")
        return
    
    # 6. 验证导出的模型
    try:
        import onnx
        onnx_model = onnx.load('model/cnn_model.onnx')
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 模型验证通过")
        
        print("\n模型信息:")
        print(f"  输入节点: {[input.name for input in onnx_model.graph.input]}")
        print(f"  输出节点: {[output.name for output in onnx_model.graph.output]}")
        print(f"  算子数量: {len(onnx_model.graph.node)}")
    except ImportError:
        print("提示: 未安装 onnx 库，跳过模型验证")
    except Exception as e:
        print(f"✗ ONNX 模型验证失败: {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("PaddlePaddle 模型转 ONNX 工具")
    print("=" * 50)
    convert_model()
    print("=" * 50)