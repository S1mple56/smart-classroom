#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型推理工具
用于在 RK3588 开发板上运行行为识别模型
"""

import onnxruntime as ort
import numpy as np
from PIL import Image

class ONNXModel:
    def __init__(self, model_path='model/cnn_model.onnx'):
        """
        初始化 ONNX 模型
        
        Args:
            model_path: ONNX 模型文件路径
        """
        # 加载 ONNX 模型
        self.session = ort.InferenceSession(model_path)
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 行为标签
        self.action_labels = ['听讲', '举手', '低头', '站立', '走动']
    
    def preprocess(self, image_path):
        """
        预处理图片：resize -> normalize -> transpose
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            预处理后的 numpy 数组，形状为 [1, 3, 64, 64]
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize((64, 64), Image.BILINEAR)
        img_np = np.array(img).astype('float32') / 255.0
        img_np = img_np.transpose((2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)
        return img_np
    
    def predict(self, image_path):
        """
        预测图片中的行为
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            dict: 包含 action（行为）和 confidence（置信度）
        """
        try:
            # 预处理图片
            input_data = self.preprocess(image_path)
            
            # 执行推理
            result = self.session.run([self.output_name], {self.input_name: input_data})
            
            # 解析结果
            output = result[0][0]
            pred_idx = np.argmax(output)
            
            # 计算置信度（softmax）
            from scipy.special import softmax
            prob = softmax(output)
            confidence = prob[pred_idx] * 100
            
            return {
                'action': self.action_labels[pred_idx],
                'confidence': f'{confidence:.1f}%',
                'success': True
            }
        except Exception as e:
            return {
                'action': '识别失败',
                'confidence': '0%',
                'success': False,
                'error': str(e)
            }

# 测试函数
if __name__ == '__main__':
    model = ONNXModel('model/cnn_model.onnx')
    print("✓ ONNX 模型加载成功")
    
    # 测试推理
    result = model.predict('test.jpg') if __name__ == '__main__' else None
    if result and result['success']:
        print(f"预测结果: {result['action']} (置信度: {result['confidence']})")
    else:
        print("测试完成（无测试图片）")