#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载示例学生行为图片数据
"""

import os
import urllib.request
import zipfile

def download_sample_data():
    """下载示例数据"""
    url = "https://github.com/harryhan618/Human-Action-Recognition/raw/master/dataset.zip"
    save_path = "data/sample_dataset.zip"
    extract_path = "data/sample_data"
    
    # 创建目录
    os.makedirs("data", exist_ok=True)
    
    print("正在下载示例数据集...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print("下载完成！")
        
        # 解压
        print("正在解压...")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("解压完成！")
        
        # 查看数据集结构
        print("\n数据集结构：")
        for root, dirs, files in os.walk(extract_path):
            level = root.replace(extract_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... 还有 {len(files) - 3} 个文件")
                
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载数据集：")
        print("URL: https://github.com/harryhan618/Human-Action-Recognition/raw/master/dataset.zip")

if __name__ == "__main__":
    download_sample_data()