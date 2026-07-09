#!/usr/bin/env python3
"""修复 app.py 中的 dlib 导入问题"""
import os

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')

with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = 'import dlib_face_recognizer as face_rec'
new = """try:
    import dlib_face_recognizer as face_rec
    _face_rec_available = True
except ImportError:
    face_rec = None
    _face_rec_available = False
    print("提示: dlib 未安装，人脸识别功能不可用")"""

if old in content:
    content = content.replace(old, new, 1)
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('修改完成')
else:
    print('未找到 dlib 导入，可能已修改')
