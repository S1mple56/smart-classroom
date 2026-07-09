#!/usr/bin/env python3
"""修复 app.py 中的中文标签乱码"""
import os

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')

with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 CHINESE_LABELS 行
old_line = 'CHINESE_LABELS = {"hand-raising": "举手", "reading": "阅读", "writing": "写作", "using phone": "使用手机", "bowing the head": "低头", "leaning over the table": "靠在桌子上"}'

lines = content.split('\n')
fixed = False
for i, line in enumerate(lines):
    if 'CHINESE_LABELS' in line and 'hand-raising' in line:
        lines[i] = old_line
        fixed = True
        print(f'Fixed line {i}')
        break

if fixed:
    content = '\n'.join(lines)
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Done')
else:
    print('CHINESE_LABELS not found')
