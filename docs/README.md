# 智瞳课堂 - 智能课堂行为分析系统

> 基于 RK3588 开发板的智能课堂行为分析系统，集成人脸识别考勤、行为检测、AI 助手等功能。

## 项目简介

智瞳课堂是一套面向教育场景的智能分析系统，通过 YOLO 行为检测和人脸识别技术，实现课堂考勤自动化和学生行为分析。

## 主要功能

### 1. 人脸识别考勤
- 拍照识别学生人脸
- 自动记录考勤信息
- 支持手动输入学号考勤
- 考勤数据统计与导出

### 2. 行为检测分析
- YOLO 实时行为检测
- 支持 6 种课堂行为识别：
  - 举手 (hand-raising)
  - 阅读 (reading)
  - 写字 (writing)
  - 使用手机 (using phone)
  - 低头 (bowing the head)
  - 趴桌子 (leaning over the table)

### 3. AI 智能助手
- 接入 SiliconFlow API
- 支持文字对话
- 智能问答与学习辅导

### 4. 实时监控
- 摄像头实时画面显示
- 检测结果实时标注
- 网页端远程查看

## 技术架构

### 硬件平台
- **主控板**: ELF2 RK3588 开发板
- **摄像头**: USB 摄像头 (设备号 21)
- **触摸屏**: P08-RT2660TP-MAIN-V1.0

### 软件架构
```
┌─────────────────────────────────────────────────────────┐
│                    用户界面层                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Qt 触摸屏   │  │  Web 网页    │  │  移动端      │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
├─────────┴────────────────┴────────────────┴─────────────┤
│                    API 接口层                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Flask RESTful API                   │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    业务逻辑层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ 人脸识别  │  │ 行为检测  │  │ AI 助手  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
├─────────────────────────────────────────────────────────┤
│                    数据存储层                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │           JSON 文件 + 本地存储                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
smart-classroom/
├── code/                    # 核心代码
│   ├── app.py              # Flask 后端主程序
│   ├── qt_touch_app.py     # Qt 触摸屏界面
│   ├── templates/          # 网页模板
│   │   ├── index.html      # 主页面
│   │   └── login.html      # 登录页面
│   ├── static/             # 静态资源
│   └── web/                # Web 相关文件
│
├── docs/                    # 项目文档
│   ├── README.md           # 项目说明（本文件）
│   ├── 智瞳课堂_项目报告.docx  # 项目报告
│   └── DEPLOY_RK3588.md    # 部署文档
│
├── scripts/                 # 脚本文件
│   ├── start_qt.sh         # Qt 应用启动脚本
│   ├── deploy.sh           # 部署脚本
│   └── start_board.sh      # 板端启动脚本
│
├── config/                  # 配置文件
│   ├── requirements.txt    # Python 依赖
│   ├── zhitong-classroom.service  # 系统服务配置
│   └── face-attendance.service    # 考勤服务配置
│
├── tools/                   # 工具脚本
│   ├── diagnose_npu.py     # NPU 诊断工具
│   ├── fix_*.py            # 修复工具
│   └── test_dlib.py        # dlib 测试工具
│
└── backups/                 # 备份文件
    └── *.bak               # 各版本备份
```

## 快速开始

### 环境要求
- Python 3.8+
- PyQt5
- OpenCV
- Flask
- dlib (人脸识别)
- ultralytics (YOLO)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动应用

#### 1. 启动 Flask 后端
```bash
cd code
python app.py
```

#### 2. 启动 Qt 界面
```bash
export DISPLAY=:1
cd code
python qt_touch_app.py
```

#### 3. 访问网页
打开浏览器访问: `http://<板子IP>:5000`

## API 接口

### 人脸考勤
- `POST /api/face/attendance` - 人脸考勤
- `GET /api/face/attendance/summary` - 考勤汇总
- `GET /api/face/records` - 考勤记录

### 行为检测
- `POST /api/yolo/predict-frame` - 实时检测
- `GET /api/yolo/stats` - 检测统计
- `POST /api/yolo/capture` - 截图检测

### AI 助手
- `POST /api/chat` - AI 对话

## 系统配置

### 摄像头配置
默认使用 USB 摄像头，设备号为 21。可在 `qt_touch_app.py` 中修改：
```python
CAMERA_INDEX = 21
```

### AI API 配置
系统使用 SiliconFlow API，在启动脚本中配置：
```bash
export SILICONFLOW_API_KEY="your-api-key"
```

### 模型配置
- YOLO 模型: `models/class_behavior_best.pt`
- 人脸识别模型: dlib 预训练模型

## 部署说明

### 1. 系统服务部署
```bash
# 复制服务文件
sudo cp config/zhitong-classroom.service /etc/systemd/system/

# 启用服务
sudo systemctl enable zhitong-classroom.service
sudo systemctl start zhitong-classroom.service
```

### 2. 开机自启动
服务配置为开机自动启动，无需手动干预。

## 常见问题

### Q: 人脸考勤不工作？
A: 请确保：
1. dlib 库已正确安装
2. 模型文件已下载到正确位置
3. 学生照片已上传到系统

### Q: 检测框显示乱码？
A: 请检查系统是否安装了中文字体：
```bash
sudo apt-get install fonts-noto-cjk
```

### Q: 摄像头无法打开？
A: 请检查：
1. 摄像头是否正确连接
2. 设备号是否正确
3. 是否有其他程序占用摄像头

## 开发团队

- 开发者: S1mple56
- 项目时间: 2026年7月

## 许可证

本项目仅供学习交流使用。

## 更新日志

### v1.0.0 (2026-07-09)
- 初始版本发布
- 实现人脸识别考勤功能
- 实现 YOLO 行为检测功能
- 实现 AI 智能助手功能
- 实现 Qt 触摸屏界面
- 实现 Web 网页界面
