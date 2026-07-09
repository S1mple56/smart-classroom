# ELF2 RK3588 部署指南

本文档说明如何将学生行为识别系统部署到 ELF2 RK3588 开发板上。

## 目录

- [快速部署](#快速部署)
- [手动部署](#手动部署)
- [服务管理](#服务管理)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

---

## 快速部署

### 步骤 1: 传输项目到板子

```bash
# 在你的 PC 上执行
scp -r Students-Action-Recognition-master root@<板子IP>:/home/root/
```

### 步骤 2: 在板子上运行部署脚本

```bash
# SSH 连接到板子
ssh root@<板子IP>

# 进入项目目录
cd /home/root/Students-Action-Recognition-master

# 添加执行权限
chmod +x deploy.sh

# 运行部署脚本
./deploy.sh
```

部署脚本会自动完成:
- 安装系统依赖
- 创建 Python 虚拟环境
- 安装 Python 包
- 创建必要的目录
- 配置 systemd 服务
- 创建快捷命令

### 步骤 3: 启动服务

```bash
# 方式 1: 使用快捷命令
./start.sh

# 方式 2: 使用 systemd
systemctl start face-attendance
```

### 步骤 4: 访问服务

在浏览器中访问: `http://<板子IP>:5000`

---

## 手动部署

如果自动部署脚本出现问题，可以手动执行以下步骤:

### 1. 安装系统依赖

```bash
apt update
apt install -y python3 python3-pip python3-venv python3-dev
apt install -y libjpeg-dev zlib1g-dev libpng-dev libopencv-dev
apt install -y v4l-utils ffmpeg
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements_rk3588.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 安装 systemd 服务

```bash
cp face-attendance.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable face-attendance
systemctl start face-attendance
```

---

## 服务管理

### 基本命令

```bash
# 启动服务
./start.sh
# 或
systemctl start face-attendance

# 停止服务
./stop.sh
# 或
systemctl stop face-attendance

# 重启服务
./restart.sh
# 或
systemctl restart face-attendance

# 查看状态
./status.sh
# 或
systemctl status face-attendance

# 查看日志
tail -f log/stdout.log
journalctl -u face-attendance -f
```

### 开机自启

默认情况下服务已设置为开机自启。如需修改:

```bash
# 禁用开机自启
systemctl disable face-attendance

# 启用开机自启
systemctl enable face-attendance
```

---

## 性能优化

### 1. 使用 RK3588 NPU 加速

RK3588 内置 NPU，可以加速 ONNX 模型推理:

```bash
# 在 PC 上优化模型
python optimize_onnx.py

# 将优化后的模型传到板子
scp model/cnn_model_rk3588.onnx root@<板子IP>:/home/root/Students-Action-Recognition-master/model/
```

### 2. 使用 RKNN Toolkit（可选）

如需更深度优化，可以使用 RKNN Toolkit:

```bash
# 在 PC 上安装 RKNN Toolkit
pip install rknn-toolkit

# 运行 RKNN 转换
python optimize_onnx.py --rknn

# 将 .rknn 模型传到板子使用
```

### 3. Gunicorn 多进程（高并发）

如需处理更多并发请求，可以使用 Gunicorn:

```bash
# 安装 gunicorn
pip install gunicorn

# 使用 gunicorn 启动
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 故障排除

### 服务无法启动

```bash
# 查看错误日志
journalctl -u face-attendance -n 50

# 检查端口占用
netstat -tlnp | grep 5000
```

### 权限问题

```bash
# 确保文件有执行权限
chmod +x *.sh

# 确保数据目录可写
chmod -R 755 data/
chmod -R 755 log/
```

### 摄像头无法使用

```bash
# 检查摄像头设备
ls -l /dev/video*

# 检查摄像头权限
ls -l /dev/video0

# 如需修改权限
chmod 777 /dev/video0
```

### 模型推理失败

```bash
# 检查模型文件
ls -la model/

# 测试 ONNX 模型
python -c "import onnx; m = onnx.load('model/cnn_model.onnx')"
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `deploy.sh` | 一键部署脚本 |
| `start.sh` | 启动服务脚本 |
| `stop.sh` | 停止服务脚本 |
| `restart.sh` | 重启服务脚本 |
| `status.sh` | 查看服务状态脚本 |
| `backup.sh` | 备份脚本（排除训练数据） |
| `optimize_onnx.py` | ONNX 模型优化工具 |
| `face-attendance.service` | systemd 服务文件 |
| `requirements_rk3588.txt` | RK3588 依赖清单 |

---

## 网络配置

### 查看板子 IP

```bash
ip addr show
# 或
ifconfig
```

### 防火墙设置

```bash
# 开放 5000 端口
iptables -A INPUT -p tcp --dport 5000 -j ACCEPT

# 或使用 ufw
ufw allow 5000/tcp
```

---

## 技术支持

如遇问题，请检查:

1. 服务日志: `journalctl -u face-attendance -f`
2. 应用日志: `tail -f log/stdout.log`
3. Python 错误: `source venv/bin/activate && python app.py`
