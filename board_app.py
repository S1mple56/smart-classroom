#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校园智能监控系统 - 开发板版本 (RK3588/RK356x)
集成：人脸识别考勤 + YOLO行为检测（可选） + RKNN CNN行为分类
"""

import os
import sys
import json
import base64
import hashlib
import uuid
import shutil
import time
import threading
import queue
import mimetypes
import zipfile
import urllib.request
import urllib.error
from io import BytesIO
from glob import glob
from datetime import datetime
from collections import Counter

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, send_from_directory, session
from flask_cors import CORS

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

# 项目1路径
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'upload')
RESULT_FOLDER = os.path.join(BASE_DIR, 'data')
STUDENTS_INFO_FILE = os.path.join(BASE_DIR, 'data', 'students_info.json')
FACE_ATTENDANCE_FILE = os.path.join(BASE_DIR, 'data', 'attendance_records.json')
USER_DB_FILE = os.path.join(BASE_DIR, 'data', 'users.json')

# 项目2路径
TRY_ORIGIN = os.path.join(BASE_DIR, 'TRY', 'origin')
TRY_TARGET = os.path.join(BASE_DIR, 'TRY', 'target')
CAPTURES_DIR = os.path.join(BASE_DIR, 'data', 'captures')
CAPTURES_ORIGINALS = os.path.join(CAPTURES_DIR, 'originals')
CAPTURES_ANNOTATED = os.path.join(CAPTURES_DIR, 'annotated')
CAPTURES_STATS = os.path.join(CAPTURES_DIR, 'stats')
CAPTURES_ALERTS = os.path.join(CAPTURES_DIR, 'alerts')
ALERTS_FILE = os.path.join(BASE_DIR, 'data', 'alerts.json')
YOLO_ATTENDANCE_FILE = os.path.join(BASE_DIR, 'data', 'yolo_attendance_records.json')
FACE_FEATURES_CACHE = os.path.join(BASE_DIR, 'data', 'face_features_rknn.json')

# 导入板子端 RKNN 人脸识别模块（替代旧的 Haar+直方图方案）
import face_rknn

# 导入节能控制模块
try:
    import energy_controller as energy_ctrl
except ImportError:
    energy_ctrl = None
    print("警告: energy_controller 模块未找到，节能控制系统不可用")

# 确保目录存在
for d in [UPLOAD_FOLDER, TRY_ORIGIN, TRY_TARGET,
          CAPTURES_ORIGINALS, CAPTURES_ANNOTATED, CAPTURES_STATS, CAPTURES_ALERTS]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# Flask 应用
# ============================================================================
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.secret_key = 'campus_monitor_secret_key'
app.config['JSON_AS_ASCII'] = False
CORS(app)

@app.after_request
def set_charset(response):
    if response.content_type and 'text' in response.content_type:
        if 'charset' not in response.content_type:
            response.content_type = response.content_type + '; charset=utf-8'
    return response

# ============================================================================
# 登录系统（JSON 用户数据库 + SHA256 密码哈希）
# ============================================================================

def _load_users():
    if os.path.isfile(USER_DB_FILE):
        with open(USER_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def _save_users(users):
    with open(USER_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def _hash_password(password, salt=None):
    if salt is None:
        salt = uuid.uuid4().hex[:16]
    h = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return salt, h

def _verify_password(password, salt, expected_hash):
    _, actual_hash = _hash_password(password, salt)
    return actual_hash == expected_hash

def _ensure_default_admin():
    users = _load_users()
    if not users:
        salt, h = _hash_password('admin123')
        users['admin'] = {
            'username': 'admin',
            'password_hash': h,
            'salt': salt,
            'role': 'admin',
            'created_at': datetime.utcnow().isoformat()
        }
        _save_users(users)
        return True
    return False

_ensure_default_admin()

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'ok': False, 'error': '请输入用户名和密码'}), 400
    users = _load_users()
    user = users.get(username)
    if user and _verify_password(password, user.get('salt', ''), user.get('password_hash', '')):
        session['logged_in'] = True
        session['username'] = username
        session['role'] = user.get('role', 'user')
        return jsonify({'ok': True, 'role': user.get('role', 'user')})
    return jsonify({'ok': False, 'error': '用户名或密码错误'}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    if not session.get('logged_in'):
        return jsonify({'ok': False, 'error': '请先登录'}), 401
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or len(username) < 2:
        return jsonify({'ok': False, 'error': '用户名至少2个字符'}), 400
    if not password or len(password) < 4:
        return jsonify({'ok': False, 'error': '密码至少4位'}), 400
    users = _load_users()
    if username in users:
        return jsonify({'ok': False, 'error': '用户名已存在'}), 409
    salt, h = _hash_password(password)
    users[username] = {
        'username': username,
        'password_hash': h,
        'salt': salt,
        'role': 'user',
        'created_at': datetime.utcnow().isoformat()
    }
    _save_users(users)
    return jsonify({'ok': True, 'message': '注册成功'})

@app.route('/api/change-password', methods=['POST'])
def api_change_password():
    """修改密码（支持登录页未登录状态下修改）"""
    data = request.get_json(silent=True) or {}
    old_password = data.get('old_password', '')
    new_password = data.get('new_password', '')
    username = data.get('username', '') or session.get('username', '')
    if not username or not old_password or not new_password:
        return jsonify({'ok': False, 'error': '请填写完整'}), 400
    if len(new_password) < 4:
        return jsonify({'ok': False, 'error': '新密码至少4位'}), 400
    users = _load_users()
    user = users.get(username)
    if not user:
        return jsonify({'ok': False, 'error': '用户不存在'}), 404
    if not _verify_password(old_password, user.get('salt', ''), user.get('password_hash', '')):
        return jsonify({'ok': False, 'error': '原密码错误'}), 403
    salt, h = _hash_password(new_password)
    user['password_hash'] = h
    user['salt'] = salt
    _save_users(users)
    return jsonify({'ok': True, 'message': '密码修改成功'})

@app.route('/api/user-info', methods=['GET'])
def api_user_info():
    if not session.get('logged_in'):
        return jsonify({'ok': False, 'error': '未登录'}), 401
    return jsonify({'ok': True, 'username': session.get('username'), 'role': session.get('role', 'user')})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'ok': True})

# ============================================================================
# 管理员功能：用户列表、重置密码、主控密钥
# ============================================================================
MASTER_RESET_KEY = 'reset2024'

@app.route('/api/admin/users', methods=['GET'])
def api_admin_users():
    if session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '无权限'}), 403
    users = _load_users()
    result = []
    for k, v in users.items():
        result.append({
            'username': v.get('username', k),
            'role': v.get('role', 'user'),
            'created_at': v.get('created_at', '-')
        })
    return jsonify({'ok': True, 'users': result})

@app.route('/api/admin/reset-password', methods=['POST'])
def api_admin_reset_password():
    data = request.get_json(silent=True) or {}
    target = data.get('username', '').strip()
    new_password = data.get('new_password', '').strip()
    is_master = data.get('master_key', '') == MASTER_RESET_KEY
    if not is_master and session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '无权限，需要管理员登录或主控密钥'}), 403
    if not target:
        return jsonify({'ok': False, 'error': '请输入要重置的用户名'}), 400
    users = _load_users()
    if target not in users:
        return jsonify({'ok': False, 'error': '用户「' + target + '」不存在'}), 404
    if not new_password or len(new_password) < 4:
        new_password = '123456'
    salt, h = _hash_password(new_password)
    users[target]['password_hash'] = h
    users[target]['salt'] = salt
    _save_users(users)
    return jsonify({'ok': True, 'message': '密码已重置', 'new_password': new_password})

@app.route('/api/admin/delete-user', methods=['POST'])
def api_admin_delete_user():
    if session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '无权限'}), 403
    data = request.get_json(silent=True) or {}
    target = data.get('username', '').strip()
    if not target:
        return jsonify({'ok': False, 'error': '请指定要删除的用户'}), 400
    if target == session.get('username'):
        return jsonify({'ok': False, 'error': '不能删除自己'}), 400
    users = _load_users()
    if target not in users:
        return jsonify({'ok': False, 'error': '用户不存在'}), 404
    del users[target]
    _save_users(users)
    return jsonify({'ok': True, 'message': '已删除用户「' + target + '」'})

# ============================================================================
# AI 智能助手（SiliconFlow 免费模型 / DeepSeek）
# 优先使用 SiliconFlow 免费模型，其次 DeepSeek，都没有则走本地兜底
# ============================================================================
AI_API_KEY = os.environ.get('SILICONFLOW_API_KEY', '') or os.environ.get('DEEPSEEK_API_KEY', '')
AI_API_URL = os.environ.get('AI_API_URL', 'https://api.siliconflow.cn/v1/chat/completions')
AI_MODEL = os.environ.get('AI_MODEL', 'Qwen/Qwen2.5-7B-Instruct')

def _build_system_prompt():
    parts = [
        '你是一个课堂智能监测系统的AI助手。系统部署在RK3588开发板上，集成了人脸考勤和课堂行为检测功能。',
        '你可以帮助用户了解系统功能、查看数据、解答问题。回答要简洁友好。'
    ]
    try:
        students = get_students_list()
        if students:
            names = [s.get('name', '?') for s in students[:20]]
            parts.append(f'当前系统中有{len(students)}名学生：{", ".join(names)}。')
    except:
        pass
    try:
        att_file = os.path.join(RESULT_FOLDER, 'attendance_records.json')
        if os.path.isfile(att_file):
            with open(att_file, 'r', encoding='utf-8') as f:
                att_records = json.load(f)
            if att_records:
                today = datetime.utcnow().strftime('%Y-%m-%d')
                parts.append(f'累计考勤记录{len(att_records)}条。')
    except:
        pass
    return '\n'.join(parts)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(silent=True) or {}
    messages = data.get('messages', [])
    if not messages:
        return jsonify({'ok': False, 'error': 'empty messages'}), 400
    system_prompt = _build_system_prompt()
    full_messages = [{'role': 'system', 'content': system_prompt}] + messages
    if AI_API_KEY:
        try:
            req_body = json.dumps({
                'model': AI_MODEL,
                'messages': full_messages,
                'temperature': 0.7,
                'max_tokens': 600
            }).encode('utf-8')
            req = urllib.request.Request(AI_API_URL, data=req_body, headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + AI_API_KEY
            })
            resp = urllib.request.urlopen(req, timeout=15)
            resp_data = json.loads(resp.read().decode('utf-8'))
            reply = resp_data['choices'][0]['message']['content']
            return jsonify({'ok': True, 'reply': reply})
        except Exception as e:
            return jsonify({'ok': False, 'error': 'AI 服务调用失败: ' + str(e)}), 500
    else:
        reply = _local_reply(messages[-1].get('content', '') if messages else '')
        return jsonify({'ok': True, 'reply': reply, 'local': True})

def _local_reply(question):
    q = question.lower()
    if '出勤' in q or '考勤' in q:
        return '您可以通过「考勤记录」页面查看详细考勤数据，或在「数据总览」页面看汇总统计。'
    if '行为' in q or '举手' in q or '低头' in q:
        return '课堂行为检测支持6类行为：举手、阅读、写作、使用手机、低头、靠在桌子上。您可以在「实时监控」页面查看效果。'
    if '功能' in q or '介绍' in q:
        return '本系统包含以下功能模块：\n1. 实时监控 - 课堂行为检测\n2. 人脸考勤 - 人脸识别签到\n3. 异常告警 - 实时行为预警\n4. 学生档案 - 信息管理\n5. 考勤记录 - 考勤查询\n6. 行为记录 - 行为统计\n7. 数据总览 - 可视化\n8. 节能控制 - 智能管理\n\n请问您想了解哪个功能？'
    if '你好' in q or 'hello' in q.lower():
        return '您好！我是课堂智能监测系统的AI助手，有什么可以帮您的吗？'
    return '这是一个课堂智能监测系统。配置 AI API Key（SiliconFlow 或 DeepSeek）后可启用智能对话。常用问题：\n- 今天出勤情况\n- 系统有哪些功能\n- 行为检测能识别哪些行为'

# ============================================================================
# RKNN CNN 模型初始化（开发板核心：替换 subprocess 调用 main.py）
# ============================================================================
RKNN_MODEL_PATH = os.path.join(BASE_DIR, 'rknn_convert', 'cnn_model.rknn')
ACTION_LABELS = ['听讲', '举手', '低头', '站立', '走动']

_rknn_model = None
_rknn_available = False

try:
    from rknnlite.api import RKNNLite
    _rknn_available = True
except ImportError:
    print("警告: rknnlite 未安装，CNN行为预测不可用")
    RKNNLite = None

def get_rknn_model():
    """获取或初始化 RKNNLite 模型实例（懒加载）"""
    global _rknn_model
    if _rknn_model is None and _rknn_available:
        if not os.path.isfile(RKNN_MODEL_PATH):
            print(f"警告: RKNN 模型文件未找到: {RKNN_MODEL_PATH}")
            return None
        _rknn_model = RKNNLite()
        _rknn_model.load_rknn(RKNN_MODEL_PATH)
        _rknn_model.init_runtime()
        print(f"RKNN CNN 模型已加载: {RKNN_MODEL_PATH}")
    return _rknn_model

def rknn_preprocess(pil_image):
    """预处理图片为 RKNN 推理输入: resize 64x64, RGB, /255.0, NHWC (1,64,64,3)"""
    img = pil_image.convert('RGB').resize((64, 64), Image.BILINEAR)
    img_np = np.array(img).astype('float32') / 255.0                # (64, 64, 3) NHWC
    img_np = np.expand_dims(img_np, axis=0)                        # (1, 64, 64, 3)
    return img_np

def rknn_classify_single(pil_image):
    """使用 RKNN CNN 对单张图片进行分类，返回 {action, confidence} 或 None"""
    model = get_rknn_model()
    if model is None:
        return None
    try:
        input_data = rknn_preprocess(pil_image)
        outputs = model.inference(inputs=[input_data])
        output = outputs[0].flatten()
        pred_idx = int(np.argmax(output))
        # softmax
        exp_out = np.exp(output - np.max(output))
        prob = exp_out / exp_out.sum()
        return {
            'action': ACTION_LABELS[pred_idx],
            'confidence': f'{prob[pred_idx] * 100:.1f}%'
        }
    except Exception as e:
        return {'action': '识别失败', 'confidence': '0%', 'error': str(e)}

def cnn_predict_rknn():
    """
    使用 RKNN CNN 模型对 eval 数据进行批量预测（替代 main.py 的 subprocess 调用）。
    遍历 data/eval/students/ 下每个学生文件夹，投票预测行为类别，
    结果写入 data/result.txt。
    """
    eval_dir = os.path.join(BASE_DIR, 'data', 'eval', 'students')
    students_info = {}
    if os.path.exists(STUDENTS_INFO_FILE):
        with open(STUDENTS_INFO_FILE, 'r', encoding='utf-8') as f:
            students_info = json.load(f).get('students', {})

    model = get_rknn_model()
    if model is None:
        raise RuntimeError("RKNN CNN 模型不可用")

    result_path = os.path.join(RESULT_FOLDER, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("序号\t姓名\t学号\t班级\t行为\t置信度\n")

    people_number = 0
    student_dirs = sorted([d for d in os.listdir(eval_dir)
                           if os.path.isdir(os.path.join(eval_dir, d)) and d != '.DS_Store'])

    for class_dir in student_dirs:
        people_number += 1
        person_path = os.path.join(eval_dir, class_dir)
        student_info = students_info.get(class_dir, {})

        result_list = []  # 存放该学生每张照片的预测类别索引
        img_files = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        batch_inputs = []
        for img_file in img_files:
            img_path = os.path.join(person_path, img_file)
            try:
                pil_img = Image.open(img_path)
                batch_inputs.append(rknn_preprocess(pil_img))
            except Exception:
                continue

        if not batch_inputs:
            continue

        # 批量推理：逐张（rknnlite 通常按单张输入）
        for inp in batch_inputs:
            try:
                outputs = model.inference(inputs=[inp])
                pred_idx = int(np.argmax(outputs[0]))
                result_list.append(pred_idx)
            except Exception:
                continue

        if not result_list:
            continue

        # 投票：取出现次数最多的类别
        counter = Counter(result_list)
        result = counter.most_common(1)[0][0]
        total = len(result_list)
        confidence = counter[result] / total * 100

        name = student_info.get('name', '未知')
        student_id = student_info.get('student_id', '未知')
        class_name = student_info.get('class', '未知')

        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f"{people_number}\t{name}\t{student_id}\t{class_name}\t{ACTION_LABELS[result]}\t{confidence:.1f}%\n")

    print(f"CNN预测完成！结果已保存到 {result_path}")

# ============================================================================
# YOLO 模型初始化（开发板：ultralytics YOLO 优先，使用 6 类模型）
# ============================================================================
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'class_behavior_best.pt')
EXPECTED_TRACKED = ("hand-raising", "reading", "writing", "using phone", "bowing the head", "leaning over the table")
CHINESE_LABELS = {"hand-raising": "举手", "reading": "阅读", "writing": "写作", "using phone": "使用手机", "bowing the head": "低头", "leaning over the table": "靠在桌子上"}

yolo_model = None
TRACKED_CLASSES = EXPECTED_TRACKED

try:
    from ultralytics import YOLO
    _ultralytics_available = True
except ImportError:
    YOLO = None
    _ultralytics_available = False
    print("警告: ultralytics 未安装，YOLO 实时检测不可用")

if _ultralytics_available and os.path.isfile(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    TRACKED_CLASSES = tuple([str(v) for k, v in sorted(yolo_model.names.items(), key=lambda x: int(x[0]))])
    print(f"YOLO 模型已加载: {YOLO_MODEL_PATH} — 6类行为检测")
else:
    if not _ultralytics_available:
        print("YOLO 不可用: 缺少 ultralytics 依赖")
    elif not os.path.isfile(YOLO_MODEL_PATH):
        print(f"YOLO 模型未找到: {YOLO_MODEL_PATH}")

# ============================================================================
# OpenCV 摄像头（项目1）
# ============================================================================
DEFAULT_CAMERA_ID = 21      # RK3588 板子 USB 摄像头的 V4L2 设备号
_cameras = {}               # camera_id -> cv2.VideoCapture | None (None = 已确认不可用)
_cameras_lock = threading.Lock()

try:
    import cv2
    _cv2_available = True
except ImportError:
    cv2 = None
    _cv2_available = False
    print("警告: opencv-python 未安装，服务端摄像头不可用")

def get_camera(camera_id=DEFAULT_CAMERA_ID):
    """按索引获取摄像头，支持多个摄像头并存（如USB外接）"""
    if not _cv2_available:
        return None
    with _cameras_lock:
        if camera_id not in _cameras or _cameras[camera_id] is None:
            try:
                cap = cv2.VideoCapture(camera_id)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                if not cap.isOpened():
                    cap.release()
                    _cameras[camera_id] = None
                    return None
                _cameras[camera_id] = cap
            except Exception:
                _cameras[camera_id] = None
                return None
    return _cameras.get(camera_id)

def release_camera(camera_id=DEFAULT_CAMERA_ID):
    """释放指定摄像头，不传 id 则释放全部"""
    global _cameras
    with _cameras_lock:
        if camera_id is None:
            for cap in _cameras.values():
                if cap is not None:
                    cap.release()
            _cameras = {}
        elif camera_id in _cameras and _cameras[camera_id] is not None:
            _cameras[camera_id].release()
            _cameras[camera_id] = None

def generate_frames(camera_id=DEFAULT_CAMERA_ID):
    camera = get_camera(camera_id)
    if camera is None:
        return
    try:
        while True:
            with _cameras_lock:
                cap = _cameras.get(camera_id)
                if cap is None:
                    break
                success, frame = cap.read()
            if not success:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        pass
    finally:
        release_camera(camera_id)

# ============================================================================
# 节能控制系统（项目3）— 人数统计 + GPIO 继电器
# ============================================================================
_energy_manager = None
_energy_started = False

def init_energy_manager():
    """初始化节能管理器（按需启动，仅在 /video_feed 首次访问时调用）"""
    global _energy_manager, _energy_started
    if _energy_started:
        return
    try:
        if energy_ctrl is None:
            return
        _energy_manager = energy_ctrl.EnergyManager(
            get_camera_fn=lambda: get_camera(DEFAULT_CAMERA_ID),  # 共享摄像头
            config={
                "simulate": "--real" not in sys.argv,  # 默认模拟，--real 启用 GPIO
                "camera_id": DEFAULT_CAMERA_ID,
                "detect_interval": 2.0,
                "model_path": os.path.join(BASE_DIR, "yolo11n.pt"),
                "confidence": 0.3,
                "gpio_pins": {"light": 17, "fan": 27, "ac": 22},
            }
        )
        _energy_manager.start()
        _energy_started = True
        print("[*] 节能控制系统已启动（按需）")
    except Exception as e:
        print(f"[!] 节能控制系统启动失败: {e}")

def stop_energy_manager():
    """停止节能管理器"""
    global _energy_manager, _energy_started
    if _energy_manager is not None:
        try:
            _energy_manager.stop()
        except Exception:
            pass
        _energy_manager = None
    _energy_started = False

# ============================================================================
# 预警系统（项目2）
# ============================================================================
alerts_lock = threading.Lock()
alerts_queue = queue.Queue()
try:
    if os.path.isfile(ALERTS_FILE):
        with open(ALERTS_FILE, 'r', encoding='utf-8') as f:
            alerts_store = json.load(f)
    else:
        alerts_store = []
except Exception:
    alerts_store = []

def save_alerts():
    try:
        with alerts_lock:
            with open(ALERTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(alerts_store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def push_alert(alert):
    with alerts_lock:
        alerts_store.insert(0, alert)
        del alerts_store[200:]
        save_alerts()
    try:
        alerts_queue.put(alert, block=False)
    except Exception:
        pass

# ============================================================================
# YOLO 考勤存储（项目2）
# ============================================================================
yolo_attendance_lock = threading.Lock()
try:
    if os.path.isfile(YOLO_ATTENDANCE_FILE):
        with open(YOLO_ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            yolo_attendance_store = json.load(f)
    else:
        yolo_attendance_store = []
except Exception:
    yolo_attendance_store = []

def save_yolo_attendance():
    try:
        with yolo_attendance_lock:
            with open(YOLO_ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
                json.dump(yolo_attendance_store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ============================================================================
# YOLO 工具函数（项目2）—— 简化版本，直接使用 yolo_model.predict()
# ============================================================================

def _stats_from_result(result):
    """从单个 YOLO result 对象提取统计数据"""
    counts = {}
    try:
        boxes = getattr(result, 'boxes', None)
        if boxes is not None:
            cls_ids = getattr(boxes, 'cls', None)
            if hasattr(cls_ids, 'cpu'):
                cls_ids = cls_ids.cpu().numpy().tolist()
            elif hasattr(cls_ids, 'numpy'):
                cls_ids = cls_ids.numpy().tolist()
            elif isinstance(cls_ids, (list, tuple, np.ndarray)):
                cls_ids = list(cls_ids)
            else:
                cls_ids = []
            class_names = getattr(yolo_model, 'names', {}) if yolo_model else {}
            for cid in cls_ids:
                try:
                    cname = class_names.get(int(cid), f'class_{int(cid)}')
                except Exception:
                    cname = f'class_{cid}'
                counts[str(cname)] = counts.get(str(cname), 0) + 1
    except Exception:
        pass
    return counts

def _boxes_from_result(result, image):
    """从 YOLO result 提取归一化边界框"""
    boxes_out = []
    try:
        b = getattr(result, 'boxes', None)
        if b is None:
            return boxes_out
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        clsids = b.cls.cpu().numpy()
        img_w, img_h = image.size
        class_names = getattr(yolo_model, 'names', {}) if yolo_model else {}
        for bb, cf, cid in zip(xyxy, confs, clsids):
            x1, y1, x2, y2 = map(float, bb)
            cid = int(cid)
            if isinstance(class_names, (list, tuple)):
                cname = class_names[cid] if cid < len(class_names) else str(cid)
            elif isinstance(class_names, dict):
                cname = class_names.get(cid, str(cid))
            else:
                cname = str(cid)
            boxes_out.append({
                'xyxy': [round(x1/(img_w or 1), 4), round(y1/(img_h or 1), 4), round(x2/(img_w or 1), 4), round(y2/(img_h or 1), 4)],
                'conf': round(float(cf), 4), 'class_id': cid, 'class_name': str(cname)
            })
    except Exception:
        pass
    return boxes_out

def _merge_stats(all_stats):
    """合并多帧统计数据"""
    total_counts = {}
    max_per_frame = {}
    for frame_counts in all_stats:
        for cname, cnt in frame_counts.items():
            total_counts[cname] = total_counts.get(cname, 0) + cnt
            max_per_frame[cname] = max(max_per_frame.get(cname, 0), int(cnt))
    tracked = set(TRACKED_CLASSES) if TRACKED_CLASSES else set(EXPECTED_TRACKED)
    tracked_totals = {name: int(total_counts.get(name, 0)) for name in tracked}
    tracked_max = {name: int(max_per_frame.get(name, 0)) for name in tracked}
    is_video = len(all_stats) > 1
    return {
        "frame_count": int(len(all_stats)),
        "total_detections": int(sum(total_counts.values())),
        "counts": {k: int(v) for k, v in total_counts.items()},
        "tracked_totals": tracked_totals,
        "tracked_max_per_frame": tracked_max,
        "display_counts": tracked_max if is_video else tracked_totals,
        "display_mode": "max_per_frame" if is_video else "image_total",
        "display_mode_label": "视频逐帧峰值" if is_video else "图片检测数量",
    }

def ensure_tracked_keys(stats):
    """确保统计结果中包含所有预期行为类别"""
    if not isinstance(stats, dict):
        return stats
    tracked = set(TRACKED_CLASSES) if TRACKED_CLASSES else set(EXPECTED_TRACKED)
    for key in tracked:
        for section in ['counts', 'tracked_totals', 'tracked_max_per_frame', 'display_counts']:
            if section not in stats:
                stats[section] = {}
            if key not in stats[section]:
                stats[section][key] = 0
    return stats

# ============================================================================
# 人脸识别工具函数（项目1）—— 已升级为 RKNN NPU 方案
# ============================================================================
# 人脸检测使用 OpenCV DNN (YuNet) + Haar Cascade 回退
# 特征提取使用 NPU 加速的 MobileFaceNet/ArcFace 模型

def match_student(image_path):
    """
    使用 RKNN NPU 进行人脸匹配 —— 替代原来的 Haar + 直方图方案。
    接口与旧版完全兼容。
    """
    return face_rknn.match_face(
        image_path=image_path,
        features_cache_path=FACE_FEATURES_CACHE,
        upload_dir=UPLOAD_FOLDER,
        students_info_path=STUDENTS_INFO_FILE,
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def update_student_info(person_id, name, student_id, class_name):
    if os.path.exists(STUDENTS_INFO_FILE):
        with open(STUDENTS_INFO_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {'students': {}}
    data['students'][person_id] = {'name': name, 'student_id': student_id, 'class': class_name}
    with open(STUDENTS_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_students_list():
    students = []
    if os.path.exists(STUDENTS_INFO_FILE):
        with open(STUDENTS_INFO_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for person_id, info in data.get('students', {}).items():
            folder_path = os.path.join(UPLOAD_FOLDER, person_id)
            photo_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')]) if os.path.exists(folder_path) else []
            students.append({
                'person_id': person_id,
                'name': info.get('name', '未知'),
                'student_id': info.get('student_id', '未知'),
                'class': info.get('class', '未知'),
                'photo_count': len(photo_files),
                'photos': photo_files,
            })
    return students

def recognize_action(image_path):
    """行为识别（开发板版本：优先使用 RKNN CNN）"""
    try:
        img = Image.open(image_path).convert('RGB')
        result = rknn_classify_single(img)
        if result is None:
            return {'action': '识别失败', 'confidence': '0%', 'error': 'RKNN模型不可用'}
        return result
    except Exception as e:
        return {'action': '识别失败', 'confidence': '0%', 'error': str(e)}

def add_face_attendance_record(student_id, student_name, student_class, status='present', timestamp=None):
    if not timestamp:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(FACE_ATTENDANCE_FILE):
        with open(FACE_ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        records = []
    records.append({'student_id': student_id, 'student_name': student_name, 'student_class': student_class, 'status': status, 'timestamp': timestamp})
    with open(FACE_ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def prepare_eval_data():
    eval_dir = os.path.join(BASE_DIR, 'data', 'eval', 'students')
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    for person_folder in os.listdir(UPLOAD_FOLDER):
        src_path = os.path.join(UPLOAD_FOLDER, person_folder)
        dst_path = os.path.join(eval_dir, person_folder)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
    zip_path = os.path.join(BASE_DIR, 'data', 'zipped-eval', 'students.zip')
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for person_folder in os.listdir(eval_dir):
            person_path = os.path.join(eval_dir, person_folder)
            for file in os.listdir(person_path):
                if file.endswith('.jpg'):
                    zf.write(os.path.join(person_path, file), f'students/{person_folder}/{file}')

def read_prediction_results():
    result_file = os.path.join(RESULT_FOLDER, 'result.txt')
    if not os.path.exists(result_file):
        return []
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            results.append({'number': parts[0], 'name': parts[1], 'student_id': parts[2], 'class': parts[3], 'action': parts[4], 'confidence': parts[5]})
    return results

# ============================================================================
# 模板过滤器
# ============================================================================
@app.template_filter('get_action_class')
def get_action_class(action):
    return {'听讲': 'listening', '举手': 'hand', '低头': 'down', '站立': 'standing', '走动': 'walking'}.get(action, 'walking')

# ============================================================================
# 路由：主页（SPA）
# ============================================================================
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect('/login')
    students = get_students_list()
    total_photos = sum(s.get('photo_count', 0) for s in students)
    return render_template('index.html', students=students, total_photos=total_photos)

# ============================================================================
# 路由：学生管理（项目1）
# ============================================================================
@app.route('/api/students', methods=['GET'])
def api_students():
    students = get_students_list()
    total_photos = sum(s.get('photo_count', 0) for s in students)
    return jsonify({'students': students, 'total_photos': total_photos})

@app.route('/api/students/add', methods=['POST'])
def api_add_student():
    student_name = request.form.get('student_name', '').strip()
    student_id = request.form.get('student_id', '').strip()
    student_class = request.form.get('student_class', '').strip()
    if 'photos' not in request.files:
        return jsonify({'ok': False, 'error': '请选择图片'}), 400
    files = request.files.getlist('photos')
    student_folder = os.path.join(UPLOAD_FOLDER, f'person_{student_id}')
    os.makedirs(student_folder, exist_ok=True)
    existing = [f for f in os.listdir(student_folder) if f.startswith('photo_') and f.endswith('.jpg')]
    start_index = len(existing) + 1
    count = 0
    for i, file in enumerate(files):
        if file.filename and allowed_file(file.filename):
            file.save(os.path.join(student_folder, f'photo_{start_index + i}.jpg'))
            count += 1
    update_student_info(f'person_{student_id}', student_name, student_id, student_class)
    # 重建人脸特征数据库（NPU）
    face_rknn.rebuild_features(UPLOAD_FOLDER, STUDENTS_INFO_FILE, FACE_FEATURES_CACHE)
    return jsonify({'ok': True, 'message': f'成功添加学生: {student_name}（{count} 张照片）'})

@app.route('/api/students/delete/<person_id>', methods=['POST'])
def api_delete_student(person_id):
    student_folder = os.path.join(UPLOAD_FOLDER, person_id)
    if os.path.exists(student_folder):
        shutil.rmtree(student_folder)
    if os.path.exists(STUDENTS_INFO_FILE):
        with open(STUDENTS_INFO_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if person_id in data.get('students', {}):
            name = data['students'][person_id].get('name', '未知')
            del data['students'][person_id]
            with open(STUDENTS_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # 重建人脸特征数据库（NPU）
            face_rknn.rebuild_features(UPLOAD_FOLDER, STUDENTS_INFO_FILE, FACE_FEATURES_CACHE)
            return jsonify({'ok': True, 'message': f'已删除: {name}'})
    return jsonify({'ok': False, 'error': '未找到该学生'}), 404

@app.route('/api/students/clear', methods=['POST'])
def api_clear_students():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)
    result_file = os.path.join(RESULT_FOLDER, 'result.txt')
    if os.path.exists(result_file):
        os.remove(result_file)
    # 清除人脸特征缓存
    if os.path.exists(FACE_FEATURES_CACHE):
        os.remove(FACE_FEATURES_CACHE)
    return jsonify({'ok': True, 'message': '已清空所有学生数据'})

@app.route('/api/students/photos/<person_id>/<filename>')
def api_student_photo(person_id, filename):
    """提供学生照片文件"""
    folder = os.path.join(UPLOAD_FOLDER, person_id)
    if not os.path.isdir(folder):
        return jsonify({'error': 'not found'}), 404
    safe_name = os.path.basename(filename)
    return send_from_directory(folder, safe_name)

@app.route('/api/students/delete-photo', methods=['POST'])
def api_delete_photo():
    """删除学生的一张照片"""
    data = request.get_json(silent=True) or {}
    person_id = data.get('person_id', '').strip()
    filename = data.get('filename', '').strip()
    if not person_id or not filename:
        return jsonify({'ok': False, 'error': '缺少参数'}), 400
    safe_name = os.path.basename(filename)
    photo_path = os.path.join(UPLOAD_FOLDER, person_id, safe_name)
    if not os.path.isfile(photo_path):
        return jsonify({'ok': False, 'error': '照片不存在'}), 404
    os.remove(photo_path)
    face_rknn.rebuild_features(UPLOAD_FOLDER, STUDENTS_INFO_FILE, FACE_FEATURES_CACHE)
    return jsonify({'ok': True, 'message': f'已删除 {safe_name}'})

# ============================================================================
# 路由：人脸考勤（项目1）
# ============================================================================
@app.route('/api/face/attendance', methods=['POST'])
def api_face_attendance():
    if 'captured_image' in request.form:
        captured_image_data = request.form.get('captured_image')
        if captured_image_data and captured_image_data.startswith('data:image/'):
            try:
                image_bytes = base64.b64decode(captured_image_data.split(',')[1])
                temp_path = os.path.join(BASE_DIR, 'data', 'temp_recognize', 'temp_attendance.jpg')
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(image_bytes)
                matched = match_student(temp_path)
                if matched['name'] != '未知':
                    add_face_attendance_record(matched['student_id'], matched['name'], matched['class'])
                    return jsonify({'ok': True, 'message': f'考勤成功！{matched["name"]}（匹配度 {matched["match_confidence"]}）', 'student': matched})
                elif matched.get('is_stranger'):
                    return jsonify({'ok': False, 'message': f'识别为陌生人（匹配度 {matched["match_confidence"]}）'})
                else:
                    return jsonify({'ok': False, 'message': '系统中尚无学生数据'})
            except Exception as e:
                return jsonify({'ok': False, 'message': f'图片处理失败: {str(e)}'})
        return jsonify({'ok': False, 'message': '图片数据格式不正确'})

    elif 'recognize_photo' in request.files:
        file = request.files['recognize_photo']
        if file.filename and allowed_file(file.filename):
            temp_path = os.path.join(BASE_DIR, 'data', 'temp_recognize', 'temp_attendance.jpg')
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            matched = match_student(temp_path)
            if matched['name'] != '未知':
                add_face_attendance_record(matched['student_id'], matched['name'], matched['class'])
                return jsonify({'ok': True, 'message': f'考勤成功！{matched["name"]}（匹配度 {matched["match_confidence"]}）', 'student': matched})
            elif matched.get('is_stranger'):
                return jsonify({'ok': False, 'message': f'识别为陌生人（匹配度 {matched["match_confidence"]}）'})
            else:
                return jsonify({'ok': False, 'message': '系统中尚无学生数据'})
        return jsonify({'ok': False, 'message': '文件格式不正确'})

    elif 'student_id' in request.form:
        student_id = request.form.get('student_id')
        students_info = {}
        if os.path.exists(STUDENTS_INFO_FILE):
            with open(STUDENTS_INFO_FILE, 'r', encoding='utf-8') as f:
                students_info = json.load(f).get('students', {})
        for person_id, info in students_info.items():
            if info.get('student_id') == student_id:
                add_face_attendance_record(info['student_id'], info['name'], info['class'])
                return jsonify({'ok': True, 'message': f'考勤成功！{info["name"]}'})
        return jsonify({'ok': False, 'message': '未找到该学生'})

    return jsonify({'ok': False, 'message': '缺少参数'}), 400

@app.route('/api/face/attendance/video', methods=['POST'])
def api_face_attendance_video():
    if 'video' not in request.files:
        return jsonify({'ok': False, 'error': '请上传视频文件'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'ok': False, 'error': '文件名为空'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ('mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'):
        return jsonify({'ok': False, 'error': f'不支持的视频格式: .{ext}'}), 400

    interval_sec = float(request.form.get('interval', '2'))

    temp_dir = os.path.join(BASE_DIR, 'data', 'temp_video')
    os.makedirs(temp_dir, exist_ok=True)
    safe_name = f"face_video_{int(time.time())}.{ext}"
    video_path = os.path.join(temp_dir, safe_name)
    file.save(video_path)

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'ok': False, 'error': '无法打开视频文件'}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        frame_interval = max(1, int(fps * interval_sec))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        recognized_students = {}
        processed_frames = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                processed_frames += 1
                # 转换为RGB后保存（cv2.imwrite 原样写入，必须转 RGB，否则 PIL 读取时红蓝颠倒）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = os.path.join(temp_dir, f'temp_frame_{frame_idx}.jpg')
                cv2.imwrite(frame_path, frame_rgb)
                try:
                    matched = match_student(frame_path)
                    if matched['name'] != '未知' and not matched.get('is_stranger'):
                        sid = matched['student_id']
                        if sid not in recognized_students:
                            recognized_students[sid] = matched
                            recognized_students[sid]['first_frame'] = frame_idx
                            add_face_attendance_record(sid, matched['name'], matched['class'])
                except Exception:
                    pass
                try:
                    os.remove(frame_path)
                except Exception:
                    pass
            frame_idx += 1

        cap.release()

        return jsonify({
            'ok': True,
            'message': f'视频处理完成，共 {total_frames} 帧，采样 {processed_frames} 帧，识别到 {len(recognized_students)} 人',
            'recognized': list(recognized_students.values()),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'视频处理失败: {str(e)}'}), 500
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass

@app.route('/api/face/records', methods=['GET'])
def api_face_records():
    records = []
    if os.path.exists(FACE_ATTENDANCE_FILE):
        with open(FACE_ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
    records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify({'records': records})

@app.route('/api/face/attendance/summary', methods=['GET'])
def api_face_attendance_summary():
    today = time.strftime('%Y-%m-%d')
    attended_set = {}
    if os.path.exists(FACE_ATTENDANCE_FILE):
        with open(FACE_ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        for r in records:
            ts = r.get('timestamp', '')
            if ts.startswith(today):
                sid = r.get('student_id', '')
                if sid and sid not in attended_set:
                    attended_set[sid] = {
                        'status': r.get('status', 'present'),
                        'timestamp': ts
                    }
    students_all = []
    if os.path.exists(STUDENTS_INFO_FILE):
        with open(STUDENTS_INFO_FILE, 'r', encoding='utf-8') as f:
            all_info = json.load(f).get('students', {})
        for person_id, info in all_info.items():
            sid = info.get('student_id', '')
            if sid in attended_set:
                status = attended_set[sid]['status']
                check_time = attended_set[sid]['timestamp']
            else:
                status = 'absent'
                check_time = ''
            students_all.append({
                'student_name': info.get('name', ''),
                'student_id': sid,
                'student_class': info.get('class', ''),
                'status': status,
                'check_time': check_time
            })
    return jsonify({'students': students_all, 'date': today})

@app.route('/api/quick_recognize', methods=['POST'])
def api_quick_recognize():
    if 'recognize_photo' not in request.files:
        return jsonify({'ok': False, 'error': '请选择图片'}), 400
    file = request.files['recognize_photo']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'ok': False, 'error': '文件格式不正确'}), 400
    temp_path = os.path.join(BASE_DIR, 'data', 'temp_recognize', 'temp.jpg')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    matched = match_student(temp_path)
    action_result = recognize_action(temp_path)
    with open(temp_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    return jsonify({
        'ok': True,
        'student': matched,
        'action': action_result['action'],
        'confidence': action_result['confidence'],
        'image_data': image_data
    })

# ============================================================================
# 路由：CNN 批量识别（项目1）—— 开发板版本：使用 RKNN，不再 subprocess
# ============================================================================
@app.route('/api/cnn/predict', methods=['POST'])
def api_cnn_predict():
    if not os.listdir(UPLOAD_FOLDER):
        return jsonify({'ok': False, 'error': '请先添加学生数据'}), 400
    if not _rknn_available:
        return jsonify({'ok': False, 'error': 'RKNN 模型不可用，请安装 rknnlite'}), 500
    prepare_eval_data()
    try:
        cnn_predict_rknn()
        return jsonify({'ok': True, 'results': read_prediction_results()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/cnn/results', methods=['GET'])
def api_cnn_results():
    return jsonify({'results': read_prediction_results()})

# ============================================================================
# 路由：YOLO 实时检测（项目2）—— 使用 class_Detection_end 的 6 类模型
# ============================================================================
@app.route('/api/yolo/model-info', methods=['GET'])
def api_yolo_model_info():
    return jsonify({
        'model_path': YOLO_MODEL_PATH,
        'model_name': os.path.basename(YOLO_MODEL_PATH),
        'model_classes': getattr(yolo_model, 'names', None) if yolo_model else None,
        'yolo_available': yolo_model is not None,
        'board_mode': True,
    })

@app.route('/api/yolo/predict', methods=['POST'])
def api_yolo_predict():
    if not yolo_model:
        return jsonify({'error': 'YOLO 模型未加载'}), 503
    payload = request.get_json(silent=True) or {}
    filename = payload.get('file')
    media_type = payload.get('type', 'image')
    if not filename:
        return jsonify({'error': 'file missing'}), 400
    safe_name = os.path.basename(filename)
    src_path = os.path.join(TRY_ORIGIN, safe_name)
    if not os.path.isfile(src_path):
        return jsonify({'error': 'file not found'}), 404

    if media_type == 'image':
        import cv2
        data = np.fromfile(src_path, np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        result = yolo_model.predict(img_bgr, conf=0.3, verbose=False)[0]
        stats = _stats_from_result(result)
        stats = _merge_stats([stats])
        stats = ensure_tracked_keys(stats)
        try:
            annotated = result.plot()
            image = Image.fromarray(annotated)
        except Exception:
            image = Image.open(src_path).convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        boxes_out = _boxes_from_result(result, image)

        if stats.get('display_counts', {}).get('leaning over the table', 0) > 0:
            push_alert({'id': int(time.time()*1000), 'timestamp': datetime.utcnow().isoformat()+'Z', 'level': 'warning', 'type': 'abnormal', 'count': stats['display_counts']['leaning over the table'], 'meta': {'source': 'predict', 'file': safe_name}, 'image_base64': image_b64})

        return jsonify({'image_base64': image_b64, 'stats': stats, 'boxes': boxes_out})

    elif media_type == 'video':
        import cv2
        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = safe_name.rsplit('.', 1)[0] + '_annotated.mp4'
        out_path = os.path.join(TRY_TARGET, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        all_stats = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = yolo_model.predict(frame, conf=0.3, verbose=False)[0]
            f_stats = _stats_from_result(result)
            all_stats.append(f_stats)
            annotated = result.plot()
            writer.write(annotated)
        cap.release()
        writer.release()
        stats = _merge_stats(all_stats)
        stats = ensure_tracked_keys(stats)
        rel_path = os.path.relpath(out_path, TRY_TARGET).replace('\\', '/')
        return jsonify({'video_url': f'/api/yolo/target/{rel_path}', 'stats': stats})

    return jsonify({'error': f'unsupported type: {media_type}'}), 400

@app.route('/api/yolo/upload-video', methods=['POST'])
def api_yolo_upload_video():
    if not yolo_model:
        return jsonify({'error': 'YOLO 模型未加载'}), 503
    if 'video' not in request.files:
        return jsonify({'error': '请上传视频文件'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': '文件名为空'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ('mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'):
        return jsonify({'error': f'不支持的视频格式: .{ext}'}), 400

    safe_name = f"upload_{int(time.time())}_{file.filename}"
    src_path = os.path.join(TRY_ORIGIN, safe_name)
    file.save(src_path)

    try:
        import cv2
        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = safe_name.rsplit('.', 1)[0] + '_annotated.mp4'
        out_path = os.path.join(TRY_TARGET, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        all_stats = []
        per_frame = []

        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = yolo_model.predict(frame, conf=0.3, verbose=False)[0]
            f_stats = _stats_from_result(result)
            all_stats.append(f_stats)
            per_frame.append({'frame': fi, 'detections': f_stats})
            annotated = result.plot()
            writer.write(annotated)
            fi += 1

        cap.release()
        writer.release()
        stats = _merge_stats(all_stats)
        stats = ensure_tracked_keys(stats)
        rel_path = os.path.relpath(out_path, TRY_TARGET).replace('\\', '/')

        return jsonify({
            'ok': True,
            'video_url': f'/api/yolo/target/{rel_path}',
            'stats': stats,
            'per_frame': per_frame,
            'frame_count': len(all_stats),
        })
    except Exception as e:
        return jsonify({'error': f'推理失败: {str(e)}'}), 500
    finally:
        try:
            if os.path.isfile(src_path):
                os.remove(src_path)
        except Exception:
            pass

@app.route('/api/yolo/predict-frame', methods=['POST'])
def api_yolo_predict_frame():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image')
    if not image_data:
        return jsonify({'error': 'image missing'}), 400
    if isinstance(image_data, str) and image_data.startswith('data:'):
        image_b64 = image_data.split(',', 1)[1]
    else:
        image_b64 = image_data
    try:
        decoded = base64.b64decode(image_b64)
        img = Image.open(BytesIO(decoded)).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'invalid image', 'detail': str(e)}), 400

    # ── 方案 A: ultralytics YOLO（6类目标检测，CPU）──
    if yolo_model is not None:
        try:
            arr = np.asarray(img)[:, :, ::-1].copy()
            result = yolo_model.predict(arr, conf=0.3, verbose=False)[0]
            stats = _stats_from_result(result)
            stats = _merge_stats([stats])
            stats = ensure_tracked_keys(stats)
            try:
                annotated = result.plot()
                out_img = Image.fromarray(annotated)
            except Exception:
                out_img = img
            buffer = BytesIO()
            out_img.save(buffer, format='PNG')
            boxes_out = _boxes_from_result(result, out_img)

            if stats.get('display_counts', {}).get('leaning over the table', 0) > 0:
                push_alert({'id': int(time.time()*1000), 'timestamp': datetime.utcnow().isoformat()+'Z', 'level': 'warning', 'type': 'abnormal', 'count': stats['display_counts']['leaning over the table'], 'meta': {'source': 'predict-frame'}})

            return jsonify({'image_base64': base64.b64encode(buffer.getvalue()).decode('utf-8'), 'stats': stats, 'boxes': boxes_out, 'mode': 'ultralytics_yolo'})
        except Exception as e:
            print(f"YOLO predict-frame 异常: {e}")

    # ── 方案 B: RKNN CNN 回退（5类行为分类，NPU加速）──
    if _rknn_available:
        try:
            result = rknn_classify_single(img)
            if result is None:
                return jsonify({'error': 'RKNN CNN 推理失败'}), 500
            # 将 CNN 单分类结果包装为兼容 statistics 格式
            cnn_stats = _merge_stats([{result['action']: 1}])
            cnn_stats = ensure_tracked_keys(cnn_stats)
            cnn_stats['display_mode_label'] = 'RKNN CNN 行为分类'
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return jsonify({
                'image_base64': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                'stats': cnn_stats,
                'boxes': [],
                'cnn_result': result,
                'mode': 'rknn_cnn',
            })
        except Exception as e:
            print(f"RKNN CNN predict-frame 异常: {e}")

    return jsonify({'error': 'YOLO 和 RKNN 模型均不可用'}), 503

@app.route('/api/yolo/capture', methods=['POST', 'OPTIONS'])
def api_yolo_capture():
    if request.method == 'OPTIONS':
        return ('', 200)
    try:
        ct = (request.headers.get('Content-Type') or '').lower()
        if ct.startswith('text/plain'):
            image_data = request.get_data(as_text=True)
        else:
            payload = request.get_json(silent=True) or {}
            image_data = payload.get('image')
        if not image_data:
            return jsonify({'error': 'image missing'}), 400
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_b64 = image_data.split(',', 1)[1]
        else:
            image_b64 = image_data
        decoded = base64.b64decode(image_b64)
        img = Image.open(BytesIO(decoded)).convert('RGB')

        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
        orig_name = f'capture_{timestamp}.png'
        img.save(os.path.join(CAPTURES_ORIGINALS, orig_name))

        # ── 方案 A: ultralytics YOLO ──
        if yolo_model is not None:
            try:
                arr = np.asarray(img)[:, :, ::-1].copy()
                result = yolo_model.predict(arr, conf=0.3, verbose=False)[0]
                stats = _stats_from_result(result)
                stats = _merge_stats([stats])
                stats = ensure_tracked_keys(stats)
                try:
                    out_img = Image.fromarray(result.plot())
                except Exception:
                    out_img = img
                annot_name = f'capture_annot_{timestamp}.png'
                out_img.save(os.path.join(CAPTURES_ANNOTATED, annot_name))

                stats_name = f'capture_stats_{timestamp}.json'
                with open(os.path.join(CAPTURES_STATS, stats_name), 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)

                if stats.get('display_counts', {}).get('leaning over the table', 0) > 0:
                    alert = {'id': int(time.time()*1000), 'timestamp': datetime.utcnow().isoformat()+'Z', 'level': 'warning', 'type': 'abnormal', 'count': stats['display_counts']['leaning over the table'], 'meta': {'source': 'capture', 'file': orig_name}}
                    try:
                        alert_path = os.path.join(CAPTURES_ALERTS, f'alert_annot_{timestamp}.png')
                        shutil.copy2(os.path.join(CAPTURES_ANNOTATED, annot_name), alert_path)
                        with open(alert_path, 'rb') as f:
                            alert['image_base64'] = base64.b64encode(f.read()).decode('utf-8')
                    except Exception:
                        pass
                    push_alert(alert)

                return jsonify({'ok': True, 'original': f'/api/yolo/captures/originals/{orig_name}', 'annotated': f'/api/yolo/captures/annotated/{annot_name}', 'stats': stats, 'stats_url': f'/api/yolo/captures/stats/{stats_name}', 'mode': 'ultralytics_yolo'})
            except Exception as e:
                print(f"YOLO capture 异常: {e}")

        # ── 方案 B: RKNN CNN 回退 ──
        if _rknn_available:
            try:
                result = rknn_classify_single(img)
                if result is None:
                    return jsonify({'error': 'RKNN CNN 推理失败'}), 500

                cnn_stats = _merge_stats([{result['action']: 1}])
                cnn_stats = ensure_tracked_keys(cnn_stats)
                cnn_stats['display_mode_label'] = 'RKNN CNN 行为分类'

                annot_name = f'capture_annot_{timestamp}.png'
                img.save(os.path.join(CAPTURES_ANNOTATED, annot_name))

                stats_name = f'capture_stats_{timestamp}.json'
                with open(os.path.join(CAPTURES_STATS, stats_name), 'w', encoding='utf-8') as f:
                    json.dump(cnn_stats, f, ensure_ascii=False, indent=2)

                return jsonify({'ok': True, 'original': f'/api/yolo/captures/originals/{orig_name}', 'annotated': f'/api/yolo/captures/annotated/{annot_name}', 'stats': cnn_stats, 'stats_url': f'/api/yolo/captures/stats/{stats_name}', 'cnn_result': result, 'mode': 'rknn_cnn'})
            except Exception as e:
                print(f"RKNN CNN capture 异常: {e}")

        return jsonify({'error': 'YOLO 和 RKNN 模型均不可用'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/yolo/captures/<path:filename>')
def api_yolo_captures(filename):
    return send_from_directory(CAPTURES_DIR, filename)

@app.route('/api/yolo/list', methods=['GET'])
def api_yolo_list():
    images, videos = [], []
    for name in sorted(os.listdir(TRY_ORIGIN)):
        path = os.path.join(TRY_ORIGIN, name)
        if not os.path.isfile(path):
            continue
        media_type, _ = mimetypes.guess_type(path)
        if media_type and media_type.startswith('image'):
            images.append(name)
        elif media_type and media_type.startswith('video'):
            videos.append(name)
    return jsonify({'images': images, 'videos': videos})

@app.route('/api/yolo/origin/<path:filename>')
def api_yolo_origin(filename):
    return send_from_directory(TRY_ORIGIN, filename)

@app.route('/api/yolo/target/<path:filename>')
def api_yolo_target(filename):
    return send_from_directory(TRY_TARGET, filename)

@app.route('/api/yolo/stats', methods=['GET'])
def api_yolo_stats():
    total = 0
    try:
        for root, _, files in os.walk(CAPTURES_STATS):
            for fn in files:
                if fn.endswith('.json'):
                    try:
                        with open(os.path.join(root, fn), 'r', encoding='utf-8') as f:
                            s = json.load(f)
                        total += int(s.get('total_detections', 0))
                    except Exception:
                        pass
    except Exception:
        pass
    return jsonify({'total': total})

# ============================================================================
# 路由：预警系统（项目2）
# ============================================================================
@app.route('/api/alerts', methods=['GET', 'POST'])
def api_alerts():
    if request.method == 'GET':
        with alerts_lock:
            return jsonify({'alerts': alerts_store})
    payload = request.get_json(silent=True) or {}
    alert = {
        'id': payload.get('id') or int(time.time()*1000),
        'timestamp': payload.get('timestamp') or datetime.utcnow().isoformat()+'Z',
        'level': payload.get('level') or 'warning',
        'type': payload.get('type') or 'abnormal',
        'count': int(payload.get('count') or 1),
        'meta': payload.get('meta') or {},
    }
    if payload.get('image_base64'):
        alert['image_base64'] = payload['image_base64']
    push_alert(alert)
    return jsonify({'ok': True, 'alert': alert}), 201

@app.route('/api/alerts/stream')
def api_alerts_stream():
    def gen():
        while True:
            try:
                alert = alerts_queue.get(timeout=30)
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue
            try:
                data = json.dumps(alert, ensure_ascii=False)
            except Exception:
                data = json.dumps({'id': alert.get('id')})
            yield f"data: {data}\n\n"
    return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ============================================================================
# 路由：YOLO 考勤（项目2）
# ============================================================================
@app.route('/api/yolo/attendance/mark', methods=['POST'])
def api_yolo_attendance_mark():
    payload = request.get_json(silent=True) or {}
    record = {
        'id': int(time.time()*1000),
        'timestamp': datetime.utcnow().isoformat()+'Z',
        'original': payload.get('original'),
        'annotated': payload.get('annotated'),
        'stats': payload.get('stats') or {},
        'meta': payload.get('meta') or {},
    }
    with yolo_attendance_lock:
        yolo_attendance_store.insert(0, record)
        del yolo_attendance_store[2000:]
        save_yolo_attendance()
    return jsonify({'ok': True, 'record': record}), 201

@app.route('/api/yolo/attendance/records', methods=['GET'])
def api_yolo_attendance_records():
    with yolo_attendance_lock:
        return jsonify({'records': yolo_attendance_store})

# ============================================================================
# 路由：节能控制 API（项目3）
# ============================================================================
@app.route('/api/energy/status', methods=['GET'])
def api_energy_status():
    """获取节能控制状态：人数、设备开关状态等"""
    if _energy_manager is None:
        return jsonify({'error': '节能控制未初始化'}), 503
    return jsonify(_energy_manager.get_status())

@app.route('/api/energy/override', methods=['POST'])
def api_energy_override():
    """手动覆盖设备状态  {"device": "light", "state": true/false}"""
    if _energy_manager is None:
        return jsonify({'error': '节能控制未初始化'}), 503
    payload = request.get_json(silent=True) or {}
    device = payload.get('device', '')
    state = bool(payload.get('state', False))
    ok = _energy_manager.manual_override(device, state)
    return jsonify({'ok': ok, 'device': device, 'state': state})

@app.route('/api/energy/toggle-simulate', methods=['POST'])
def api_energy_toggle_simulate():
    """切换模拟/真实 GPIO 模式"""
    if _energy_manager is None:
        return jsonify({'error': '节能控制未初始化'}), 503
    new_val = not _energy_manager.cfg['simulate']
    _energy_manager.cfg['simulate'] = new_val
    if _energy_manager._relay_board:
        _energy_manager._relay_board.simulate = new_val
        for r in _energy_manager._relay_board.relays.values():
            r.simulate = new_val
    return jsonify({'ok': True, 'simulate': new_val})

# ============================================================================
# 路由：摄像头（项目1）
# ============================================================================
@app.route('/video_feed')
def video_feed():
    camera_id = request.args.get('camera', DEFAULT_CAMERA_ID, type=int)
    camera = get_camera(camera_id)
    if camera is None:
        return "摄像头不可用 (索引: {})".format(camera_id), 503
    init_energy_manager()  # 按需启动节能控制系统
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_snapshot')
def video_snapshot():
    camera_id = request.args.get('camera', DEFAULT_CAMERA_ID, type=int)
    with _cameras_lock:
        cap = _cameras.get(camera_id)
        if cap is None or not cap.isOpened():
            return "摄像头不可用 (索引: {})".format(camera_id), 503
        success, frame = cap.read()
    if not success:
        return "截图失败", 500
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return "编码失败", 500
    return Response(buffer.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-cache'})

@app.route('/video_stop')
def video_stop():
    camera_id = request.args.get('camera', DEFAULT_CAMERA_ID, type=int)
    stop_energy_manager()  # 释放摄像头前先停掉节能控制系统
    release_camera(camera_id)
    return jsonify({'ok': True})

# ============================================================================
# 启动
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"[*] 服务启动: http://0.0.0.0:{port}")
    print(f"[*] 板子 IP 访问: 浏览器打开 http://<板子IP>:{port}")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
