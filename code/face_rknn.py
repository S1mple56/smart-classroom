#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3588 NPU 人脸识别模块
======================
替代原来的 Haar 级联 + RGB 直方图方案。

管线：
  人脸检测 (OpenCV DNN / CPU)  →  特征提取 (RKNN / NPU)  →  欧氏距离匹配

依赖：
  - rknn-toolkit-lite2 (板子上)
  - opencv-python
  - numpy, PIL

模型文件：
  - rknn_convert/face_detect.onnx  (可选，OpenCV DNN 人脸检测)
  - rknn_convert/face_feature.rknn (人脸特征提取，128D/512D embedding)
"""

import os
import json
import logging
import numpy as np
from PIL import Image

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RKNN_CONVERT_DIR = os.path.join(BASE_DIR, 'rknn_convert')
FACE_MODEL_PATH = os.path.join(RKNN_CONVERT_DIR, 'face_feature.rknn')

# 人脸特征输入尺寸（需与 ONNX 模型一致，MobileFaceNet 通常 112×112）
FACE_INPUT_SIZE = (112, 112)

# 匹配阈值
MATCH_THRESHOLD = 0.6  # 欧氏距离 < 0.6 认为是同一人
UNKNOWN_THRESHOLD = 1.0  # 距离 > 1.0 标记为陌生人

# ============================================================================
# OpenCV 人脸检测器（CPU 上跑，板子的 A76 核足够快）
# ============================================================================
_face_detector_net = None  # OpenCV DNN detector (高性能)
_face_detector_cascade = None  # Haar cascade (回退)

try:
    import cv2
    _cv2_available = True
except ImportError:
    cv2 = None
    _cv2_available = False


def _init_dnn_detector():
    """初始化 OpenCV DNN 人脸检测器（YuNet / OpenCV 官方模型）"""
    global _face_detector_net
    if _face_detector_net is not None or not _cv2_available:
        return _face_detector_net

    # 优先使用 YuNet (OpenCV 4.5.5+ 内置)
    # YuNet 模型可以从 OpenCV 仓库获取: opencv_zoo/models/face_detection_yunet/
    dnn_model = os.path.join(RKNN_CONVERT_DIR, 'face_detection_yunet_2022mar.onnx')
    if os.path.isfile(dnn_model):
        _face_detector_net = cv2.FaceDetectorYN.create(
            dnn_model, "", (320, 320), 0.6, 0.3, 5000
        )
        logging.info("人脸检测器(YuNet DNN) 已初始化")
    else:
        logging.info("YuNet 模型未找到(%s)，回退到 Haar Cascade", dnn_model)
    return _face_detector_net


def detect_faces(img_rgb):
    """
    检测图像中的所有人脸，返回 [(x1, y1, x2, y2), ...] 坐标列表。
    使用 DNN(YuNet) → Haar Cascade 回退策略。
    """
    if not _cv2_available:
        return []

    h, w = img_rgb.shape[:2]

    # --- 策略 1: YuNet DNN ---
    detector = _init_dnn_detector()
    if detector is not None:
        detector.setInputSize((w, h))
        try:
            _, faces = detector.detect(img_rgb)
            if faces is not None and len(faces) > 0:
                boxes = []
                for face in faces:
                    # YuNet 输出: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rm, y_rm, x_lm, y_lm, score]
                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    x1 = max(0, fx)
                    y1 = max(0, fy)
                    x2 = min(w, fx + fw)
                    y2 = min(h, fy + fh)
                    boxes.append((x1, y1, x2, y2))
                return boxes
        except Exception as e:
            logging.debug("YuNet 检测失败: %s", e)

    # --- 策略 2: Haar Cascade 回退 ---
    global _face_detector_cascade
    if _face_detector_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_detector_cascade = cv2.CascadeClassifier(cascade_path)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


def crop_face(img_rgb, box, expand_ratio=0.2):
    """
    裁剪并扩充人脸区域。
    :param img_rgb: numpy array (H, W, 3) RGB
    :param box: (x1, y1, x2, y2)
    :param expand_ratio: 扩展比例
    :return: 裁剪后的人脸图像 (PIL Image) 或 None
    """
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    face = img_rgb[y1:y2, x1:x2]
    return Image.fromarray(face)


# ============================================================================
# RKNN 人脸特征提取（NPU）
# ============================================================================
_face_rknn = None
_rknn_available = False

try:
    from rknnlite.api import RKNNLite
    _rknn_available = True
except ImportError:
    RKNNLite = None


def _load_face_rknn():
    """加载人脸特征提取 RKNN 模型（懒加载，NPU）"""
    global _face_rknn
    if _face_rknn is not None:
        return _face_rknn
    if not _rknn_available:
        logging.warning("rknnlite 未安装，人脸特征提取不可用")
        return None
    if not os.path.isfile(FACE_MODEL_PATH):
        logging.warning("人脸特征 RKNN 模型未找到: %s", FACE_MODEL_PATH)
        return None

    _face_rknn = RKNNLite()
    ret = _face_rknn.load_rknn(FACE_MODEL_PATH)
    if ret != 0:
        logging.error("加载人脸特征模型失败: %s", FACE_MODEL_PATH)
        _face_rknn = None
        return None
    _face_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    logging.info("RKNN 人脸特征模型已加载: %s", FACE_MODEL_PATH)
    return _face_rknn


def preprocess_face(pil_face):
    """
    预处理人脸图像用于 RKNN 推理。
    :param pil_face: PIL Image (RGB)
    :return: numpy array (1, H, W, 3) NHWC, float32, [0, 1]
    """
    img = pil_face.convert('RGB').resize(FACE_INPUT_SIZE, Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)  # (1, H, W, 3) NHWC
    return img_np


def extract_face_feature(pil_face):
    """
    提取人脸 128D/512D 特征向量。
    :param pil_face: PIL Image (RGB, 人脸区域)
    :return: numpy array (D,) 或 None
    """
    rknn = _load_face_rknn()
    if rknn is None:
        return None

    try:
        inp = preprocess_face(pil_face)
        outputs = rknn.inference(inputs=[inp])
        feature = outputs[0].flatten()
        # L2 归一化（如果模型输出未归一化）
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature
    except Exception as e:
        logging.warning("人脸特征提取失败: %s", e)
        return None


def is_face_model_ready():
    """检查人脸特征模型是否可用"""
    return _rknn_available and os.path.isfile(FACE_MODEL_PATH)


# ============================================================================
# 人脸特征数据库
# ============================================================================

def build_face_features_db(upload_dir, students_info_path, features_cache_path):
    """
    扫描学生照片，为每人提取人脸特征并缓存。
    :param upload_dir: 学生照片根目录 data/upload/
    :param students_info_path: data/students_info.json
    :param features_cache_path: data/face_features_rknn.json
    :return: dict { person_id: {"features": [...], "name": ...} }
    """
    students_info = {}
    if os.path.exists(students_info_path):
        with open(students_info_path, 'r', encoding='utf-8') as f:
            students_info = json.load(f).get('students', {})

    features_db = {}

    for person_id, info in students_info.items():
        person_dir = os.path.join(upload_dir, person_id)
        if not os.path.isdir(person_dir):
            continue

        all_features = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img_rgb = np.array(img)
                faces = detect_faces(img_rgb)
                if not faces:
                    continue
                # 取最大人脸
                largest = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                face_pil = crop_face(img_rgb, largest)
                if face_pil is None:
                    continue
                feat = extract_face_feature(face_pil)
                if feat is not None:
                    all_features.append(feat)
            except Exception as e:
                logging.warning("处理 %s 失败: %s", img_path, e)

        if all_features:
            mean_features = np.mean(all_features, axis=0)
            norm = np.linalg.norm(mean_features)
            if norm > 0:
                mean_features = mean_features / norm
            mean_features = mean_features.tolist()
        else:
            # 默认维度：尝试从已提取过特征的其他人推断，否则使用 128
            feature_dim = 128
            for _, data in features_db.items():
                feats = data.get('features', [])
                if feats and not (isinstance(feats, list) and all(f == 0.0 for f in feats)):
                    feature_dim = len(feats)
                    break
            mean_features = [0.0] * feature_dim

        features_db[person_id] = {
            "features": mean_features,
            "name": info.get('name', '未知'),
            "student_id": info.get('student_id', '未知'),
            "class": info.get('class', '未知'),
        }

    os.makedirs(os.path.dirname(features_cache_path), exist_ok=True)
    with open(features_cache_path, 'w', encoding='utf-8') as f:
        json.dump(features_db, f, ensure_ascii=False, indent=2)

    logging.info("人脸特征数据库已构建(RKNN)，共 %d 人", len(features_db))
    return features_db


def load_face_features_db(cache_path):
    """加载已缓存的人脸特征数据库"""
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 人脸匹配
# ============================================================================

def match_face(image_path, features_cache_path, upload_dir, students_info_path):
    """
    用 RKNN 进行人脸匹配。
    :return: dict { name, student_id, class, match_confidence, is_stranger }
    """
    # 加载特征数据库
    features_db = load_face_features_db(features_cache_path)
    if not features_db:
        if is_face_model_ready():
            features_db = build_face_features_db(upload_dir, students_info_path, features_cache_path)
        if not features_db:
            return {
                'name': '未知', 'student_id': '未知', 'class': '未知',
                'match_confidence': '0%', 'is_stranger': False
            }

    # 读取并检测人脸
    try:
        img = Image.open(image_path).convert('RGB')
        img_rgb = np.array(img)
        faces = detect_faces(img_rgb)
        if not faces:
            return {
                'name': '未知', 'student_id': '未知', 'class': '未知',
                'match_confidence': '0%', 'is_stranger': False
            }
        largest = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        face_pil = crop_face(img_rgb, largest)
        if face_pil is None:
            return {
                'name': '未知', 'student_id': '未知', 'class': '未知',
                'match_confidence': '0%', 'is_stranger': False
            }
        target_feat = extract_face_feature(face_pil)
        if target_feat is None:
            return {
                'name': '未知', 'student_id': '未知', 'class': '未知',
                'match_confidence': '0%', 'is_stranger': False
            }
    except Exception as e:
        logging.warning("识别图片处理失败: %s", e)
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': '0%', 'is_stranger': False
        }

    # 欧氏距离匹配
    best_distance = float('inf')
    best_match = None

    for person_id, data in features_db.items():
        stored = np.array(data['features'])
        if np.all(stored == 0.0):
            continue
        dist = np.linalg.norm(target_feat - stored)
        if dist < best_distance:
            best_distance = dist
            best_match = data

    if best_match is None:
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': '0%', 'is_stranger': False
        }

    # 距离转置信度
    confidence = max(0.0, min(100.0, (1.0 - best_distance / 1.2) * 100.0))

    if best_distance < MATCH_THRESHOLD:
        return {
            'name': best_match['name'],
            'student_id': best_match['student_id'],
            'class': best_match['class'],
            'match_confidence': f'{confidence:.1f}%',
            'is_stranger': False,
        }
    elif best_distance < UNKNOWN_THRESHOLD:
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': f'{confidence:.1f}%',
            'is_stranger': True,
        }
    else:
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': '0%', 'is_stranger': False
        }


def rebuild_features(upload_dir, students_info_path, features_cache_path):
    """重建特征数据库"""
    logging.info("重建 RKNN 人脸特征数据库...")
    return build_face_features_db(upload_dir, students_info_path, features_cache_path)
