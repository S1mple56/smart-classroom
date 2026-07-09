#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Dlib ResNet CNN 的人脸识别模块
替换原来的 Haar Cascade + 直方图比对方案

模型引用自同目录下的 Dlib_face_recognition_from_camera-master/data/data_dlib/
"""

import os
import json
import logging
import numpy as np
import dlib
import cv2
from PIL import Image

# ============================================================================
# 模型路径配置
# qiansai/ 和 Dlib_face_recognition_from_camera-master/ 在同级目录下
# ============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DLIB_MODEL_DIR = os.path.join(BASE_DIR, '..',
                               'Dlib_face_recognition_from_camera-master',
                               'data', 'data_dlib')

SHAPE_PREDICTOR_PATH = os.path.join(DLIB_MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
FACE_RECO_MODEL_PATH = os.path.join(DLIB_MODEL_DIR, 'dlib_face_recognition_resnet_model_v1.dat')

# 匹配阈值：欧氏距离 < 0.4 认为是同一人
MATCH_THRESHOLD = 0.4

# ============================================================================
# Dlib 模型（懒加载 + 全局单例，避免重复加载）
# ============================================================================
_detector = None
_predictor = None
_face_reco_model = None


def _load_models():
    """懒加载 Dlib 三个模型（处理中文路径兼容性）"""
    global _detector, _predictor, _face_reco_model
    if _detector is None:
        import tempfile
        import shutil

        # Dlib C++ 后端不支持中文路径，将模型文件复制到临时目录
        _temp_dir = os.path.join(tempfile.gettempdir(), 'dlib_models')
        os.makedirs(_temp_dir, exist_ok=True)

        _tmp_shape = os.path.join(_temp_dir, 'shape_predictor_68_face_landmarks.dat')
        _tmp_reco = os.path.join(_temp_dir, 'dlib_face_recognition_resnet_model_v1.dat')

        if not os.path.isfile(_tmp_shape):
            logging.info("复制模型文件到临时目录: %s", _temp_dir)
            shutil.copy2(SHAPE_PREDICTOR_PATH, _tmp_shape)
            shutil.copy2(FACE_RECO_MODEL_PATH, _tmp_reco)

        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(_tmp_shape)
        _face_reco_model = dlib.face_recognition_model_v1(_tmp_reco)
        logging.info("Dlib 人脸识别模型加载成功")


# ============================================================================
# 工具函数
# ============================================================================

def pil_to_cv2_rgb(pil_image):
    """PIL Image (RGB) -> OpenCV numpy array (RGB)"""
    return np.array(pil_image.convert('RGB'))


def detect_largest_face(img_rgb):
    """
    用 Dlib HOG 检测器检测人脸，返回面积最大的人脸区域和坐标。
    :param img_rgb: numpy array (RGB)
    :return: (face_dlib_rect, (x1, y1, x2, y2)) 或 (None, None)
    """
    _load_models()
    faces = _detector(img_rgb, 0)
    if len(faces) == 0:
        return None, None

    best = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    x1, y1 = best.left(), best.top()
    x2, y2 = best.right(), best.bottom()
    return best, (x1, y1, x2, y2)


def extract_face_descriptor(img_rgb, face_rect=None):
    """
    从图像中提取人脸 128D 特征向量。
    :param img_rgb: numpy array (RGB)
    :param face_rect: dlib rectangle，不传则自动检测
    :return: numpy array (128,) 或 None
    """
    _load_models()
    if face_rect is None:
        face_rect, _ = detect_largest_face(img_rgb)
        if face_rect is None:
            return None

    try:
        shape = _predictor(img_rgb, face_rect)
        descriptor = _face_reco_model.compute_face_descriptor(img_rgb, shape)
        return np.array(descriptor)
    except Exception as e:
        logging.warning("特征提取失败: %s", e)
        return None


def euclidean_distance(f1, f2):
    """两个 128D 特征向量的欧氏距离"""
    return np.sqrt(np.sum(np.square(np.array(f1) - np.array(f2))))


# ============================================================================
# 人脸特征数据库
# ============================================================================

def build_features_database(upload_dir, students_info_path, features_cache_path):
    """
    扫描学生照片文件夹，计算并缓存所有人的 128D 平均特征。
    结果存入 features_cache_path JSON 文件。
    """
    _load_models()

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
                img = Image.open(img_path)
                img_rgb = pil_to_cv2_rgb(img)
                desc = extract_face_descriptor(img_rgb)
                if desc is not None:
                    all_features.append(desc)
            except Exception as e:
                logging.warning("处理 %s 失败: %s", img_path, e)

        if all_features:
            mean_features = np.mean(all_features, axis=0).tolist()
        else:
            mean_features = [0.0] * 128

        features_db[person_id] = {
            "features": mean_features,
            "name": info.get('name', '未知'),
            "student_id": info.get('student_id', '未知'),
            "class": info.get('class', '未知'),
        }

    os.makedirs(os.path.dirname(features_cache_path), exist_ok=True)
    with open(features_cache_path, 'w', encoding='utf-8') as f:
        json.dump(features_db, f, ensure_ascii=False, indent=2)

    logging.info("人脸特征数据库已构建，共 %d 人", len(features_db))
    return features_db


def load_features_database(features_cache_path):
    """加载已缓存的人脸特征数据库"""
    if not os.path.exists(features_cache_path):
        return {}
    with open(features_cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 人脸匹配（核心函数 —— 替代原来的 match_student）
# ============================================================================

def match_face_dlib(image_path, features_cache_path, upload_dir, students_info_path):
    """
    用 Dlib 进行人脸匹配。

    :return: dict { name, student_id, class, match_confidence, is_stranger }
    """
    _load_models()

    features_db = load_features_database(features_cache_path)
    if not features_db:
        logging.info("特征数据库为空，开始构建...")
        features_db = build_features_database(upload_dir, students_info_path, features_cache_path)

    if not features_db:
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': '0%', 'is_stranger': False
        }

    # 提取待识别图像的特征
    try:
        img = Image.open(image_path).convert('RGB')
        img_rgb = pil_to_cv2_rgb(img)
        face_rect, _ = detect_largest_face(img_rgb)
        if face_rect is None:
            return {
                'name': '未知', 'student_id': '未知', 'class': '未知',
                'match_confidence': '0%', 'is_stranger': False
            }
        target_desc = extract_face_descriptor(img_rgb, face_rect)
        if target_desc is None:
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

    # 与数据库中每个人比对
    best_distance = float('inf')
    best_match = None

    for person_id, data in features_db.items():
        stored_features = np.array(data['features'])
        if np.all(stored_features == 0.0):
            continue
        dist = euclidean_distance(target_desc, stored_features)
        logging.debug("  与 %s (%s) 欧氏距离: %.4f", data['name'], person_id, dist)
        if dist < best_distance:
            best_distance = dist
            best_match = data

    if best_match is None:
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': '0%', 'is_stranger': False
        }

    confidence = max(0.0, min(100.0, (1.0 - best_distance / 1.0) * 100.0))

    if best_distance < MATCH_THRESHOLD:
        return {
            'name': best_match['name'],
            'student_id': best_match['student_id'],
            'class': best_match['class'],
            'match_confidence': f'{confidence:.1f}%',
            'is_stranger': False,
        }
    else:
        return {
            'name': '未知', 'student_id': '未知', 'class': '未知',
            'match_confidence': f'{confidence:.1f}%',
            'is_stranger': True,
        }


def rebuild_features_on_register(upload_dir, students_info_path, features_cache_path):
    """学生注册/删除后重建特征数据库"""
    logging.info("重建人脸特征数据库...")
    return build_features_database(upload_dir, students_info_path, features_cache_path)
