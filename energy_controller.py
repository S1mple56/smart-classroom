#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能教室节能控制模块
====================
集成到 qiansai/board_app.py 中，作为后台线程运行。
使用 yolo11n.pt 检测教室人数，通过 GPIO 继电器控制灯光/风扇/空调。

依赖：
  - ultralytics (YOLO)
  - relay.py (GPIO 控制)
"""

import os
import sys
import time
import logging
import threading

import numpy as np

logger = logging.getLogger("EnergyCtrl")

# ———— 路径 ————
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

# ———— 配置默认值（可被 board_app.py 覆盖） ————
DEFAULT_CONFIG = {
    "simulate": True,              # 模拟模式（不操作真实 GPIO）
    "camera_id": 0,                # 摄像头索引
    "detect_interval": 2.0,        # 检测间隔（秒）
    "model_path": os.path.join(BASE_DIR, "yolo11n.pt"),  # YOLO 模型路径
    "confidence": 0.3,             # 置信度阈值
    "gpio_pins": {                 # GPIO 引脚
        "light": 17,
        "fan":   27,
        "ac":    22,
    },
    "rules": [                     # (min, max, light, fan, ac)
        (0,   1,    False, False, False),
        (1,   6,    True,  True,  True),
        (6,   16,   True,  True,  True),
        (16,  999,  True,  True,  True),
    ],
    "debounce_frames": 3,          # 防抖帧数
}


class EnergyManager:
    """
    节能管理器 — 后台线程运行。

    用法:
        mgr = EnergyManager(get_camera_fn, config={...})
        mgr.start()
        # ...
        status = mgr.get_status()   # {"person_count": 12, "light": True, ...}
        mgr.stop()
    """

    def __init__(self, get_camera_fn, config=None):
        """
        :param get_camera_fn: 函数，返回 cv2.VideoCapture 实例
        :param config:        配置字典，覆盖 DEFAULT_CONFIG
        """
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg

        self._get_camera = get_camera_fn
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # ———— 状态 ————
        self.person_count = 0
        self.device_state = {"light": False, "fan": False, "ac": False}
        self.fps = 0.0
        self.running = False
        self.last_detect_time = 0

        # 防抖
        self._count_history = []

        # ———— YOLO 模型 ————
        self._yolo = None
        self._init_yolo()

        # ———— 继电器 ————
        self._relay_board = None
        self._init_relay()

        logger.info("EnergyManager 初始化完成 "
                     f"(模型={os.path.basename(cfg['model_path'])}, "
                     f"模式={'模拟' if cfg['simulate'] else '真实GPIO'})")

    # ==================== 初始化 ====================

    def _init_yolo(self):
        """加载 yolo11n.pt，只关注 person 类 (class_id=0)"""
        model_path = self.cfg["model_path"]
        if not os.path.isfile(model_path):
            logger.warning(f"YOLO 模型未找到: {model_path}，节能控制不可用")
            return
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(model_path)
            logger.info(f"YOLO 模型已加载: {os.path.basename(model_path)}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")

    def _init_relay(self):
        """初始化继电器板"""
        try:
            from relay import RelayBoard
            self._relay_board = RelayBoard(
                self.cfg["gpio_pins"],
                simulate=self.cfg["simulate"]
            )
        except Exception as e:
            logger.error(f"继电器初始化失败: {e}")

    # ==================== 检测 ====================

    def _count_persons(self, frame_bgr):
        """
        对一帧运行 YOLO 检测，统计 person 数量。
        返回: person_count (int)
        """
        if self._yolo is None:
            return 0

        try:
            results = self._yolo.predict(
                frame_bgr,
                imgsz=320,             # 小尺寸加速
                conf=self.cfg["confidence"],
                classes=[0],           # 只检测 person
                verbose=False,
                device="cpu",          # RK3588 用 CPU；有 NPU 可改为合适的
            )
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    return len(boxes)
        except Exception as e:
            logger.error(f"检测出错: {e}")
        return 0

    # ==================== 控制决策 ====================

    def _apply_control(self, count):
        """根据人数 + 防抖，控制继电器"""
        self._count_history.append(count)
        if len(self._count_history) > self.cfg["debounce_frames"]:
            self._count_history.pop(0)

        # 不稳定不操作
        if len(set(self._count_history)) != 1:
            return

        rule = None
        for min_n, max_n, light, fan, ac in self.cfg["rules"]:
            if min_n <= count < max_n:
                rule = (light, fan, ac)
                break
        if rule is None:
            return

        light_on, fan_on, ac_on = rule
        target = {"light": light_on, "fan": fan_on, "ac": ac_on}

        if self._relay_board:
            for name in ["light", "fan", "ac"]:
                if target[name] != self.device_state[name]:
                    self._relay_board.set(name, target[name])
                    self.device_state[name] = target[name]

    # ==================== 手动控制 ====================

    def manual_override(self, device, state):
        """手动覆盖某个设备状态"""
        if device not in self.device_state:
            return False
        if self._relay_board:
            with self._lock:
                self._relay_board.set(device, state)
                self.device_state[device] = state
            logger.info(f"手动控制: {device} → {'ON' if state else 'OFF'}")
            return True
        return False

    # ==================== 状态 ====================

    def get_status(self):
        """线程安全获取当前状态"""
        with self._lock:
            return {
                "person_count": self.person_count,
                "light": self.device_state["light"],
                "fan": self.device_state["fan"],
                "ac": self.device_state["ac"],
                "fps": round(self.fps, 1),
                "simulate": self.cfg["simulate"],
                "model_loaded": self._yolo is not None,
            }

    # ==================== 后台线程 ====================

    def _run_loop(self):
        """后台主循环"""
        logger.info("EnergyManager 后台线程启动")
        self.running = True
        fps_timer = time.time()
        fps_frames = 0

        while not self._stop_event.is_set():
            try:
                cap = self._get_camera()
                if cap is None:
                    time.sleep(1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                fps_frames += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    self.fps = fps_frames / (now - fps_timer)
                    fps_frames = 0
                    fps_timer = now

                # 定时检测
                if now - self.last_detect_time >= self.cfg["detect_interval"]:
                    self.last_detect_time = now
                    count = self._count_persons(frame)

                    with self._lock:
                        self.person_count = count

                    self._apply_control(count)

                    logger.debug(
                        f"人数={count}  "
                        f"L={'ON' if self.device_state['light'] else 'OFF'} "
                        f"F={'ON' if self.device_state['fan'] else 'OFF'} "
                        f"A={'ON' if self.device_state['ac'] else 'OFF'}"
                    )

                time.sleep(0.2)

            except Exception as e:
                logger.error(f"EnergyManager 循环出错: {e}")
                time.sleep(1)

        self.running = False
        logger.info("EnergyManager 后台线程已停止")

    def start(self):
        """启动后台线程"""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止后台线程并清理"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._relay_board:
            self._relay_board.all_off()
            self._relay_board.cleanup()
        logger.info("EnergyManager 已完全停止")


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import cv2
    logging.basicConfig(level=logging.INFO)

    print("EnergyManager 模块测试")
    print("请在此文件被 board_app.py 导入后通过 API 使用")
    print("或运行: python board_app.py")
