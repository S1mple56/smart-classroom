#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPIO 继电器控制模块
支持两种模式：
  - 真实模式：通过 Linux sysfs (/sys/class/gpio/) 控制 GPIO
  - 模拟模式：打印日志模拟控制，用于 PC 开发调试
"""

import os
import time
import logging

logger = logging.getLogger("Relay")


class Relay:
    """单个继电器控制器"""

    def __init__(self, pin, name="Relay", active_low=False, simulate=False):
        """
        :param pin:        GPIO 引脚编号 (BCM编号)
        :param name:       继电器名称，用于日志
        :param active_low: True=低电平触发，False=高电平触发
        :param simulate:   模拟模式，不操作真实硬件，仅打印日志
        """
        self.pin = pin
        self.name = name
        self.active_low = active_low
        self.simulate = simulate
        self._state = False  # False=关, True=开
        self._gpio_path = f"/sys/class/gpio/gpio{pin}"

        if not self.simulate:
            self._export()
            time.sleep(0.1)
            self._set_direction("out")
            self.off()  # 默认关闭
        else:
            logger.info(f"[模拟] {self.name} (GPIO{pin}) 已初始化 — 默认关闭")

    def _export(self):
        """导出 GPIO"""
        if not os.path.exists(self._gpio_path):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(self.pin))

    def _set_direction(self, direction):
        """设置 GPIO 方向"""
        path = f"{self._gpio_path}/direction"
        with open(path, "w") as f:
            f.write(direction)

    def _write_value(self, value):
        """写入 GPIO 值"""
        path = f"{self._gpio_path}/value"
        with open(path, "w") as f:
            f.write(str(value))

    def on(self):
        """打开继电器（通电）"""
        if self._state:
            return
        self._state = True

        if self.simulate:
            logger.info(f"[模拟] ⚡ {self.name} → 开启")
        else:
            val = 0 if self.active_low else 1
            self._write_value(val)
            logger.info(f"⚡ {self.name} (GPIO{self.pin}) → 开启")

    def off(self):
        """关闭继电器（断电）"""
        if not self._state:
            return
        self._state = False

        if self.simulate:
            logger.info(f"[模拟] 🔌 {self.name} → 关闭")
        else:
            val = 1 if self.active_low else 0
            self._write_value(val)
            logger.info(f"🔌 {self.name} (GPIO{self.pin}) → 关闭")

    def is_on(self):
        return self._state

    def cleanup(self):
        """释放 GPIO"""
        self.off()
        if not self.simulate and os.path.exists(self._gpio_path):
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.pin))


class RelayBoard:
    """继电器板管理 — 管理多个继电器"""

    def __init__(self, config, simulate=False):
        """
        :param config:   dict, 如 {"light": 17, "fan": 27, "ac": 22}
        :param simulate: 模拟模式
        """
        self.simulate = simulate
        self.relays = {}
        for name, pin in config.items():
            self.relays[name] = Relay(
                pin=pin,
                name=name,
                active_low=False,
                simulate=simulate,
            )

    def turn_on(self, name):
        if name in self.relays:
            self.relays[name].on()

    def turn_off(self, name):
        if name in self.relays:
            self.relays[name].off()

    def set(self, name, state: bool):
        if state:
            self.turn_on(name)
        else:
            self.turn_off(name)

    def all_off(self):
        for relay in self.relays.values():
            relay.off()

    def all_on(self):
        for relay in self.relays.values():
            relay.on()

    def status(self):
        return {name: r.is_on() for name, r in self.relays.items()}

    def cleanup(self):
        for relay in self.relays.values():
            relay.cleanup()

    def __repr__(self):
        status_str = ", ".join(
            f"{name}=ON" if r.is_on() else f"{name}=OFF"
            for name, r in self.relays.items()
        )
        return f"RelayBoard({status_str})"
