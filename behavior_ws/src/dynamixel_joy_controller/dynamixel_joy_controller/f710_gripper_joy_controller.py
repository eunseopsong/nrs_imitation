#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2026 eunseop

"""
Logitech F710 joystick controller for a Dynamixel UMI gripper.

Subscribes:
  sensor_msgs/Joy on joy_topic

Publishes:
  std_msgs/Int32 on command_topic, using motor tick targets.

Default F710 / Xbox-like mapping:
  A  button[0]     -> close, max_tick
  B  button[1]     -> open, min_tick
  D-pad horizontal -> step target by step_tick

The optional trigger axis mode is disabled by default so startup joystick
messages do not move the motor unexpectedly.
"""

import time
from typing import List, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import Int32


def clamp(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def clamp_float(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def map_range(value: float, in_a: float, in_b: float, out_a: int, out_b: int) -> int:
    if abs(in_b - in_a) < 1.0e-9:
        return int(out_a)

    ratio = (float(value) - in_a) / (in_b - in_a)
    ratio = clamp_float(ratio, 0.0, 1.0)
    return int(round(out_a + ratio * (out_b - out_a)))


class F710GripperJoyController(Node):
    def __init__(self):
        super().__init__("f710_gripper_joy_controller")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("command_topic", "/gripper/command")
        self.declare_parameter("present_position_topic", "/gripper/present_position")
        self.declare_parameter("use_present_position", True)

        self.declare_parameter("min_tick", 590)
        self.declare_parameter("max_tick", 2500)
        self.declare_parameter("home_tick", -1)
        self.declare_parameter("initial_tick", -1)
        self.declare_parameter("step_tick", 50)

        self.declare_parameter("button_open", 1)
        self.declare_parameter("button_close", 0)
        self.declare_parameter("button_home", -1)
        self.declare_parameter("button_enable", -1)
        self.declare_parameter("button_debounce_sec", 0.20)

        self.declare_parameter("dpad_axis", 6)
        self.declare_parameter("dpad_threshold", 0.50)
        self.declare_parameter("dpad_invert", False)
        self.declare_parameter("dpad_debounce_sec", 0.15)

        self.declare_parameter("axis_control.enabled", False)
        self.declare_parameter("axis_control.axis_index", 5)
        self.declare_parameter("axis_control.released_value", 1.0)
        self.declare_parameter("axis_control.pressed_value", -1.0)
        self.declare_parameter("axis_control.activation_threshold", 0.05)
        self.declare_parameter("axis_control.min_change_tick", 5)
        self.declare_parameter("axis_control.publish_rate_hz", 30.0)
        self.declare_parameter("axis_control.enable_button", -1)

        self.declare_parameter("log_commands", True)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.present_position_topic = str(
            self.get_parameter("present_position_topic").value
        )
        self.use_present_position = bool(self.get_parameter("use_present_position").value)

        self.min_tick = int(self.get_parameter("min_tick").value)
        self.max_tick = int(self.get_parameter("max_tick").value)
        if self.min_tick > self.max_tick:
            self.min_tick, self.max_tick = self.max_tick, self.min_tick

        self.home_tick = int(self.get_parameter("home_tick").value)
        if self.home_tick < 0:
            self.home_tick = (self.min_tick + self.max_tick) // 2
        self.home_tick = clamp(self.home_tick, self.min_tick, self.max_tick)

        self.initial_tick = int(self.get_parameter("initial_tick").value)
        if self.initial_tick >= 0:
            self.initial_tick = clamp(self.initial_tick, self.min_tick, self.max_tick)

        self.step_tick = max(1, int(self.get_parameter("step_tick").value))

        self.button_open = int(self.get_parameter("button_open").value)
        self.button_close = int(self.get_parameter("button_close").value)
        self.button_home = int(self.get_parameter("button_home").value)
        self.button_enable = int(self.get_parameter("button_enable").value)
        self.button_debounce_sec = float(
            self.get_parameter("button_debounce_sec").value
        )

        self.dpad_axis = int(self.get_parameter("dpad_axis").value)
        self.dpad_threshold = float(self.get_parameter("dpad_threshold").value)
        self.dpad_invert = bool(self.get_parameter("dpad_invert").value)
        self.dpad_debounce_sec = float(self.get_parameter("dpad_debounce_sec").value)

        self.axis_enabled = bool(self.get_parameter("axis_control.enabled").value)
        self.axis_index = int(self.get_parameter("axis_control.axis_index").value)
        self.axis_released_value = float(
            self.get_parameter("axis_control.released_value").value
        )
        self.axis_pressed_value = float(
            self.get_parameter("axis_control.pressed_value").value
        )
        self.axis_activation_threshold = float(
            self.get_parameter("axis_control.activation_threshold").value
        )
        self.axis_min_change_tick = max(
            1,
            int(self.get_parameter("axis_control.min_change_tick").value),
        )
        self.axis_publish_rate_hz = float(
            self.get_parameter("axis_control.publish_rate_hz").value
        )
        self.axis_enable_button = int(
            self.get_parameter("axis_control.enable_button").value
        )

        self.log_commands = bool(self.get_parameter("log_commands").value)

        self.prev_buttons: Optional[List[int]] = None
        self.prev_dpad_sign = 0
        self.last_button_time = 0.0
        self.last_dpad_time = 0.0
        self.last_axis_publish_time = 0.0
        self.last_axis_tick: Optional[int] = None
        self.last_published_tick: Optional[int] = None
        self.latest_position: Optional[int] = None
        self.target_tick: Optional[int] = (
            self.initial_tick if self.initial_tick >= 0 else None
        )

        self.cmd_pub = self.create_publisher(Int32, self.command_topic, 10)
        self.joy_sub = self.create_subscription(
            Joy,
            self.joy_topic,
            self.joy_callback,
            20,
        )
        self.position_sub = None
        if self.use_present_position:
            self.position_sub = self.create_subscription(
                Int32,
                self.present_position_topic,
                self.present_position_callback,
                20,
            )

        self.get_logger().info(
            "\n".join(
                [
                    "F710 gripper joy controller ready",
                    f"  joy_topic: {self.joy_topic}",
                    f"  command_topic: {self.command_topic}",
                    f"  range: [{self.min_tick}, {self.max_tick}]",
                    f"  B button[{self.button_open}] -> open",
                    f"  A button[{self.button_close}] -> close",
                    "  home: disabled"
                    if self.button_home < 0
                    else f"  home button[{self.button_home}] -> home={self.home_tick}",
                    f"  D-pad axis[{self.dpad_axis}] step={self.step_tick}",
                    f"  axis_control.enabled: {self.axis_enabled}",
                ]
            )
        )

    def present_position_callback(self, msg: Int32):
        self.latest_position = clamp(int(msg.data), self.min_tick, self.max_tick)

    @staticmethod
    def _button_down(buttons: List[int], index: int) -> bool:
        if index < 0 or index >= len(buttons):
            return False
        return int(buttons[index]) == 1

    @classmethod
    def _button_pressed(
        cls,
        current_buttons: List[int],
        previous_buttons: List[int],
        index: int,
    ) -> bool:
        if index < 0:
            return False
        if index >= len(current_buttons) or index >= len(previous_buttons):
            return False
        return cls._button_down(current_buttons, index) and not cls._button_down(
            previous_buttons,
            index,
        )

    def _commands_enabled(self, buttons: List[int]) -> bool:
        return self.button_enable < 0 or self._button_down(buttons, self.button_enable)

    def _axis_mode_enabled(self, buttons: List[int]) -> bool:
        if not self.axis_enabled:
            return False
        if self.axis_enable_button < 0:
            return True
        return self._button_down(buttons, self.axis_enable_button)

    def _get_axis(self, axes: List[float], index: int) -> Optional[float]:
        if index < 0 or index >= len(axes):
            return None
        return float(axes[index])

    def _dpad_sign(self, axes: List[float]) -> int:
        value = self._get_axis(axes, self.dpad_axis)
        if value is None:
            return 0
        if self.dpad_invert:
            value = -value
        if value > self.dpad_threshold:
            return 1
        if value < -self.dpad_threshold:
            return -1
        return 0

    def _base_tick_for_step(self) -> int:
        if self.target_tick is not None:
            return self.target_tick
        if self.latest_position is not None:
            return self.latest_position
        if self.initial_tick >= 0:
            return self.initial_tick
        return self.home_tick

    def publish_tick(self, tick: int, reason: str):
        tick = clamp(int(tick), self.min_tick, self.max_tick)
        self.target_tick = tick
        self.last_published_tick = tick

        msg = Int32()
        msg.data = tick
        self.cmd_pub.publish(msg)

        if self.log_commands:
            self.get_logger().info(f"[F710 GRIPPER] {reason} -> {tick}")

    def _handle_button_commands(
        self,
        current_buttons: List[int],
        previous_buttons: List[int],
        now: float,
    ) -> bool:
        if now - self.last_button_time < self.button_debounce_sec:
            return False

        if self._button_pressed(current_buttons, previous_buttons, self.button_open):
            self.publish_tick(self.min_tick, "open")
            self.last_button_time = now
            return True

        if self._button_pressed(current_buttons, previous_buttons, self.button_close):
            self.publish_tick(self.max_tick, "close")
            self.last_button_time = now
            return True

        if self._button_pressed(current_buttons, previous_buttons, self.button_home):
            self.publish_tick(self.home_tick, "home")
            self.last_button_time = now
            return True

        return False

    def _handle_dpad_command(self, axes: List[float], now: float) -> bool:
        sign = self._dpad_sign(axes)
        should_step = sign != 0 and (
            self.prev_dpad_sign == 0
            or now - self.last_dpad_time >= self.dpad_debounce_sec
        )

        self.prev_dpad_sign = sign
        if not should_step:
            return False

        base_tick = self._base_tick_for_step()
        tick = base_tick + sign * self.step_tick
        self.publish_tick(tick, "step_close" if sign > 0 else "step_open")
        self.last_dpad_time = now
        return True

    def _handle_axis_command(self, axes: List[float], now: float) -> bool:
        value = self._get_axis(axes, self.axis_index)
        if value is None:
            return False

        raw_ratio = (value - self.axis_released_value) / (
            self.axis_pressed_value - self.axis_released_value
        )
        ratio = clamp_float(raw_ratio, 0.0, 1.0)
        if ratio < self.axis_activation_threshold:
            return False

        tick = map_range(
            value,
            self.axis_released_value,
            self.axis_pressed_value,
            self.min_tick,
            self.max_tick,
        )

        if (
            self.last_axis_tick is not None
            and abs(tick - self.last_axis_tick) < self.axis_min_change_tick
        ):
            return False

        min_period = 0.0
        if self.axis_publish_rate_hz > 0.0:
            min_period = 1.0 / self.axis_publish_rate_hz
        if now - self.last_axis_publish_time < min_period:
            return False

        self.last_axis_tick = tick
        self.last_axis_publish_time = now
        self.publish_tick(tick, f"axis[{self.axis_index}]={value:.3f}")
        return True

    def joy_callback(self, msg: Joy):
        current_buttons = list(msg.buttons)
        axes = list(msg.axes)

        if self.prev_buttons is None:
            self.prev_buttons = current_buttons
            self.prev_dpad_sign = self._dpad_sign(axes)
            return

        now = time.time()

        if self._commands_enabled(current_buttons):
            handled = self._handle_button_commands(
                current_buttons,
                self.prev_buttons,
                now,
            )
            if not handled:
                handled = self._handle_dpad_command(axes, now)
            if not handled and self._axis_mode_enabled(current_buttons):
                self._handle_axis_command(axes, now)

        self.prev_buttons = current_buttons


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = F710GripperJoyController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except (Exception, KeyboardInterrupt):
                pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
