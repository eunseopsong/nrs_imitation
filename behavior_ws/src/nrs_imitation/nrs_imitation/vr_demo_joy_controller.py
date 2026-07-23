#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vr_demo_joy_controller.py

Joystick command node for VR demo HDF5 recorder.

Requires:
  ros2 run joy joy_node --ros-args -r /joy:=/joy_il
or use the launch file that starts joy_node with the same remap.

This node subscribes:
  /joy_il                      sensor_msgs/Joy

This node publishes:
  /vr_demo_recorder/command    std_msgs/String
  /gripper/command             std_msgs/Int32

Default Logitech F710 / Xbox-like mapping:
  RB       buttons[5]  -> start_recording
  LB       buttons[4]  -> end_recording
  Start    buttons[7]  -> gripper close
  Back     buttons[6]  -> gripper open
  RT       axes[5]     -> gripper close by one step
  LT       axes[2]     -> gripper open by one step
"""

import time
from typing import List, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import Int32, String

from nrs_imitation.pretty_print import block


class VRDemoJoyController(Node):
    def __init__(self):
        super().__init__("vr_demo_joy_controller")

        self.declare_parameter("joy_topic", "/joy_il")
        self.declare_parameter("command_topic", "/vr_demo_recorder/command")
        self.declare_parameter("gripper_command_topic", "/gripper/command")
        self.declare_parameter(
            "gripper_present_position_topic",
            "/gripper/present_position",
        )

        self.declare_parameter("button_start", 5)
        self.declare_parameter("button_end", 4)
        self.declare_parameter("button_gripper_close", 7)
        self.declare_parameter("button_gripper_open", 6)

        self.declare_parameter("gripper_close_tick", -653)
        self.declare_parameter("gripper_open_tick", 733)
        self.declare_parameter("gripper_step_tick", 50)

        self.declare_parameter("fine_close_axis", 5)
        self.declare_parameter("fine_open_axis", 2)
        self.declare_parameter("fine_trigger_threshold", 0.50)
        self.declare_parameter("fine_repeat_sec", 0.15)

        # Legacy optional D-pad control. Negative axis disables it.
        self.declare_parameter("dpad_axis", -1)
        self.declare_parameter("dpad_threshold", 0.50)
        self.declare_parameter("dpad_invert", False)
        self.declare_parameter("dpad_repeat_sec", 0.15)

        self.declare_parameter("button_debounce_sec", 0.20)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.gripper_command_topic = str(
            self.get_parameter("gripper_command_topic").value
        )
        self.gripper_present_position_topic = str(
            self.get_parameter("gripper_present_position_topic").value
        )

        self.button_start = int(self.get_parameter("button_start").value)
        self.button_end = int(self.get_parameter("button_end").value)
        self.button_gripper_close = int(
            self.get_parameter("button_gripper_close").value
        )
        self.button_gripper_open = int(
            self.get_parameter("button_gripper_open").value
        )
        self.gripper_close_tick = int(self.get_parameter("gripper_close_tick").value)
        self.gripper_open_tick = int(self.get_parameter("gripper_open_tick").value)
        self.gripper_step_tick = max(
            1,
            int(self.get_parameter("gripper_step_tick").value),
        )

        self.fine_close_axis = int(self.get_parameter("fine_close_axis").value)
        self.fine_open_axis = int(self.get_parameter("fine_open_axis").value)
        self.fine_trigger_threshold = abs(
            float(self.get_parameter("fine_trigger_threshold").value)
        )
        self.fine_repeat_sec = max(
            0.0,
            float(self.get_parameter("fine_repeat_sec").value),
        )

        self.dpad_axis = int(self.get_parameter("dpad_axis").value)
        self.dpad_threshold = abs(float(self.get_parameter("dpad_threshold").value))
        self.dpad_invert = bool(self.get_parameter("dpad_invert").value)
        self.dpad_repeat_sec = max(
            0.0,
            float(self.get_parameter("dpad_repeat_sec").value),
        )

        self.button_debounce_sec = float(self.get_parameter("button_debounce_sec").value)

        self.prev_buttons: Optional[List[int]] = None
        self.prev_dpad_sign = 0
        self.prev_fine_close_pressed = False
        self.prev_fine_open_pressed = False
        self.last_recorder_button_time = 0.0
        self.last_gripper_button_time = 0.0
        self.last_fine_step_time = 0.0
        self.last_dpad_time = 0.0
        self.latest_gripper_position: Optional[int] = None
        self.gripper_target_tick: Optional[int] = None

        self.recorder_cmd_pub = self.create_publisher(String, self.command_topic, 10)
        self.gripper_cmd_pub = self.create_publisher(
            Int32,
            self.gripper_command_topic,
            10,
        )
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 20)
        self.gripper_position_sub = self.create_subscription(
            Int32,
            self.gripper_present_position_topic,
            self.gripper_position_callback,
            20,
        )

        self.get_logger().info(block("JOY CONTROLLER READY", [
            ("joy_topic", self.joy_topic),
            ("recorder_topic", self.command_topic),
            ("gripper_topic", self.gripper_command_topic),
            ("gripper_position", self.gripper_present_position_topic),
            ("RB", f"button[{self.button_start}] -> start_recording"),
            ("LB", f"button[{self.button_end}] -> end_recording"),
            (
                "Start",
                f"button[{self.button_gripper_close}] -> close={self.gripper_close_tick}",
            ),
            (
                "Back",
                f"button[{self.button_gripper_open}] -> open={self.gripper_open_tick}",
            ),
            (
                "RT/LT",
                f"axis[{self.fine_close_axis}]/axis[{self.fine_open_axis}] "
                f"-> close/open step={self.gripper_step_tick}",
            ),
        ]))

    @property
    def gripper_min_tick(self) -> int:
        return min(self.gripper_close_tick, self.gripper_open_tick)

    @property
    def gripper_max_tick(self) -> int:
        return max(self.gripper_close_tick, self.gripper_open_tick)

    def clamp_gripper_tick(self, tick: int) -> int:
        return max(self.gripper_min_tick, min(int(tick), self.gripper_max_tick))

    def gripper_position_callback(self, msg: Int32):
        self.latest_gripper_position = self.clamp_gripper_tick(msg.data)

    def publish_recorder_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.recorder_cmd_pub.publish(msg)
        self.get_logger().info(f"[JOY RECORDER] {cmd}")

    def publish_gripper_command(self, tick: int, action: str):
        tick = self.clamp_gripper_tick(tick)
        self.gripper_target_tick = tick
        msg = Int32()
        msg.data = tick
        self.gripper_cmd_pub.publish(msg)
        self.get_logger().info(f"[JOY GRIPPER] {action} -> {msg.data}")

    def _dpad_sign(self, axes: List[float]) -> int:
        if self.dpad_axis < 0 or self.dpad_axis >= len(axes):
            return 0

        value = float(axes[self.dpad_axis])
        if self.dpad_invert:
            value = -value
        if value > self.dpad_threshold:
            return 1
        if value < -self.dpad_threshold:
            return -1
        return 0

    def _trigger_pressed(self, axes: List[float], axis_idx: int) -> bool:
        """Return true when an F710 trigger axis is pressed toward -1."""
        if axis_idx < 0 or axis_idx >= len(axes):
            return False
        return float(axes[axis_idx]) <= -self.fine_trigger_threshold

    @staticmethod
    def _step_toward(current: int, target: int, step: int) -> int:
        if current < target:
            return min(current + step, target)
        if current > target:
            return max(current - step, target)
        return target

    def _gripper_step_base(self) -> Optional[int]:
        if self.gripper_target_tick is not None:
            return self.gripper_target_tick
        if self.latest_gripper_position is not None:
            return self.latest_gripper_position
        return None

    @staticmethod
    def _button_pressed(curr_buttons: List[int], prev_buttons: List[int], idx: int) -> bool:
        if idx < 0:
            return False
        if idx >= len(curr_buttons) or idx >= len(prev_buttons):
            return False
        return curr_buttons[idx] == 1 and prev_buttons[idx] == 0

    def _handle_recorder_buttons(
        self,
        curr_buttons: List[int],
        prev_buttons: List[int],
        now: float,
    ):
        if now - self.last_recorder_button_time < self.button_debounce_sec:
            return

        if self._button_pressed(curr_buttons, prev_buttons, self.button_start):
            self.publish_recorder_command("start_recording")
            self.last_recorder_button_time = now
        elif self._button_pressed(curr_buttons, prev_buttons, self.button_end):
            self.publish_recorder_command("end_recording")
            self.last_recorder_button_time = now

    def _handle_gripper_buttons(
        self,
        curr_buttons: List[int],
        prev_buttons: List[int],
        now: float,
    ):
        if now - self.last_gripper_button_time < self.button_debounce_sec:
            return

        if self._button_pressed(curr_buttons, prev_buttons, self.button_gripper_close):
            self.publish_gripper_command(self.gripper_close_tick, "close")
            self.last_gripper_button_time = now
        elif self._button_pressed(curr_buttons, prev_buttons, self.button_gripper_open):
            self.publish_gripper_command(self.gripper_open_tick, "open")
            self.last_gripper_button_time = now

    def _handle_gripper_dpad(self, axes: List[float], now: float):
        sign = self._dpad_sign(axes)
        should_step = sign != 0 and (
            self.prev_dpad_sign == 0
            or now - self.last_dpad_time >= self.dpad_repeat_sec
        )
        self.prev_dpad_sign = sign

        if not should_step:
            return

        base_tick = self._gripper_step_base()
        if base_tick is None:
            self.get_logger().warning(
                "[JOY GRIPPER] D-pad command ignored: waiting for present position"
            )
            self.last_dpad_time = now
            return

        endpoint = self.gripper_close_tick if sign > 0 else self.gripper_open_tick
        action = "step_close" if sign > 0 else "step_open"
        tick = self._step_toward(base_tick, endpoint, self.gripper_step_tick)
        self.publish_gripper_command(tick, action)
        self.last_dpad_time = now

    def _handle_gripper_fine_triggers(self, axes: List[float], now: float):
        close_pressed = self._trigger_pressed(axes, self.fine_close_axis)
        open_pressed = self._trigger_pressed(axes, self.fine_open_axis)

        close_rising = close_pressed and not self.prev_fine_close_pressed
        open_rising = open_pressed and not self.prev_fine_open_pressed
        repeat_ready = now - self.last_fine_step_time >= self.fine_repeat_sec

        self.prev_fine_close_pressed = close_pressed
        self.prev_fine_open_pressed = open_pressed

        # Simultaneous RT/LT is ambiguous, so do not move the gripper.
        if close_pressed and open_pressed:
            return

        direction = 0
        if close_pressed and (close_rising or repeat_ready):
            direction = 1
        elif open_pressed and (open_rising or repeat_ready):
            direction = -1

        if direction == 0:
            return

        base_tick = self._gripper_step_base()
        if base_tick is None:
            self.get_logger().warning(
                "[JOY GRIPPER] RT/LT command ignored: waiting for present position"
            )
            self.last_fine_step_time = now
            return

        endpoint = (
            self.gripper_close_tick
            if direction > 0
            else self.gripper_open_tick
        )
        action = "step_close" if direction > 0 else "step_open"
        tick = self._step_toward(base_tick, endpoint, self.gripper_step_tick)
        self.publish_gripper_command(tick, action)
        self.last_fine_step_time = now

    def joy_callback(self, msg: Joy):
        curr_buttons = list(msg.buttons)
        axes = list(msg.axes)

        if self.prev_buttons is None:
            self.prev_buttons = curr_buttons
            self.prev_dpad_sign = self._dpad_sign(axes)
            self.prev_fine_close_pressed = self._trigger_pressed(
                axes,
                self.fine_close_axis,
            )
            self.prev_fine_open_pressed = self._trigger_pressed(
                axes,
                self.fine_open_axis,
            )
            return

        now = time.time()

        self._handle_recorder_buttons(curr_buttons, self.prev_buttons, now)
        self._handle_gripper_buttons(curr_buttons, self.prev_buttons, now)
        self._handle_gripper_fine_triggers(axes, now)
        self._handle_gripper_dpad(axes, now)

        self.prev_buttons = curr_buttons


def main(args=None):
    rclpy.init(args=args)
    node = VRDemoJoyController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
