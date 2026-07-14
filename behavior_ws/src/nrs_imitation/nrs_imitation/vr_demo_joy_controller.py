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
  A        buttons[0]  -> gripper close
  B        buttons[1]  -> gripper open
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

        self.declare_parameter("button_start", 5)
        self.declare_parameter("button_end", 4)
        self.declare_parameter("button_gripper_close", 0)
        self.declare_parameter("button_gripper_open", 1)

        self.declare_parameter("gripper_close_tick", -653)
        self.declare_parameter("gripper_open_tick", 733)

        self.declare_parameter("button_debounce_sec", 0.20)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.gripper_command_topic = str(
            self.get_parameter("gripper_command_topic").value
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

        self.button_debounce_sec = float(self.get_parameter("button_debounce_sec").value)

        self.prev_buttons: Optional[List[int]] = None
        self.last_recorder_button_time = 0.0
        self.last_gripper_button_time = 0.0

        self.recorder_cmd_pub = self.create_publisher(String, self.command_topic, 10)
        self.gripper_cmd_pub = self.create_publisher(
            Int32,
            self.gripper_command_topic,
            10,
        )
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 20)

        self.get_logger().info(block("JOY CONTROLLER READY", [
            ("joy_topic", self.joy_topic),
            ("recorder_topic", self.command_topic),
            ("gripper_topic", self.gripper_command_topic),
            ("RB", f"button[{self.button_start}] -> start_recording"),
            ("LB", f"button[{self.button_end}] -> end_recording"),
            ("A", f"button[{self.button_gripper_close}] -> close={self.gripper_close_tick}"),
            ("B", f"button[{self.button_gripper_open}] -> open={self.gripper_open_tick}"),
        ]))

    def publish_recorder_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.recorder_cmd_pub.publish(msg)
        self.get_logger().info(f"[JOY RECORDER] {cmd}")

    def publish_gripper_command(self, tick: int, action: str):
        msg = Int32()
        msg.data = int(tick)
        self.gripper_cmd_pub.publish(msg)
        self.get_logger().info(f"[JOY GRIPPER] {action} -> {msg.data}")

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

    def joy_callback(self, msg: Joy):
        curr_buttons = list(msg.buttons)

        if self.prev_buttons is None:
            self.prev_buttons = curr_buttons
            return

        now = time.time()

        self._handle_recorder_buttons(curr_buttons, self.prev_buttons, now)
        self._handle_gripper_buttons(curr_buttons, self.prev_buttons, now)

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
