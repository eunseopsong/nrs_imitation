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

Default Logitech F710 / Xbox-like mapping:
  LB       buttons[4]  -> start_recording
  RB       buttons[5]  -> end_recording
"""

import time
from typing import List, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import String

from nrs_imitation.pretty_print import block


class VRDemoJoyController(Node):
    def __init__(self):
        super().__init__("vr_demo_joy_controller")

        self.declare_parameter("joy_topic", "/joy_il")
        self.declare_parameter("command_topic", "/vr_demo_recorder/command")

        self.declare_parameter("button_start", 4)
        self.declare_parameter("button_end", 5)
        self.declare_parameter("button_a", -1)
        self.declare_parameter("button_b", -1)

        self.declare_parameter("button_debounce_sec", 0.20)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.button_start = int(self.get_parameter("button_start").value)
        self.button_end = int(self.get_parameter("button_end").value)

        button_a = int(self.get_parameter("button_a").value)
        button_b = int(self.get_parameter("button_b").value)
        if button_a >= 0:
            self.button_start = button_a
        if button_b >= 0:
            self.button_end = button_b

        self.button_debounce_sec = float(self.get_parameter("button_debounce_sec").value)

        self.prev_buttons: Optional[List[int]] = None
        self.last_button_cmd_time = 0.0

        self.cmd_pub = self.create_publisher(String, self.command_topic, 10)
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 20)

        self.get_logger().info(block("JOY CONTROLLER READY", [
            ("joy_topic", self.joy_topic),
            ("command_topic", self.command_topic),
            ("LB", f"button[{self.button_start}] -> start_recording"),
            ("RB", f"button[{self.button_end}] -> end_recording"),
        ]))

    def publish_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"[JOY CMD] {cmd}")

    @staticmethod
    def _button_pressed(curr_buttons: List[int], prev_buttons: List[int], idx: int) -> bool:
        if idx < 0:
            return False
        if idx >= len(curr_buttons) or idx >= len(prev_buttons):
            return False
        return curr_buttons[idx] == 1 and prev_buttons[idx] == 0

    def joy_callback(self, msg: Joy):
        curr_buttons = list(msg.buttons)

        if self.prev_buttons is None:
            self.prev_buttons = curr_buttons
            return

        now = time.time()

        if now - self.last_button_cmd_time >= self.button_debounce_sec:
            if self._button_pressed(curr_buttons, self.prev_buttons, self.button_start):
                self.publish_command("start_recording")
                self.last_button_cmd_time = now
            elif self._button_pressed(curr_buttons, self.prev_buttons, self.button_end):
                self.publish_command("end_recording")
                self.last_button_cmd_time = now

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
