#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

"""
UMI Gripper PWM Control Node (keyboard)

Keys (terminal):
  o : open  (negative pwm)
  c : close (positive pwm)
  s : stop  (0 pwm)
  q : quit
"""

# Dynamixel control table
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_CURRENT_LIMIT = 38
ADDR_GOAL_PWM = 100
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132

OPMODE_PWM = 16
TORQUE_DISABLE = 0
TORQUE_ENABLE = 1


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def lsb_signed(u16):
    return u16 - 0x10000 if u16 is not None and u16 > 0x7FFF else u16


def mA2lsb(ma):
    return int(round(ma / 2.69))


class TerminalKeyReader:
    def __init__(self):
        self._enabled = sys.stdin.isatty()
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._th = None

    def start(self):
        if not self._enabled:
            return
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        if not self._enabled:
            return
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=0.5)

    def _run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ch = sys.stdin.read(1)
                if ch:
                    self._q.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def get_key(self):
        if not self._enabled:
            return None
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


class GripperPwmNode(Node):
    def __init__(self):
        super().__init__("umi_gripper_sub_pwm")

        self.declare_parameter("dxl.port", "/dev/ttyUSB0")
        self.declare_parameter("dxl.baud", 3000000)
        self.declare_parameter("dxl.gripper_id", 1)
        self.declare_parameter("dxl.cmd_rate_hz", 200.0)
        self.declare_parameter("dxl.current_limit_mA", 300)
        self.declare_parameter("dxl.goal_pwm", 120)
        self.declare_parameter("dxl.goal_pwm_limit", 300)  # XH430: 0~885
        self.declare_parameter("gripper.min_tick", 1800)
        self.declare_parameter("gripper.max_tick", 4000)
        self.declare_parameter("gripper.present_current_mA_topic", "/gripper/present_current_mA")
        self.declare_parameter("gripper.present_position_topic", "/gripper/present_position")

        self.port_name = self.get_parameter("dxl.port").get_parameter_value().string_value
        self.baud = int(self.get_parameter("dxl.baud").get_parameter_value().integer_value)
        self.gripper_id = int(self.get_parameter("dxl.gripper_id").get_parameter_value().integer_value)
        self.cmd_rate_hz = float(self.get_parameter("dxl.cmd_rate_hz").get_parameter_value().double_value)
        self.current_limit_ma = int(self.get_parameter("dxl.current_limit_mA").get_parameter_value().integer_value)
        self.goal_pwm = int(self.get_parameter("dxl.goal_pwm").get_parameter_value().integer_value)
        self.goal_pwm_limit = int(self.get_parameter("dxl.goal_pwm_limit").get_parameter_value().integer_value)
        self.min_tick = int(self.get_parameter("gripper.min_tick").get_parameter_value().integer_value)
        self.max_tick = int(self.get_parameter("gripper.max_tick").get_parameter_value().integer_value)
        self.goal_pwm_limit = clamp(self.goal_pwm_limit, 1, 885)
        self.goal_pwm = clamp(self.goal_pwm, 1, self.goal_pwm_limit)

        cur_topic = self.get_parameter("gripper.present_current_mA_topic").get_parameter_value().string_value
        pos_topic = self.get_parameter("gripper.present_position_topic").get_parameter_value().string_value
        self.curr_pub = self.create_publisher(Float32, cur_topic, 20)
        self.pos_pub = self.create_publisher(Int32, pos_topic, 20)

        self.port = PortHandler(self.port_name)
        if not self.port.openPort():
            raise RuntimeError(f"openPort failed: {self.port_name}")
        if not self.port.setBaudRate(self.baud):
            raise RuntimeError(f"setBaudRate failed: {self.baud}")
        self.ph = PacketHandler(2.0)

        self._last_warn_time = {}
        self._last_sent_pwm = None
        self._active_pwm = 0
        self._stop_requested = False

        self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self._w1(ADDR_OPERATING_MODE, OPMODE_PWM)
        self._w2u(ADDR_CURRENT_LIMIT, clamp(mA2lsb(self.current_limit_ma), 1, 0xFFFF))
        self._w1(ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        time.sleep(0.05)

        self.key_reader = TerminalKeyReader()
        self.key_reader.start()

        self.get_logger().info(
            f"PWM node ON | port={self.port_name} baud={self.baud} id={self.gripper_id} "
            f"| pwm={self.goal_pwm} limit={self.goal_pwm_limit} "
            f"| pos_limit=[{self.min_tick}..{self.max_tick}]"
        )
        self.get_logger().info("Keys: o=open, c=close, s=stop, q=quit")
        self.timer = self.create_timer(1.0 / self.cmd_rate_hz, self._loop)

    def _log_warn_throttle(self, interval, msg):
        now = time.time()
        key = msg.split()[0] if msg else "default"
        if key not in self._last_warn_time or (now - self._last_warn_time[key]) >= interval:
            self.get_logger().warn(msg)
            self._last_warn_time[key] = now

    def _w1(self, addr, v):
        res, err = self.ph.write1ByteTxRx(self.port, self.gripper_id, addr, int(v) & 0xFF)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"write1 fail a={addr} r={res} e={err}")

    def _w2u(self, addr, v):
        v = int(v) & 0xFFFF
        res, err = self.ph.write2ByteTxRx(self.port, self.gripper_id, addr, v)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"write2u fail a={addr} r={res} e={err}")

    def _w2s(self, addr, v):
        v = int(v)
        if v < 0:
            v = (1 << 16) + v
        self._w2u(addr, v)

    def _r2u(self, addr):
        v, res, err = self.ph.read2ByteTxRx(self.port, self.gripper_id, addr)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"read2 fail a={addr} r={res} e={err}")
        return v

    def _r4u(self, addr):
        v, res, err = self.ph.read4ByteTxRx(self.port, self.gripper_id, addr)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"read4 fail a={addr} r={res} e={err}")
        return v

    def _loop(self):
        k = self.key_reader.get_key()
        if k == "o":
            self._active_pwm = -self.goal_pwm
            self.get_logger().info(f"OPEN pwm={self._active_pwm}")
        elif k == "c":
            self._active_pwm = self.goal_pwm
            self.get_logger().info(f"CLOSE pwm={self._active_pwm}")
        elif k == "s":
            self._active_pwm = 0
            self.get_logger().info("STOP pwm=0")
        elif k == "q":
            self._active_pwm = 0
            self._stop_requested = True

        if self._last_sent_pwm != self._active_pwm:
            self._w2s(ADDR_GOAL_PWM, self._active_pwm)
            self._last_sent_pwm = self._active_pwm

        cur_u = self._r2u(ADDR_PRESENT_CURRENT)
        cur_s = lsb_signed(cur_u)
        if cur_s is not None:
            msg = Float32()
            msg.data = float(cur_s * 2.69)
            self.curr_pub.publish(msg)

        pos_u = self._r4u(ADDR_PRESENT_POSITION)
        if pos_u is not None:
            # Stop applying PWM beyond position limits:
            # close(+pwm) at max_tick, open(-pwm) at min_tick.
            if self._active_pwm > 0 and int(pos_u) >= int(self.max_tick):
                self._active_pwm = 0
                self.get_logger().info(f"STOP at close limit: pos={int(pos_u)} >= {int(self.max_tick)}")
            elif self._active_pwm < 0 and int(pos_u) <= int(self.min_tick):
                self._active_pwm = 0
                self.get_logger().info(f"STOP at open limit: pos={int(pos_u)} <= {int(self.min_tick)}")

            # Ensure stop command is pushed immediately after limit hit.
            if self._last_sent_pwm != self._active_pwm:
                self._w2s(ADDR_GOAL_PWM, self._active_pwm)
                self._last_sent_pwm = self._active_pwm

            p = Int32()
            p.data = int(pos_u)
            self.pos_pub.publish(p)

        if self._stop_requested:
            rclpy.shutdown()

    def destroy_node(self):
        try:
            self._w2s(ADDR_GOAL_PWM, 0)
        except Exception:
            pass
        try:
            self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        except Exception:
            pass
        try:
            self.port.closePort()
        except Exception:
            pass
        self.key_reader.stop()
        self.get_logger().info("umi gripper pwm node: stopped.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GripperPwmNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
