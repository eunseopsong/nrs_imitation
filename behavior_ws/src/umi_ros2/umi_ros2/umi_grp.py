#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
import cv2
import numpy as np

"""
umi Gripper Control Node (ROS2)

- Input:
    Keyboard input via OpenCV waitKey
    o / O : open
    c / C : close
    + / = : increase target
    - / _ : decrease target
    ESC   : exit

- Motor:
    Dynamixel XM430-W350
    ID = 0
    Baudrate = 57600 bps
    Protocol = 2.0
    Operating Mode = 5, Current-based Position Control Mode

- Publish:
    /gripper/present_current_mA : Float32
    /gripper/present_position   : Int32
"""

# ================= Control Table =================
ADDR_OPERATING_MODE       = 11
ADDR_CURRENT_LIMIT        = 38
ADDR_MAX_POSITION_LIMIT   = 48
ADDR_MIN_POSITION_LIMIT   = 52
ADDR_TORQUE_ENABLE        = 64
ADDR_GOAL_CURRENT         = 102
ADDR_PROFILE_ACCEL        = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_CURRENT      = 126  # signed 2B, 1 LSB ≈ 2.69 mA
ADDR_PRESENT_POSITION     = 132
LEN_PRESENT_STATE         = (ADDR_PRESENT_POSITION + 4) - ADDR_PRESENT_CURRENT
OFFSET_PRESENT_CURRENT    = 0
OFFSET_PRESENT_POSITION   = ADDR_PRESENT_POSITION - ADDR_PRESENT_CURRENT

# Operating Mode
# 3: Position Control Mode
# 5: Current-based Position Control Mode
OPMODE_CURRENT_BASED_POSITION = 5

TORQUE_DISABLE = 0
TORQUE_ENABLE  = 1


# ================= Utilities =================
def clamp(v, a, b):
    return a if v < a else (b if v > b else v)


def map_range(x, x0, x1, y0, y1):
    if x1 == x0:
        return int(y0)

    r = (float(x) - x0) / float(x1 - x0)
    r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)

    return int(round(y0 + r * (y1 - y0)))


def mA2lsb(mA):
    return int(round(mA / 2.69))


def lsb_signed(u16):
    return u16 - 0x10000 if u16 is not None and u16 > 0x7FFF else u16


def i32_signed(u32):
    return u32 - 0x100000000 if u32 is not None and u32 > 0x7FFFFFFF else u32


def le_u16(data, offset):
    return int(data[offset]) | (int(data[offset + 1]) << 8)


def le_u32(data, offset):
    return (
        int(data[offset])
        | (int(data[offset + 1]) << 8)
        | (int(data[offset + 2]) << 16)
        | (int(data[offset + 3]) << 24)
    )


# ================= Node Body =================
class GripperNode(Node):
    def __init__(self):
        super().__init__('umi_gripper')

        # ================= Parameters =================
        # Dynamixel Wizard setting 기준
        self.declare_parameter('dxl.port', '/dev/ttyUSB0')
        self.declare_parameter('dxl.baud', 57600)
        self.declare_parameter('dxl.gripper_id', 0)

        # Trigger input range
        self.declare_parameter('trigger.min_tick', 590)
        self.declare_parameter('trigger.max_tick', 2500)

        # Gripper position range
        # Dynamixel Wizard:
        # Min Position Limit = 590
        # Max Position Limit = 2500
        self.declare_parameter('gripper.min_tick', 590)
        self.declare_parameter('gripper.max_tick', 2500)

        # False:
        #   o -> 590
        #   c -> 2500
        # 만약 c를 눌렀을 때 열리면 True로 변경
        self.declare_parameter('gripper.invert', False)

        # Command tuning
        self.declare_parameter('dxl.cmd_rate_hz', 200.0)
        self.declare_parameter('dxl.pos_deadband', 2)

        # 너무 빠른 움직임 방지용 소프트 제한
        self.declare_parameter('dxl.pos_slew_per_sec', 3000.0)

        # Dynamixel Wizard setting 기준
        # Profile Acceleration = 0
        # Profile Velocity = 0
        self.declare_parameter('dxl.profile_accel', 0)
        self.declare_parameter('dxl.profile_velocity', 0)

        # Dynamixel Wizard:
        # Current Limit = 500 LSB = 1345 mA
        self.declare_parameter('dxl.current_limit_mA', 1345)

        # Dynamixel Wizard:
        # Goal Current = 74 LSB = 199.06 mA
        self.declare_parameter('dxl.goal_current_mA', 200)

        # 물체를 잡았을 때 전류 기준으로 멈추는 기능
        self.declare_parameter('gripper.current_stop_enabled', True)

        # 너무 빨리 멈추면 250~300으로 올리면 됨
        self.declare_parameter('gripper.close_current_stop_mA', 400)
        self.declare_parameter('gripper.current_stop_debounce_sec', 0.12)

        # Monitoring
        self.declare_parameter('monitor.enabled', True)
        self.declare_parameter('monitor.print_period_sec', 0.5)

        # Topics
        self.declare_parameter('gripper.present_current_mA_topic', '/gripper/present_current_mA')
        self.declare_parameter('gripper.present_position_topic', '/gripper/present_position')
        self.declare_parameter('gripper.command_topic', '/gripper/command')

        # Keyboard
        self.declare_parameter('keyboard.step_size', 50)

        # ================= Get Parameters =================
        PORT = self.get_parameter('dxl.port').get_parameter_value().string_value
        BAUD = self.get_parameter('dxl.baud').get_parameter_value().integer_value
        PROTO = 2.0

        GRIPPER_ID = self.get_parameter('dxl.gripper_id').get_parameter_value().integer_value

        TR_MIN = self.get_parameter('trigger.min_tick').get_parameter_value().integer_value
        TR_MAX = self.get_parameter('trigger.max_tick').get_parameter_value().integer_value

        GR_MIN = self.get_parameter('gripper.min_tick').get_parameter_value().integer_value
        GR_MAX = self.get_parameter('gripper.max_tick').get_parameter_value().integer_value
        INVERT = self.get_parameter('gripper.invert').get_parameter_value().bool_value

        CMD_RATE_HZ = self.get_parameter('dxl.cmd_rate_hz').get_parameter_value().double_value
        POS_DEADBAND = self.get_parameter('dxl.pos_deadband').get_parameter_value().integer_value
        POS_SLEW_PER_SEC = self.get_parameter('dxl.pos_slew_per_sec').get_parameter_value().double_value

        PROFILE_ACCEL = self.get_parameter('dxl.profile_accel').get_parameter_value().integer_value
        PROFILE_VELOCITY = self.get_parameter('dxl.profile_velocity').get_parameter_value().integer_value
        CURRENT_LIMIT_mA = self.get_parameter('dxl.current_limit_mA').get_parameter_value().integer_value
        GOAL_CURRENT_mA = self.get_parameter('dxl.goal_current_mA').get_parameter_value().integer_value

        CURRENT_STOP_ENABLED = self.get_parameter('gripper.current_stop_enabled').get_parameter_value().bool_value
        CLOSE_CURRENT_STOP_mA = self.get_parameter('gripper.close_current_stop_mA').get_parameter_value().integer_value
        CURRENT_STOP_DEBOUNCE_SEC = self.get_parameter('gripper.current_stop_debounce_sec').get_parameter_value().double_value

        MONITOR_ENABLED = self.get_parameter('monitor.enabled').get_parameter_value().bool_value
        MONITOR_PRINT_PERIOD_SEC = self.get_parameter('monitor.print_period_sec').get_parameter_value().double_value

        CURR_TOPIC = self.get_parameter('gripper.present_current_mA_topic').get_parameter_value().string_value
        POS_TOPIC = self.get_parameter('gripper.present_position_topic').get_parameter_value().string_value
        CMD_TOPIC = self.get_parameter('gripper.command_topic').get_parameter_value().string_value

        KEYBOARD_STEP = self.get_parameter('keyboard.step_size').get_parameter_value().integer_value

        # ================= Store Variables =================
        self._lk = threading.Lock()
        self._tr_tick = None
        self._direct_goal_position = None
        self._last_cmd = None

        self.PORT = PORT
        self.BAUD = BAUD
        self.PROTO = PROTO
        self.GRIPPER_ID = GRIPPER_ID

        self.TR_MIN = TR_MIN
        self.TR_MAX = TR_MAX
        self.GR_MIN = GR_MIN
        self.GR_MAX = GR_MAX
        self.INVERT = INVERT

        self.CMD_RATE_HZ = CMD_RATE_HZ
        self.POS_DEADBAND = POS_DEADBAND
        self.POS_SLEW_PER_SEC = POS_SLEW_PER_SEC

        self.PROFILE_ACCEL = PROFILE_ACCEL
        self.PROFILE_VELOCITY = PROFILE_VELOCITY
        self.CURRENT_LIMIT_mA = CURRENT_LIMIT_mA
        self.GOAL_CURRENT_mA = GOAL_CURRENT_mA

        self.CURRENT_STOP_ENABLED = CURRENT_STOP_ENABLED
        self.CLOSE_CURRENT_STOP_mA = CLOSE_CURRENT_STOP_mA
        self.CURRENT_STOP_DEBOUNCE_SEC = max(0.0, CURRENT_STOP_DEBOUNCE_SEC)

        self.MONITOR_ENABLED = MONITOR_ENABLED
        self.MONITOR_PRINT_PERIOD_SEC = MONITOR_PRINT_PERIOD_SEC
        self._last_monitor_print_time = 0.0

        self.CLOSE_INCREASES_TICK = not INVERT
        self.KEYBOARD_STEP = KEYBOARD_STEP

        self._close_stop_latched = False
        self._close_stop_position = None
        self._close_overcurrent_start_time = None

        self._last_warn_time = {}
        self._shutdown_requested = threading.Event()
        self._destroyed = False

        # ================= ROS Publishers =================
        self.curr_pub = self.create_publisher(Float32, CURR_TOPIC, 20)
        self.pos_pub = self.create_publisher(Int32, POS_TOPIC, 20)
        self.cmd_sub = self.create_subscription(Int32, CMD_TOPIC, self._cmd_callback, 10)

        # Do not command open/close on startup. The first goal should come from
        # keyboard input or /gripper/command after the operator checks the state.
        self._tr_tick = None

        # ================= Dynamixel Init =================
        self.port = PortHandler(PORT)

        if not self.port.openPort():
            raise RuntimeError(f"openPort failed: {PORT}")

        if not self.port.setBaudRate(BAUD):
            raise RuntimeError(f"setBaudRate failed: {BAUD}")

        self.ph = PacketHandler(PROTO)

        # Safety sequence:
        # Torque OFF -> Mode / Current / Profile / Goal Current 설정
        # -> 현재 위치를 Goal Position에 seed -> Torque ON
        self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

        self._w1(ADDR_OPERATING_MODE, OPMODE_CURRENT_BASED_POSITION)

        # Current Limit: Address 38
        current_limit_lsb = clamp(mA2lsb(CURRENT_LIMIT_mA), 1, 0xFFFF)
        self._w2u(ADDR_CURRENT_LIMIT, current_limit_lsb)

        # Goal Current: Address 102
        goal_current_lsb = clamp(mA2lsb(GOAL_CURRENT_mA), 1, 0xFFFF)
        self._w2u(ADDR_GOAL_CURRENT, goal_current_lsb)

        # Profile 설정
        self._w4u(ADDR_PROFILE_ACCEL, PROFILE_ACCEL)
        self._w4u(ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)

        # 현재 위치를 goal register에 먼저 넣어서 torque on 시 이전 goal로 움직이지 않게 함.
        startup_cur_u = None
        startup_pos_u = None
        startup_pos = None

        for _ in range(3):
            startup_cur_u, startup_pos_u = self._read_present_state(warn=False)

            if startup_pos_u is not None:
                startup_pos = int(i32_signed(startup_pos_u))
                break

            time.sleep(0.05)

        startup_cur_s = lsb_signed(startup_cur_u)
        startup_current_mA = (
            float(startup_cur_s * 2.69)
            if startup_cur_s is not None
            else None
        )

        startup_goal_seeded = False

        if startup_pos is not None and self.GR_MIN <= startup_pos <= self.GR_MAX:
            startup_goal_seeded = self._w4u(ADDR_GOAL_POSITION, startup_pos)
            self._last_cmd = startup_pos if startup_goal_seeded else None
        elif startup_pos is not None:
            self._last_cmd = None
            self.get_logger().warn(
                f"Startup position {startup_pos} tick "
                f"(raw={startup_pos_u}) is outside configured range "
                f"[{self.GR_MIN}, {self.GR_MAX}]; not seeding Goal Position."
            )
        else:
            self._last_cmd = None
            self.get_logger().warn(
                "Could not read startup position; torque on may reuse the "
                "previous motor goal."
            )

        self._w1(ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        time.sleep(0.1)

        enabled_cur_u, enabled_pos_u = self._read_present_state(warn=False)
        enabled_pos = (
            int(i32_signed(enabled_pos_u))
            if enabled_pos_u is not None
            else None
        )
        enabled_cur_s = lsb_signed(enabled_cur_u)
        enabled_current_mA = (
            float(enabled_cur_s * 2.69)
            if enabled_cur_s is not None
            else None
        )

        startup_pos_str = "None" if startup_pos is None else str(startup_pos)
        startup_pos_raw_str = "None" if startup_pos_u is None else str(startup_pos_u)
        startup_cur_str = (
            "None"
            if startup_current_mA is None
            else f"{startup_current_mA:.2f}"
        )
        enabled_pos_str = "None" if enabled_pos is None else str(enabled_pos)
        enabled_pos_raw_str = "None" if enabled_pos_u is None else str(enabled_pos_u)
        enabled_cur_str = (
            "None"
            if enabled_current_mA is None
            else f"{enabled_current_mA:.2f}"
        )
        hold_goal_str = "None" if self._last_cmd is None else str(self._last_cmd)

        self.get_logger().info(
            "Startup gripper state | "
            f"startup_pos={startup_pos_str} tick | "
            f"startup_pos_raw={startup_pos_raw_str} | "
            f"startup_current={startup_cur_str} mA | "
            f"hold_goal={hold_goal_str} | "
            f"goal_seeded={startup_goal_seeded} | "
            f"after_torque_pos={enabled_pos_str} tick | "
            f"after_torque_pos_raw={enabled_pos_raw_str} | "
            f"after_torque_current={enabled_cur_str} mA | "
            "initial_motion_command=disabled"
        )

        self.get_logger().info(
            f"umi gripper node ON | "
            f"port={PORT} baud={BAUD} id={GRIPPER_ID} | "
            f"command_topic={CMD_TOPIC} | "
            f"mode=Current-based Position Control(5) | "
            f"map [{TR_MIN}..{TR_MAX}] -> [{GR_MIN}..{GR_MAX}] "
            f"invert={INVERT} | "
            f"software_range=[{GR_MIN}..{GR_MAX}] | "
            f"current_limit={CURRENT_LIMIT_mA}mA "
            f"goal_current={GOAL_CURRENT_mA}mA | "
            f"current_stop={CURRENT_STOP_ENABLED} "
            f"threshold={CLOSE_CURRENT_STOP_mA}mA "
            f"debounce={self.CURRENT_STOP_DEBOUNCE_SEC:.2f}s"
        )

        # Keyboard input
        self._setup_opencv_keyboard()

        # Main control loop timer
        self.timer = self.create_timer(1.0 / CMD_RATE_HZ, self._control_loop)

    # ================= Keyboard =================
    def _setup_opencv_keyboard(self):
        """Setup OpenCV window and keyboard input in a separate thread"""

        def keyboard_thread():
            cv2.namedWindow('Gripper Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gripper Control', 360, 120)

            img = np.zeros((120, 360, 3), dtype=np.uint8)

            cv2.putText(
                img,
                'Gripper Control',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                'o=open, c=close',
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.putText(
                img,
                '+/-=adjust, ESC=exit',
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.imshow('Gripper Control', img)

            try:
                while rclpy.ok() and not self._shutdown_requested.is_set():
                    key = cv2.waitKey(10) & 0xFF

                    if key == 27:
                        self._shutdown_requested.set()
                        self.get_logger().info("ESC pressed - shutting down...")

                        def shutdown_ros():
                            rclpy.shutdown()

                        threading.Thread(target=shutdown_ros, daemon=True).start()
                        break

                    elif key != 255:
                        try:
                            with self._lk:
                                current = self._tr_tick if self._tr_tick is not None else (
                                    self.TR_MIN + self.TR_MAX
                                ) // 2

                                if key == ord('o') or key == ord('O'):
                                    new_val = self.TR_MIN
                                    self._tr_tick = clamp(new_val, self.TR_MIN, self.TR_MAX)
                                    self._direct_goal_position = None
                                    self.get_logger().info(f"Open: trigger={self._tr_tick}")

                                elif key == ord('c') or key == ord('C'):
                                    new_val = self.TR_MAX
                                    self._tr_tick = clamp(new_val, self.TR_MIN, self.TR_MAX)
                                    self._direct_goal_position = None
                                    self.get_logger().info(f"Close: trigger={self._tr_tick}")

                                elif key == ord('+') or key == ord('='):
                                    new_val = current + self.KEYBOARD_STEP
                                    self._tr_tick = clamp(new_val, self.TR_MIN, self.TR_MAX)
                                    self._direct_goal_position = None
                                    self.get_logger().info(f"Increase: trigger={self._tr_tick}")

                                elif key == ord('-') or key == ord('_'):
                                    new_val = current - self.KEYBOARD_STEP
                                    self._tr_tick = clamp(new_val, self.TR_MIN, self.TR_MAX)
                                    self._direct_goal_position = None
                                    self.get_logger().info(f"Decrease: trigger={self._tr_tick}")

                        except Exception as e:
                            self.get_logger().warn(f"Keyboard handler error: {e}")

            finally:
                cv2.destroyAllWindows()

        self.keyboard_thread = threading.Thread(target=keyboard_thread, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(
            "Keyboard controls: "
            "'o'/'O'=open, 'c'/'C'=close, '+/-'=adjust, ESC=exit"
        )

    def _cmd_callback(self, msg: Int32):
        """Receive a direct gripper target position in motor ticks."""
        goal = clamp(msg.data, self.GR_MIN, self.GR_MAX)
        with self._lk:
            self._direct_goal_position = goal
        self.get_logger().info(f"Topic command: {msg.data} -> goal={goal}")

    # ================= Dynamixel I/O =================
    def _w1(self, addr, v):
        res = None
        err = None

        for attempt in range(2):
            res, err = self.ph.write1ByteTxRx(
                self.port,
                self.GRIPPER_ID,
                addr,
                int(v) & 0xFF
            )

            if res == COMM_SUCCESS and err == 0:
                return True

            if attempt == 0:
                time.sleep(0.003)

        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(
                1.0,
                f"write1 fail addr={addr} val={v} res={res} err={err}"
            )
            return False

        return True

    def _w2u(self, addr, v):
        v = int(v) & 0xFFFF

        res = None
        err = None

        for attempt in range(2):
            res, err = self.ph.write2ByteTxRx(
                self.port,
                self.GRIPPER_ID,
                addr,
                v
            )

            if res == COMM_SUCCESS and err == 0:
                return True

            if attempt == 0:
                time.sleep(0.003)

        if res != COMM_SUCCESS or err != 0:
            err_str = self._get_error_string(err) if err != 0 else "COMM_ERROR"
            self._log_warn_throttle(
                1.0,
                f"write2u fail addr={addr} val={v} res={res} err={err} ({err_str})"
            )
            return False

        return True

    def _w4u(self, addr, v):
        v = int(v) & 0xFFFFFFFF

        res = None
        err = None

        for attempt in range(2):
            res, err = self.ph.write4ByteTxRx(
                self.port,
                self.GRIPPER_ID,
                addr,
                v
            )

            if res == COMM_SUCCESS and err == 0:
                return True

            if attempt == 0:
                time.sleep(0.003)

        if res != COMM_SUCCESS or err != 0:
            err_str = self._get_error_string(err) if err != 0 else "COMM_ERROR"
            self._log_warn_throttle(
                1.0,
                f"write4u fail addr={addr} val={v} res={res} err={err} ({err_str})"
            )
            return False

        return True

    def _read_present_state(self, warn=True):
        data = []
        res = None
        err = None

        for attempt in range(2):
            data, res, err = self.ph.readTxRx(
                self.port,
                self.GRIPPER_ID,
                ADDR_PRESENT_CURRENT,
                LEN_PRESENT_STATE,
            )

            if (
                res == COMM_SUCCESS
                and err == 0
                and len(data) >= LEN_PRESENT_STATE
            ):
                cur_u = le_u16(data, OFFSET_PRESENT_CURRENT)
                pos_u = le_u32(data, OFFSET_PRESENT_POSITION)
                return cur_u, pos_u

            if attempt == 0:
                time.sleep(0.003)

        if warn:
            self._log_warn_throttle(
                1.0,
                f"read_state fail addr={ADDR_PRESENT_CURRENT} "
                f"len={LEN_PRESENT_STATE} res={res} err={err} "
                f"data_len={len(data)}"
            )

        return None, None

    def _r2u(self, addr):
        v, res, err = self.ph.read2ByteTxRx(
            self.port,
            self.GRIPPER_ID,
            addr
        )

        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(
                1.0,
                f"read2 fail addr={addr} res={res} err={err}"
            )
            return None

        return v

    def _r4u(self, addr):
        v, res, err = self.ph.read4ByteTxRx(
            self.port,
            self.GRIPPER_ID,
            addr
        )

        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(
                1.0,
                f"read4 fail addr={addr} res={res} err={err}"
            )
            return None

        return v

    def _get_error_string(self, err_code):
        error_map = {
            0: "NO_ERROR",
            1: "RESULT_FAIL",
            2: "INSTRUCTION_ERROR",
            3: "CRC_ERROR",
            4: "DATA_RANGE_ERROR",
            5: "DATA_LENGTH_ERROR",
            6: "DATA_LIMIT_ERROR",
            7: "ACCESS_ERROR",
            128: "ALERT"
        }

        return error_map.get(err_code, f"UNKNOWN({err_code})")

    def _log_warn_throttle(self, interval, msg):
        now = time.time()
        key = msg.split()[0] if msg else "default"

        if key not in self._last_warn_time or (now - self._last_warn_time[key]) >= interval:
            self.get_logger().warn(msg)
            self._last_warn_time[key] = now

    # ================= Current Stop Logic =================
    def _is_closing_motion(self, target, reference):
        if target is None or reference is None:
            return False

        if self.CLOSE_INCREASES_TICK:
            return target > reference + self.POS_DEADBAND

        return target < reference - self.POS_DEADBAND

    def _is_opening_motion(self, target, reference):
        if target is None or reference is None:
            return False

        if self.CLOSE_INCREASES_TICK:
            return target < reference - self.POS_DEADBAND

        return target > reference + self.POS_DEADBAND

    def _apply_current_stop(self, cmd, present_position, present_current_mA):
        if not self.CURRENT_STOP_ENABLED:
            self._close_overcurrent_start_time = None
            return cmd

        hold_position = present_position if present_position is not None else self._last_cmd

        if self._close_stop_latched:
            if self._is_opening_motion(cmd, self._close_stop_position):
                self._close_stop_latched = False
                self._close_stop_position = None
                self._close_overcurrent_start_time = None
                self.get_logger().info("Gripper current stop released by opening command.")
                return cmd

            return self._close_stop_position

        if (
            present_current_mA is not None
            and abs(present_current_mA) >= self.CLOSE_CURRENT_STOP_mA
            and self._is_closing_motion(cmd, self._last_cmd)
        ):
            now = time.time()

            if self._close_overcurrent_start_time is None:
                self._close_overcurrent_start_time = now
                return cmd

            if now - self._close_overcurrent_start_time < self.CURRENT_STOP_DEBOUNCE_SEC:
                return cmd

            self._close_stop_latched = True
            self._close_stop_position = int(hold_position) if hold_position is not None else int(cmd)
            self._close_overcurrent_start_time = None

            self._tr_tick = map_range(
                self._close_stop_position,
                self.GR_MIN if not self.INVERT else self.GR_MAX,
                self.GR_MAX if not self.INVERT else self.GR_MIN,
                self.TR_MIN,
                self.TR_MAX,
            )
            self._direct_goal_position = self._close_stop_position

            self.get_logger().warn(
                f"Close current stop latched at pos={self._close_stop_position}, "
                f"current={present_current_mA:.1f}mA. "
                f"Holding position until opening command."
            )

            return self._close_stop_position

        self._close_overcurrent_start_time = None
        return cmd

    # ================= Monitoring =================
    def _print_monitor(self, tr, goal, cmd, present_position, present_current_mA):
        if not self.MONITOR_ENABLED:
            return

        now = time.time()

        if now - self._last_monitor_print_time < self.MONITOR_PRINT_PERIOD_SEC:
            return

        self._last_monitor_print_time = now

        tr_str = "None" if tr is None else str(tr)
        goal_str = "None" if goal is None else str(goal)
        cmd_str = "None" if cmd is None else str(cmd)
        pos_str = "None" if present_position is None else str(present_position)
        cur_str = "None" if present_current_mA is None else f"{present_current_mA:.2f}"

        self.get_logger().info(
            f"[MONITOR] "
            f"trigger={tr_str} | "
            f"goal={goal_str} | "
            f"cmd={cmd_str} | "
            f"present_pos={pos_str} tick | "
            f"present_current={cur_str} mA | "
            f"last_cmd={self._last_cmd} | "
            f"range=[{self.GR_MIN}, {self.GR_MAX}]"
        )

    # ================= Main Loop =================
    def _control_loop(self):
        if self._shutdown_requested.is_set():
            return

        try:
            goal = None
            cmd = None

            with self._lk:
                tr = self._tr_tick
                direct_goal = self._direct_goal_position

            # Present Current / Position in one read transaction.
            cur_u, pos_u = self._read_present_state()
            cur_s = lsb_signed(cur_u)
            present_current_mA = float(cur_s * 2.69) if cur_s is not None else None

            present_position = (
                int(i32_signed(pos_u))
                if pos_u is not None
                else None
            )

            if direct_goal is not None:
                goal = clamp(direct_goal, self.GR_MIN, self.GR_MAX)
            elif tr is not None:
                # Trigger tick -> Goal position mapping
                if not self.INVERT:
                    goal = map_range(
                        tr,
                        self.TR_MIN,
                        self.TR_MAX,
                        self.GR_MIN,
                        self.GR_MAX
                    )
                else:
                    goal = map_range(
                        tr,
                        self.TR_MIN,
                        self.TR_MAX,
                        self.GR_MAX,
                        self.GR_MIN
                    )

                goal = clamp(goal, self.GR_MIN, self.GR_MAX)

            if goal is not None:

                # Soft slew, rate limiting
                if self._last_cmd is None:
                    cmd = goal
                else:
                    max_step = max(
                        1,
                        int(round(self.POS_SLEW_PER_SEC / self.CMD_RATE_HZ))
                    )

                    delta = goal - self._last_cmd

                    if delta > max_step:
                        delta = max_step
                    elif delta < -max_step:
                        delta = -max_step

                    cmd = self._last_cmd + delta

                cmd = clamp(int(round(cmd)), self.GR_MIN, self.GR_MAX)

                # Current stop
                cmd = self._apply_current_stop(
                    cmd,
                    present_position,
                    present_current_mA
                )

                # Send command
                if self._last_cmd is None or abs(cmd - self._last_cmd) > self.POS_DEADBAND:
                    if self.GR_MIN <= cmd <= self.GR_MAX:
                        if self._w4u(ADDR_GOAL_POSITION, cmd):
                            self._last_cmd = cmd
                    else:
                        self.get_logger().warn(
                            f"Position {cmd} out of configured range "
                            f"({self.GR_MIN}-{self.GR_MAX}), clamping"
                        )

                        cmd_clamped = clamp(cmd, self.GR_MIN, self.GR_MAX)

                        if self._w4u(ADDR_GOAL_POSITION, cmd_clamped):
                            self._last_cmd = cmd_clamped

                elif self._close_stop_latched and self._close_stop_position is not None:
                    hold_cmd = clamp(
                        int(self._close_stop_position),
                        self.GR_MIN,
                        self.GR_MAX
                    )

                    if self._w4u(ADDR_GOAL_POSITION, hold_cmd):
                        self._last_cmd = hold_cmd

            # Monitoring print
            self._print_monitor(
                tr,
                goal,
                cmd,
                present_position,
                present_current_mA
            )

            # Publish present current
            if present_current_mA is not None:
                msg = Float32()
                msg.data = present_current_mA
                self.curr_pub.publish(msg)

            # Publish present position
            if present_position is not None:
                pos_msg = Int32()
                pos_msg.data = present_position
                self.pos_pub.publish(pos_msg)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")

    # ================= Cleanup =================
    def destroy_node(self):
        if self._destroyed:
            return

        self._destroyed = True
        self._shutdown_requested.set()

        try:
            self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        except Exception:
            pass

        try:
            self.port.closePort()
        except Exception:
            pass

        self.get_logger().info("umi gripper node: torque OFF, port closed.")

        try:
            super().destroy_node()
        except KeyboardInterrupt:
            pass


# ================= main =================
def main(args=None):
    rclpy.init(args=args)

    node = None
    executor = None

    try:
        node = GripperNode()

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        try:
            while rclpy.ok() and not node._shutdown_requested.is_set():
                executor.spin_once(timeout_sec=0.1)

        except KeyboardInterrupt:
            pass

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            executor.shutdown()
        except Exception:
            pass

        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
