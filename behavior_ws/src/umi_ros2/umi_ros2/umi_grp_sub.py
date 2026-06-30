#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray, Int32
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

"""
umi Gripper Control Node (ROS2) - Topic-based
- Input:  /gripper/command (Int32) - target position in ticks
- Control: Dynamixel XM430-W350, Current-based Position Mode(5)
- Publish: /gripper/present_current_mA (Float32, mA)
          /gripper/present_position (Int32, position ticks)
"""

# ================= Parameters =================
# These will be set from ROS2 parameters in __init__

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

OPMODE_CURRENT_BASED_POSITION = 5
TORQUE_DISABLE = 0
TORQUE_ENABLE  = 1

# ================= Utilities =================
def clamp(v, a, b): return a if v < a else (b if v > b else v)

def mA2lsb(mA): return int(round(mA / 2.69))

def joy_to_tick(value, min_tick, max_tick):
    value = max(-1.0, min(1.0, float(value)))
    ratio = (value + 1.0) * 0.5
    return int(round(min_tick + ratio * (max_tick - min_tick)))

def lsb_signed(u16):
    return u16 - 0x10000 if u16 is not None and u16 > 0x7FFF else u16

# ================= Node Body =================
class GripperSubNode(Node):
    def __init__(self):
        super().__init__('umi_gripper_sub')
        
        # Declare parameters
        self.declare_parameter('dxl.port', '/dev/ttyUSB0')
        self.declare_parameter('dxl.baud', 57600)
        self.declare_parameter('dxl.gripper_id', 0)
        self.declare_parameter('gripper.min_tick', 590)
        self.declare_parameter('gripper.max_tick', 2500)
        self.declare_parameter('dxl.cmd_rate_hz', 200.0)
        self.declare_parameter('dxl.pos_deadband', 2)
        self.declare_parameter('dxl.pos_slew_per_sec', 3000.0)
        self.declare_parameter('dxl.profile_accel', 0)
        self.declare_parameter('dxl.profile_velocity', 0)
        self.declare_parameter('dxl.current_limit_mA', 1345)
        self.declare_parameter('dxl.goal_current_mA', 200)
        self.declare_parameter('gripper.current_stop_enabled', True)
        self.declare_parameter('gripper.close_current_stop_mA', 400)
        self.declare_parameter('gripper.close_increases_tick', True)
        self.declare_parameter('gripper.present_current_mA_topic', '/gripper/present_current_mA')
        self.declare_parameter('gripper.present_position_topic', '/gripper/present_position')
        self.declare_parameter('gripper.command_topic', '/gripper/command')
        self.declare_parameter('joystick.enabled', True)
        self.declare_parameter('joystick.command_topic', '/ur10skku/joy_move')
        self.declare_parameter('joystick.axis_index', 5)
        
        # Get parameters
        PORT = self.get_parameter('dxl.port').get_parameter_value().string_value
        BAUD = self.get_parameter('dxl.baud').get_parameter_value().integer_value
        PROTO = 2.0
        
        GRIPPER_ID = self.get_parameter('dxl.gripper_id').get_parameter_value().integer_value
        
        # Gripper limits
        GR_MIN = self.get_parameter('gripper.min_tick').get_parameter_value().integer_value
        GR_MAX = self.get_parameter('gripper.max_tick').get_parameter_value().integer_value
        
        # Movement tuning
        CMD_RATE_HZ = self.get_parameter('dxl.cmd_rate_hz').get_parameter_value().double_value
        POS_DEADBAND = self.get_parameter('dxl.pos_deadband').get_parameter_value().integer_value
        POS_SLEW_PER_SEC = self.get_parameter('dxl.pos_slew_per_sec').get_parameter_value().double_value
        
        # Motor profile/limits
        PROFILE_ACCEL = self.get_parameter('dxl.profile_accel').get_parameter_value().integer_value
        PROFILE_VELOCITY = self.get_parameter('dxl.profile_velocity').get_parameter_value().integer_value
        CURRENT_LIMIT_mA = self.get_parameter('dxl.current_limit_mA').get_parameter_value().integer_value
        GOAL_CURRENT_mA = self.get_parameter('dxl.goal_current_mA').get_parameter_value().integer_value
        CURRENT_STOP_ENABLED = self.get_parameter('gripper.current_stop_enabled').get_parameter_value().bool_value
        CLOSE_CURRENT_STOP_mA = self.get_parameter('gripper.close_current_stop_mA').get_parameter_value().integer_value
        CLOSE_INCREASES_TICK = self.get_parameter('gripper.close_increases_tick').get_parameter_value().bool_value
        
        # Topic names
        CURR_TOPIC = self.get_parameter('gripper.present_current_mA_topic').get_parameter_value().string_value
        POS_TOPIC = self.get_parameter('gripper.present_position_topic').get_parameter_value().string_value
        CMD_TOPIC = self.get_parameter('gripper.command_topic').get_parameter_value().string_value
        JOY_ENABLED = self.get_parameter('joystick.enabled').get_parameter_value().bool_value
        JOY_TOPIC = self.get_parameter('joystick.command_topic').get_parameter_value().string_value
        JOY_AXIS_INDEX = self.get_parameter('joystick.axis_index').get_parameter_value().integer_value
        
        # Store as instance variables
        self._lk = threading.Lock()
        self._goal_position = None
        self._last_cmd = None
        self.PORT = PORT
        self.BAUD = BAUD
        self.PROTO = PROTO
        self.GRIPPER_ID = GRIPPER_ID
        self.GR_MIN = GR_MIN
        self.GR_MAX = GR_MAX
        self.CMD_RATE_HZ = CMD_RATE_HZ
        self.POS_DEADBAND = POS_DEADBAND
        self.POS_SLEW_PER_SEC = POS_SLEW_PER_SEC
        self.PROFILE_ACCEL = PROFILE_ACCEL
        self.PROFILE_VELOCITY = PROFILE_VELOCITY
        self.CURRENT_LIMIT_mA = CURRENT_LIMIT_mA
        self.GOAL_CURRENT_mA = GOAL_CURRENT_mA
        self.CURRENT_STOP_ENABLED = CURRENT_STOP_ENABLED
        self.CLOSE_CURRENT_STOP_mA = CLOSE_CURRENT_STOP_mA
        self.CLOSE_INCREASES_TICK = CLOSE_INCREASES_TICK
        self.JOY_ENABLED = JOY_ENABLED
        self.JOY_TOPIC = JOY_TOPIC
        self.JOY_AXIS_INDEX = JOY_AXIS_INDEX
        self._close_stop_latched = False
        self._close_stop_position = None
        
        # Throttle logging
        self._last_warn_time = {}
        
        # ROS pub/sub
        self.curr_pub = self.create_publisher(Float32, CURR_TOPIC, 20)
        self.pos_pub = self.create_publisher(Int32, POS_TOPIC, 20)
        self.cmd_sub = self.create_subscription(Int32, CMD_TOPIC, self._cmd_callback, 10)
        self.joy_sub = None
        if JOY_ENABLED:
            self.joy_sub = self.create_subscription(
                Float64MultiArray,
                JOY_TOPIC,
                self._joy_callback,
                10
            )
        
        # DXL init
        self.port = PortHandler(PORT)
        if not self.port.openPort():
            raise RuntimeError("openPort failed: %s" % PORT)
        if not self.port.setBaudRate(BAUD):
            raise RuntimeError("setBaudRate failed: %d" % BAUD)
        self.ph = PacketHandler(PROTO)
        
        # Safety sequence: Torque OFF -> Position Limit / Mode / Current / Profile / Goal Current -> Torque ON
        self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self._w4u(ADDR_MAX_POSITION_LIMIT, GR_MAX)
        self._w4u(ADDR_MIN_POSITION_LIMIT, GR_MIN)
        self._w1(ADDR_OPERATING_MODE, OPMODE_CURRENT_BASED_POSITION)
        self._w2u(ADDR_CURRENT_LIMIT, clamp(mA2lsb(CURRENT_LIMIT_mA), 1, 0xFFFF))
        self._w2u(ADDR_GOAL_CURRENT, clamp(mA2lsb(GOAL_CURRENT_mA), 1, 0xFFFF))
        self._w4u(ADDR_PROFILE_ACCEL, PROFILE_ACCEL)
        self._w4u(ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)
        self._w1(ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        
        # Small delay to ensure motor is ready
        time.sleep(0.1)
        
        # Read current position to initialize _last_cmd
        current_pos = self._r4u(ADDR_PRESENT_POSITION)
        if current_pos is not None:
            self._last_cmd = int(current_pos)
            self.get_logger().info(f"Initial gripper position: {self._last_cmd}")
        else:
            self._last_cmd = None
            self.get_logger().warn("Could not read initial position, will use goal position")
        
        self.get_logger().info(
            f"umi gripper sub node ON | port={PORT} baud={BAUD} id={GRIPPER_ID} | "
            f"mode=Current-based Position Control(5) | "
            f"range [{GR_MIN}..{GR_MAX}] | command topic: {CMD_TOPIC} | "
            f"joy={'on' if JOY_ENABLED else 'off'} topic={JOY_TOPIC} axis_index={JOY_AXIS_INDEX} | "
            f"current_limit={CURRENT_LIMIT_mA}mA goal_current={GOAL_CURRENT_mA}mA | "
            f"current_stop={CURRENT_STOP_ENABLED} threshold={CLOSE_CURRENT_STOP_mA}mA"
        )
        
        # Create timer for main control loop
        self.timer = self.create_timer(1.0 / CMD_RATE_HZ, self._control_loop)
    
    def _cmd_callback(self, msg: Int32):
        """Callback for gripper command topic"""
        with self._lk:
            # Clamp the received position to valid range
            goal = clamp(msg.data, self.GR_MIN, self.GR_MAX)
            self._goal_position = goal
            self.get_logger().debug(f"Received command: {msg.data} -> clamped to {goal}")

    def _joy_callback(self, msg: Float64MultiArray):
        """Map a joystick axis in [-1, 1] to the gripper tick range."""
        if self.JOY_AXIS_INDEX < 0:
            self._log_warn_throttle(2.0, f"joy invalid axis index: {self.JOY_AXIS_INDEX}")
            return

        if self.JOY_AXIS_INDEX >= len(msg.data):
            self._log_warn_throttle(
                2.0,
                f"joy axis index {self.JOY_AXIS_INDEX} out of range for len={len(msg.data)}"
            )
            return

        axis_value = msg.data[self.JOY_AXIS_INDEX]
        goal = clamp(joy_to_tick(axis_value, self.GR_MIN, self.GR_MAX), self.GR_MIN, self.GR_MAX)
        with self._lk:
            self._goal_position = goal
        self.get_logger().debug(
            f"Joystick axis[{self.JOY_AXIS_INDEX}]={axis_value:.3f} -> goal={goal}"
        )
    
    # ---------- DXL I/O ----------
    def _w1(self, addr, v):
        res, err = self.ph.write1ByteTxRx(self.port, self.GRIPPER_ID, addr, int(v) & 0xFF)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"write1 fail a={addr} r={res} e={err}")
            return False
        return True
    
    def _w2u(self, addr, v):
        v = int(v) & 0xFFFF
        res, err = self.ph.write2ByteTxRx(self.port, self.GRIPPER_ID, addr, v)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"write2u fail a={addr} r={res} e={err}")
            return False
        return True
    
    def _w4u(self, addr, v):
        v = int(v) & 0xFFFFFFFF
        res, err = self.ph.write4ByteTxRx(self.port, self.GRIPPER_ID, addr, v)
        if res != COMM_SUCCESS or err != 0:
            err_str = self._get_error_string(err) if err != 0 else "COMM_ERROR"
            self._log_warn_throttle(1.0, f"write4u fail addr={addr} val={v} res={res} err={err} ({err_str})")
            return False
        return True
    
    def _r2u(self, addr):
        v, res, err = self.ph.read2ByteTxRx(self.port, self.GRIPPER_ID, addr)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"read2 fail a={addr} r={res} e={err}")
            return None
        return v
    
    def _r4u(self, addr):
        v, res, err = self.ph.read4ByteTxRx(self.port, self.GRIPPER_ID, addr)
        if res != COMM_SUCCESS or err != 0:
            self._log_warn_throttle(1.0, f"read4 fail a={addr} r={res} e={err}")
            return None
        return v
    
    def _get_error_string(self, err_code):
        """Convert Dynamixel error code to string"""
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
        """Throttled warning logger"""
        now = time.time()
        key = msg.split()[0] if msg else "default"
        if key not in self._last_warn_time or (now - self._last_warn_time[key]) >= interval:
            self.get_logger().warn(msg)
            self._last_warn_time[key] = now

    def _is_closing_motion(self, target, reference):
        if target is None or reference is None:
            return False
        if self.CLOSE_INCREASES_TICK:
            return target > reference + self.POS_DEADBAND
        return target < reference - self.POS_DEADBAND

    def _apply_current_stop(self, cmd, present_position, present_current_mA):
        if not self.CURRENT_STOP_ENABLED:
            return cmd

        hold_position = present_position if present_position is not None else self._last_cmd
        if self._close_stop_latched:
            if self._is_closing_motion(cmd, self._close_stop_position):
                return self._close_stop_position
            self._close_stop_latched = False
            self._close_stop_position = None
            self.get_logger().info("Gripper current stop released by opening command.")
            return cmd

        if (
            present_current_mA is not None
            and abs(present_current_mA) >= self.CLOSE_CURRENT_STOP_mA
            and self._is_closing_motion(cmd, self._last_cmd)
        ):
            self._close_stop_latched = True
            self._close_stop_position = int(hold_position) if hold_position is not None else int(cmd)
            with self._lk:
                self._goal_position = self._close_stop_position
            self.get_logger().warn(
                f"Close current stop latched at pos={self._close_stop_position}, "
                f"current={present_current_mA:.1f}mA. Holding position until opening command."
            )
            return self._close_stop_position

        return cmd
    
    # ---------- Main Loop ----------
    def _control_loop(self):
        try:
            with self._lk:
                goal = self._goal_position

            cur_u = self._r2u(ADDR_PRESENT_CURRENT)
            cur_s = lsb_signed(cur_u)
            present_current_mA = float(cur_s * 2.69) if cur_s is not None else None
            pos_u = self._r4u(ADDR_PRESENT_POSITION)
            present_position = int(pos_u) if pos_u is not None else None

            # Process command if received
            if goal is not None:
                # Soft slew (rate limiting)
                if self._last_cmd is None:
                    cmd = goal
                else:
                    max_step = max(1, int(round(self.POS_SLEW_PER_SEC / self.CMD_RATE_HZ)))
                    delta = goal - self._last_cmd
                    if delta > max_step: delta = max_step
                    elif delta < -max_step: delta = -max_step
                    cmd = self._last_cmd + delta
                cmd = clamp(int(round(cmd)), self.GR_MIN, self.GR_MAX)
                cmd = self._apply_current_stop(cmd, present_position, present_current_mA)

                # Send command if exceeds deadband
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
                    hold_cmd = clamp(int(self._close_stop_position), self.GR_MIN, self.GR_MAX)
                    if self._w4u(ADDR_GOAL_POSITION, hold_cmd):
                        self._last_cmd = hold_cmd

            if present_current_mA is not None:
                msg = Float32()
                msg.data = present_current_mA
                self.curr_pub.publish(msg)

            if present_position is not None:
                pos_msg = Int32()
                pos_msg.data = present_position
                self.pos_pub.publish(pos_msg)
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
    
    def destroy_node(self):
        """Cleanup on node destruction"""
        try:
            self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        except:
            pass
        try:
            self.port.closePort()
        except:
            pass
        self.get_logger().info("umi gripper sub node: torque OFF, port closed.")
        super().destroy_node()

# ================= main =================
def main(args=None):
    rclpy.init(args=args)
    try:
        node = GripperSubNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
