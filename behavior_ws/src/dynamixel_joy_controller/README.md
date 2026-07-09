# dynamixel_joy_controller

ROS 2 joystick controller package for the Dynamixel-based UMI gripper.

The `f710_gripper_joy_controller` node subscribes to `sensor_msgs/Joy` and
publishes `std_msgs/Int32` motor tick targets on `/gripper/command`, which is
consumed by `umi_ros2` gripper nodes.

Default Logitech F710 mapping in XInput mode:

- A, `buttons[0]`: close, publish `max_tick`
- B, `buttons[1]`: open, publish `min_tick`
- D-pad horizontal, `axes[6]`: step target by `step_tick`

The optional RT proportional axis mode is disabled by default.

## Build

```bash
cd ~/nrs_imitation/behavior_ws
colcon build --packages-select dynamixel_joy_controller
source install/setup.bash
```

## Run With Existing Gripper Node

Start the UMI gripper node separately first:

```bash
ros2 launch umi_ros2 umi_grp.launch.py
```

Then start only `joy_node` and the F710 command mapper:

```bash
ros2 launch dynamixel_joy_controller f710_gripper_joy.launch.py
```

Verify joystick commands:

```bash
ros2 topic echo /gripper/command
```

This package only publishes `/gripper/command`; it does not start or configure
the `umi_ros2` gripper node.
