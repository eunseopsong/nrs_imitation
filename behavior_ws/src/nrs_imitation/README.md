# nrs_imitation

## HDF5 recorder 실행

공통 준비:

```bash
cd /home/nrs_display/nrs_imitation/behavior_ws
source install/setup.bash
```

저장 위치 기본값:

```text
~/nrs_imitation/datasets/<obs_mode>/<YYYYMMDD_HHMM>/merged_hdf5/<node_name>_<YYYYMMDD_HHMM>.hdf5
```

### 일반 recorder

Single camera:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam
```

사용 토픽:

```text
position = /calibrated_pose
force    = /ftsensor/measured_Cvalue
cam0     = /realsense/vr/color/image_raw
```

Dual camera:

```bash
ros2 run nrs_imitation hdf5_recorder_dual_cam
```

사용 토픽:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
cam1     = /realsense/global/color/image_raw
```

### Gripper recorder

Single camera + gripper:

```bash
ros2 run nrs_imitation gripper_hdf5_recorder_single_cam
```

Dual camera + gripper:

```bash
ros2 run nrs_imitation gripper_hdf5_recorder_dual_cam
```

추가 gripper 토픽:

```text
gripper position = /gripper/present_position
gripper current  = /gripper/present_current_mA
```

### 녹화 시작/종료

수동 command:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: start_recording}"
```

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: end_recording}"
```

조이스틱 command:

```bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```
