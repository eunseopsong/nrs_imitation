# nrs_imitation

## vr_demo_hdf5_recorder 실행

공통 준비:

```bash
cd /home/nrs_display/nrs_imitation/behavior_ws
source install/setup.bash
```

저장 위치 기본값:

```text
~/ACT/<YYYYMMDD_HHMM>/merged_hdf5/vr_demo_merged_<YYYYMMDD_HHMM>.hdf5
```

### Single-cam: VR tracker 기준

VR pose, FT sensor, VR camera 1대만 저장한다.

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args \
  -p recording_mode:=tracker \
  -p enable_global_cam:=false
```

사용 토픽:

```text
pose  = /calibrated_pose
force = /ftsensor/measured_Cvalue
cam0  = /realsense/vr/color/image_raw
cam1  = disabled
```

### Single-cam: robot 기준

Robot pose, robot force, robot camera 1대만 저장한다.

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args \
  -p recording_mode:=robot \
  -p enable_global_cam:=false
```

사용 토픽:

```text
pose  = /ur10skku/currentP
force = /ur10skku/currentF
cam0  = /realsense/robot/color/image_raw
cam1  = disabled
```

### Dual-cam: VR tracker + global cam

VR tracker 기준 데이터와 global camera를 함께 저장한다.

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args \
  -p recording_mode:=tracker \
  -p enable_global_cam:=true
```

사용 토픽:

```text
pose  = /calibrated_pose
force = /ftsensor/measured_Cvalue
cam0  = /realsense/vr/color/image_raw
cam1  = /realsense/global/color/image_raw
```

### Dual-cam: robot + global cam

Robot 기준 데이터와 global camera를 함께 저장한다.

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args \
  -p recording_mode:=robot \
  -p enable_global_cam:=true
```

사용 토픽:

```text
pose  = /ur10skku/currentP
force = /ur10skku/currentF
cam0  = /realsense/robot/color/image_raw
cam1  = /realsense/global/color/image_raw
```

### 카메라 토픽 직접 지정

기본 토픽과 다른 카메라를 쓰려면 `image_topic` 또는 `global_image_topic`을 override한다.

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args \
  -p recording_mode:=tracker \
  -p enable_global_cam:=false \
  -p image_topic:=/your/camera/color/image_raw
```

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args \
  -p recording_mode:=tracker \
  -p enable_global_cam:=true \
  -p image_topic:=/your/cam0/color/image_raw \
  -p global_image_topic:=/your/cam1/color/image_raw
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

기본 버튼:

```text
button_start = 4
button_end   = 5
```
