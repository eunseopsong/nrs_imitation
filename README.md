# nrs_imitation

UR10 기반 force-aware imitation learning을 위한 ROS 2 recording, dataset 변환,
Flow/ACT 학습 및 inference 저장소다. 현재 주 작업은 다음 두 task다.

- **Polishing:** robot pose/force와 RGB 영상을 이용한 접촉 작업
- **Gripper:** Polishing observation에 gripper position/current를 추가한 파지 작업

이 문서는 새 demonstration을 수집해서 학습하고 inference하는 전 과정을 실제
entrypoint와 현재 기본값 기준으로 설명한다.

## 1. 전체 흐름과 지원 범위

```text
공통 장치 준비
  ├─ UR robot state/command
  ├─ Vive tracker + VR calibration
  ├─ force/torque sensor
  ├─ RealSense camera
  └─ Gripper task만 UMI gripper
          │
          ▼
Demonstration recording (merged HDF5, image-master 30 Hz)
          │
          ▼
imitation_form 변환 (episode별 observation/action HDF5)
          │
          ▼
Flow 또는 ACT 학습 (checkpoint + dataset_stats.pkl)
          │
          ▼
ROS 2 inference + safety control + Grad-CAM
```

| Task | Single camera | Dual camera | Flow | ACT | ROS inference |
|---|---|---|---|---|---|
| Polishing | 지원 | 지원 | 지원 | 지원 | single/dual 지원 |
| Gripper | 주 사용 경로, 지원 | recorder와 legacy converter 존재 | single-camera 지원 | 전용 entrypoint 없음 | single-camera 지원 |

Gripper의 재현 가능한 기본 경로는 `single_cam`이다. Gripper dual-camera recorder는
설치되지만 전용 Grad-CAM inference launch가 없으므로 표준 end-to-end 경로로
간주하지 않는다.

### 주요 디렉터리

```text
behavior_ws/                  ROS 2 packages와 launch files
datasets/
  polishing/<obs_mode>/       Polishing recording과 imitation_form
  gripper/<obs_mode>/         Gripper recording과 imitation_form
  stage1/                     Dual-camera 2-stage workflow의 VR trajectory
checkpoints/
  flow/polishing/
  flow/gripper/
  act/polishing/
scripts/flow/                 Flow 학습 entrypoint
scripts/act/                  ACT 학습 entrypoint
source/custom/                imitation_form 변환 및 dataset utility
```

학습 구조와 모든 세부 hyperparameter의 역할은
[`scripts/README.md`](scripts/README.md)에 더 자세히 정리되어 있다. 이 문서에는
실행에 필요한 현재 기본값을 함께 기재한다.

## 2. 공통 준비

### 2.1 환경과 빌드

학습용 Python 환경이 있다면 먼저 활성화한다.

```bash
conda activate nrs_imitation
```

ROS 2 workspace를 빌드하고 source한다.

```bash
cd ~/nrs_imitation/behavior_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

새 터미널을 열 때마다 다음을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/nrs_imitation/behavior_ws/install/setup.bash
```

Python 변환/학습 명령은 저장소 root에서 실행한다.

```bash
cd ~/nrs_imitation
```

### 2.2 공통 ROS topic

| 데이터 | Demonstration 기본 topic | Robot inference 기본 topic | Type |
|---|---|---|---|
| Pose | `/calibrated_pose` | `/ur10skku/currentP` | `Float64MultiArray` |
| Force | `/ftsensor/measured_Cvalue` | `/ur10skku/currentF` | `Wrench` / `Float64MultiArray` |
| Single cam | `/realsense/vr/color/image_raw` | 동일 | `sensor_msgs/Image` |
| Robot command | 사용 환경에 따라 별도 | `/ur10skku/cmdMotion` | `Float64MultiArray` |
| Recorder command | `/vr_demo_recorder/command` | 해당 없음 | `String` |

Pose convention은 `[x, y, z, rx, ry, rz]`, force convention은 `[Fx, Fy, Fz]`다.
Tracker pose의 translation은 recorder에서 기본 `pose_xyz_scale=1000`을 적용해
meter를 millimeter로 변환한다. Robot mode recorder는 이미 millimeter인
`/ur10skku/currentP`를 사용하므로 `pose_xyz_scale=1`로 고정된다.

### 2.3 UR robot 연결

UR driver/controller는 이 저장소 외부의 장비별 bringup을 사용한다. Recording이나
calibration 전에 최소한 다음 topic이 존재해야 한다.

```bash
ros2 topic echo /ur10skku/currentP --once
ros2 topic info /ur10skku/cmdMotion
```

Inference는 실제 robot command를 publish하므로 작업 공간을 비우고 emergency stop과
safety limit가 동작하는 상태에서 실행해야 한다.

### 2.4 Vive tracker 실행

기본 tracker pose를 실행한다.

```bash
ros2 launch vive_tracker_ros2 vive_bringup.launch.py
```

기본 `tool_correction_mode=none`은 tracker 자체 pose를 `/calibrated_pose`로
publish한다. Calibration으로 계산한 tracker-to-tool transform `T_BC`를 적용해
EE/TCP pose가 필요하면 다음처럼 직접 실행한다.

```bash
ros2 run vive_tracker_ros2 vive_tracker_node --ros-args \
  -p tool_correction_mode:=t_bc
```

확인할 topic:

```bash
ros2 topic echo /raw_pose --once
ros2 topic echo /calibrated_pose --once
ros2 topic hz /calibrated_pose
```

### 2.5 VR calibration

`vr_calibration`은 동일 waypoint에서 UR EE pose와 Vive raw pose를 수집해
`vive_tracker_ros2/yaml/calibration_matrix.yaml`을 갱신한다. Robot, Vive tracker,
`/ur10skku/currentP`, `/raw_pose`, `/calibrated_pose`가 모두 실행 중이어야 한다.

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run vr_calibration vr_calibration
```

노드가 표시하는 target waypoint로 robot을 이동한 뒤 정지 상태를 유지한다. 각
waypoint에서 clean sample window가 충족되면 자동 capture하고 다음 target으로
넘어간다.

주요 기본값:

| Parameter | Default | 의미 |
|---|---:|---|
| `t_sa_mode` | `update` | `/calibrated_pose` 기준 orientation correction 갱신 |
| `t_sa_max_delta_deg` | `180.0` | T_SA 갱신 허용 회전량 |
| `radj_enable` | `false` | position-cloud 기반 추가 R_Adj는 기본 비활성 |
| `capture_hold_time_s` | `2.0` | target에서 요구하는 hold 시간 |
| `capture_min_hold_time_s` | `1.5` | clean capture 최소 hold |
| `capture_window_s` | `0.5` | 평균을 계산할 안정 구간 |
| `capture_min_clean_samples` | `20` | capture에 필요한 최소 clean sample |
| `vr_capture_age_s` | `0.2` | VR sample freshness |
| `max_capture_sync_dt_s` | `0.05` | Robot/VR 최대 시간 차 |
| `capture_max_vr_std_mm` | `10.0` | 안정 구간의 VR position 표준편차 제한 |
| `z_fix_enable` | `true` | z-plane rigid correction |
| `z_residual_enable` | `true` | XY 위치별 잔여 z 오차 보정 |
| `max_calib_position_rms_mm` | `50.0` | calibration validation RMS 제한 |

정상 완료 시 `[CALIB_VALIDATE]`, `[T_SA_DONE]`, `[YAML_SAVED]` 로그를 확인한다.
세부 계산 과정은
[`behavior_ws/src/vr_calibration/README.md`](behavior_ws/src/vr_calibration/README.md)에
정리되어 있다.

### 2.6 Force/torque sensor

VR demonstration용 F/T node:

```bash
ros2 launch nrs_ft_aq2 nrsvr_ft_aq.launch.py
```

기본 config는 sensor acquisition과 publish를 모두 500 Hz로 설정하고
`/ftsensor/measured_Cvalue`를 제공한다.

```bash
ros2 topic echo /ftsensor/measured_Cvalue --once
ros2 topic hz /ftsensor/measured_Cvalue
```

Recorder는 force 500 Hz를 그대로 행으로 저장하지 않는다. Cam0 image timestamp를
master로 사용해 pose/force를 interpolation하고 최종 dataset row를 30 Hz로 만든다.

### 2.7 RealSense camera

Single-camera 기본 topic은 `/realsense/vr/color/image_raw`다. 설치된
`realsense2_camera` package를 사용하는 예시는 다음과 같다.

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=realsense \
  camera_name:=vr \
  enable_color:=true \
  rgb_camera.color_profile:=640,480,30
```

여러 장치가 연결돼 있으면 먼저 serial을 확인한다.

```bash
python3 source/custom/check_cam_serial.py
```

Dual-camera에서는 서로 다른 serial을 지정해 다음 topic을 만들어야 한다.

```text
cam0 = /realsense/robot/color/image_raw
cam1 = /realsense/global/color/image_raw
```

예시:

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=realsense camera_name:=robot serial_no:=<CAM0_SERIAL>

ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=realsense camera_name:=global serial_no:=<CAM1_SERIAL>
```

확인:

```bash
ros2 topic hz /realsense/vr/color/image_raw
```

### 2.8 UMI gripper — Gripper task만 필요

Gripper driver:

```bash
ros2 launch umi_ros2 umi_grp.launch.py
```

현재 config의 주요 기본값:

| Parameter | Default |
|---|---:|
| Serial port | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7D6-if00-port0` |
| Baud | `57600` |
| Gripper command range | `-653 .. 733 tick` |
| Command rate | `30 Hz` |
| Position slew | `1000 tick/s` |
| Default goal current | `500 mA` |
| Goal current range | `0 .. 1345 mA` |
| Close-current stop | `400 mA`, debounce `0.12 s` |

필수 topic:

```text
/gripper/command              std_msgs/Int32
/gripper/goal_current_mA      std_msgs/Float32
/gripper/present_position     std_msgs/Int32
/gripper/present_current_mA   std_msgs/Float32
```

확인:

```bash
ros2 topic hz /gripper/present_position
ros2 topic hz /gripper/present_current_mA
```

별도 `dynamixel_joy_controller` package도 설치되지만 현재 UMI demonstration
command convention과는 다르다.

```bash
ros2 launch dynamixel_joy_controller f710_gripper_joy.launch.py
```

이 controller는 기본 range `590..2500`과 `open=min, close=max` convention을
사용한다. 현재 UMI demonstration은 `close=-653, open=733`이므로 단순히 range만
override하면 open/close가 뒤집힌다. 현재 UMI task에는 이 launch 대신 range,
방향, recorder start/end가 모두 맞는 `vr_demo_joy_controller.launch.py`를
사용한다.

## 3. Demonstration recording

### 3.1 공통 recording 동작

Polishing과 Gripper recorder는 공통적으로 다음을 수행한다.

1. 각 sensor callback을 수신 시각과 함께 buffer에 저장한다.
2. Cam0를 master timestamp로 선택한다.
3. Pose/force는 image 시각으로 linear interpolation한다.
4. Gripper position/current는 image 시각의 nearest sample을 선택한다.
5. Sync error가 제한 안에 있는 새 image만 30 Hz dataset row로 기록한다.
6. Episode 종료 후 trajectory filtering을 적용하고 merged HDF5에 저장한다.

주요 기본값:

| Parameter | Default | 기능 |
|---|---:|---|
| `sample_hz` | `30.0` | 최종 recording row rate |
| `sync_enable` | `true` | image-master synchronization 사용 |
| `sync_buffer_sec` | `1.0` | interpolation/nearest 검색 buffer |
| `sync_max_error_sec` | `0.05` | 각 stream과 image의 최대 허용 시간 차 |
| `sync_require_new_image` | `true` | 동일 image 중복 기록 방지 |
| `require_pose_fresh_sec` | `0.20` | pose freshness |
| `require_force_fresh_sec` | `0.20` | force freshness |
| `require_image_fresh_sec` | `0.50` | Cam0 freshness |
| `num_episodes` | `50` | 한 recorder file의 최대 episode 수 |
| `min_samples` | `10` | 저장할 episode 최소 길이 |
| `force_filter_mode` | `ema` | 기본 force filtering |
| `force_ema_alpha` | `0.2` | force EMA 계수 |
| `image_compression` | `gzip` | RGB HDF5 compression |
| `image_gzip_level` | `4` | gzip level |
| `image_preprocess_mode` | `raw` | RGB preprocessing |

Recorder command joystick:

```bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

기본 F710/XInput mapping:

```text
RB            start_recording
LB            end_recording
A             gripper close, -653 tick
B             gripper open, 733 tick
D-pad left    close by 50 tick
D-pad right   open by 50 tick
```

Joystick 없이 topic으로 제어할 수도 있다.

```bash
ros2 topic pub --once /vr_demo_recorder/command \
  std_msgs/msg/String "{data: start_recording}"

ros2 topic pub --once /vr_demo_recorder/command \
  std_msgs/msg/String "{data: end_recording}"
```

한 episode마다 `start_recording → demonstration 수행 → end_recording`을 반복한다.
Recorder terminal은 전체 session 동안 계속 실행한다.

### 3.2 Polishing — single camera

필요 topic:

```text
pose    /calibrated_pose
force   /ftsensor/measured_Cvalue
cam0    /realsense/vr/color/image_raw
```

일반 recorder:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam
```

Stain mask 학습 데이터를 만들 때:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam_stain_mask
```

Stain-mask recorder를 사용하면 `ep_0000`을 깨끗한 표면 reference로 먼저
recording한다. 카메라가 움직이는 demonstration이면 clean reference도 실제
demonstration과 같은 pose sweep을 포함해야 한다.

Specular highlight를 약화한 RGB를 저장하려면:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam --ros-args \
  -p image_preprocess_mode:=highlight_attenuate \
  -p image_specular_mask_mode:=bright \
  -p image_specular_v_thresh:=220 \
  -p image_specular_dilate_px:=2 \
  -p image_specular_attenuate_gain:=0.35
```

출력:

```text
datasets/polishing/single_cam/<YYYYMMDD_HHMM>/
└── merged_hdf5/hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5
```

### 3.3 Gripper — single camera

Polishing single-camera topic에 gripper state 두 개가 추가된다.

```text
gripper position   /gripper/present_position
gripper current    /gripper/present_current_mA
```

Recorder:

```bash
ros2 run nrs_imitation gripper_hdf5_recorder_single_cam
```

Gripper driver와 `vr_demo_joy_controller`를 함께 실행하면 episode command와 gripper
개폐를 같은 F710으로 수행할 수 있다.

출력:

```text
datasets/gripper/single_cam/<YYYYMMDD_HHMM>/
└── merged_hdf5/gripper_hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5
```

### 3.4 Polishing — dual camera / Stage-1 workflow

Dual-camera는 VR trajectory와 robot playback recording을 분리한다.

```text
Stage 1: Vive demonstration trajectory recording
    ↓
Stage 1 episode를 robot playback PC로 push
    ↓
Stage 2: robot playback 중 cam0 + cam1 + robot pose/force recording
```

Stage-1 recorder:

```bash
ros2 run nrs_imitation vr_stage1_hdf5_recorder
```

Episode start/end command는 single-camera와 동일하다. 출력:

```text
datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes/episode_*.hdf5
```

최신 Stage-1 directory를 push:

```bash
ros2 run nrs_imitation vr_stage1_episode_pusher
```

특정 directory:

```bash
ros2 run nrs_imitation vr_stage1_episode_pusher --ros-args \
  -p episode_dir:=/home/eunseop/nrs_imitation/datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes
```

Robot playback PC에서 필요한 topic:

```text
pose    /ur10skku/currentP
force   /ur10skku/currentF
cam0    /realsense/robot/color/image_raw
cam1    /realsense/global/color/image_raw
```

Playback 중 recorder:

```bash
ros2 run nrs_imitation hdf5_recorder_dual_cam
```

각 playback episode 시작 직전에 `start_recording`, 종료 직후 `end_recording`을
보낸다.

출력:

```text
datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/
└── merged_hdf5/hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5
```

### 3.5 Gripper — dual camera recorder

Gripper dual-camera recording이 필요한 실험에서는 다음 node를 사용할 수 있다.

```bash
ros2 run nrs_imitation gripper_hdf5_recorder_dual_cam
```

입력은 Polishing dual-camera topic에 gripper position/current가 추가된 형태다.
다만 현재 표준 학습/inference 경로는 Gripper single-camera이므로 새 실험에서는
converter와 checkpoint schema를 먼저 검증해야 한다.

## 4. imitation_form 변환

### 4.1 변환의 역할과 공통 기본값

Merged HDF5 안의 여러 episode를 학습용 `episode_*.hdf5`로 분리한다. 각 row의
observation과 action은 같은 demonstration 시점에 정렬되어 있고, 학습 Dataset이
현재 observation `t`와 미래 action sequence `t:t+chunk_size`를 구성한다.

| Parameter | Default | 기능 |
|---|---:|---|
| `input_h5` | 빈 값 | 생략하면 task/obs mode 아래 최신 merged HDF5 자동 선택 |
| `output_dir` | 빈 값 | 생략하면 같은 run의 `imitation_form/` |
| `min_len` | `10` | 이보다 짧은 episode 제외 |
| `max_len` | `0` | 0이면 truncation 없음 |
| `compression` | `gzip` | 출력 HDF5 compression |
| `gzip_level` | `4` | gzip level |
| `overwrite` | `false` | 기존 output을 교체하지 않음 |
| `write_summary` | `false` | `conversion_summary.json` 생성 여부 |
| `stain_mask_mode` | `auto` | recorder metadata에 따라 copy/reference/none 선택 |
| `stain_reference_episode` | `ep_0000` | clean reference episode |

`--overwrite`는 기존 `imitation_form` episode를 교체하므로 경로를 확인하고
사용한다. 재현 가능한 변환 기록을 남기기 위해 `--write_summary` 사용을 권장한다.

### 4.2 Polishing — single camera

최신 recording 자동 선택:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_single_cam.py \
  --overwrite \
  --write_summary
```

특정 merged HDF5:

```bash
python3 source/custom/demo_data_imitation_form_single_cam.py \
  --input_h5 datasets/polishing/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

Stain-mask mode가 `reference_episode`이면 clean reference와 current episode를
pose-sequence DTW로 매칭한 뒤 image alignment, reference difference, dark prior,
temporal fill/prune을 적용한다. `had no close clean reference`가 반복되면 threshold를
먼저 완화하기보다 동일 경로의 clean reference를 다시 recording한다.

출력:

```text
datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form/
├── episode_0.hdf5
├── episode_1.hdf5
└── conversion_summary.json
```

### 4.3 Gripper — single camera

최신 recording 자동 선택:

```bash
cd ~/nrs_imitation
python3 source/custom/gripper_data_imitation_form_single_cam.py \
  --overwrite \
  --write_summary
```

특정 merged HDF5:

```bash
python3 source/custom/gripper_data_imitation_form_single_cam.py \
  --input_h5 datasets/gripper/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/gripper_hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/gripper/single_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

`gripper_goal_current_mA` action은 recorded `present_current_mA`의 magnitude로
생성한다. Signed present current도 분석/호환 목적으로 보존한다.

### 4.4 Polishing — dual camera

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_dual_cam.py \
  --overwrite \
  --write_summary
```

특정 merged HDF5:

```bash
python3 source/custom/demo_data_imitation_form_dual_cam.py \
  --input_h5 datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

### 4.5 Gripper — dual camera legacy 변환

Dual-camera gripper merged HDF5는 legacy multimodal converter를 사용한다. 표준
Gripper single-camera converter와 출력 metadata가 다르므로 실험용 경로로
취급한다.

```bash
python3 source/custom/gripper_data_imitation_form.py \
  --input_h5 datasets/gripper/dual_cam/<YYYYMMDD_HHMM>/merged_hdf5/gripper_hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/gripper/dual_cam/<YYYYMMDD_HHMM>/imitation_form \
  --require_cam1 \
  --overwrite
```

학습 전 `cam0`, `cam1`, gripper position/current와 11D action schema를 직접
확인한다. 이 경로에는 전용 inference launch가 없다.

### 4.6 imitation_form schema

Polishing:

```text
observations/position             (T, 6)
observations/force                (T, 3)
observations/images/cam0          (T, H, W, 3)
observations/images/cam1          (T, H, W, 3), dual only
observations/images/stain_mask    (T, H, W), optional
action/position                   (T, 6)
action/force                      (T, 3)
```

Gripper는 위 single-camera schema에 다음 항목을 추가한다.

```text
observations/gripper/present_position
observations/gripper/present_current_mA
action/gripper_present_position
action/gripper_goal_current_mA
action/gripper_present_current_mA
```

## 5. Learning

### 5.1 모델이 학습하는 것

Flow 학습 sample:

```text
현재 pose + force ─────────────── State MLP ──────────┐
최근 force sequence ───────────── Force GRU ──────────┤
RGB image(s) ──────────────────── ResNet18 ───────────┤
Polishing: optional stain mask ── masked pooling ─────┤
Gripper: 현재 position/current ── scalar MLPs ────────┤
Gripper: 최근 position/current ── Joint GRU ──────────┘
                                                     │
                                                     ▼
                                           fused condition
                                                     │
noise action sequence + Flow time ── Conditional 1D U-Net
                                                     │
                                                     ▼
                                         미래 action sequence
```

Polishing action은 `pose 6 + force 3 = 9D`다. Gripper action은 여기에
`gripper position + goal current = 2D`가 추가되어 11D다.

학습 split의 min/max를 `dataset_stats.pkl`에 저장하고 pose, force, gripper
position/current, force/gripper history와 action을 기본 `[-1, 1]`로 정규화한다.
Inference는 반드시 해당 checkpoint와 같은 stats/schema를 사용해야 한다.

### 5.2 공통 Flow 기본값

| Parameter | Default | 기능 |
|---|---:|---|
| `norm_mode` | `minmax_m11` | 수치 observation/action을 `[-1,1]` 정규화 |
| `dataset_hz` | `30.0` | 동기화된 dataset row rate |
| `state_dim` | `9` | pose 6 + 현재 force 3 |
| `use_force_history` | `true` | 최근 force sequence GRU 사용 |
| `force_history_sec` | `1.0` | force history 시간 범위 |
| `force_history_len` | `30` | 30 Hz 기준 1초 |
| `force_encoder_hidden_dim` | `64` | Force GRU feature 크기 |
| `force_encoder_num_layers` | `1` | Force GRU layer |
| `force_encoder_dropout` | `0.0` | 1 layer에서는 적용되지 않음 |
| `samples_per_episode` | `50` | epoch마다 episode별 시작점 sample 수 |
| `resample_each_epoch` | `true` | train 시작점을 epoch마다 재선택 |
| `num_epochs` | `500` | 최대 epoch |
| `lr` | `1e-4` | AdamW 기준 learning rate |
| `weight_decay` | `1e-5` | AdamW regularization |
| `lr_scheduler` | `cosine` | warmup 이후 cosine decay |
| `warmup_epochs` | `10` | learning-rate warmup |
| `min_lr` | `1e-6` | cosine scheduler 최저 LR |
| `grad_clip_norm` | `1.0` | gradient clipping |
| `early_stopping_patience` | `0` | 0이면 early stopping 비활성 |
| `num_workers` | `2` | HDF5/image DataLoader worker |
| `pin_memory` | `true` | CUDA 전송용 pinned memory |
| `persistent_workers` | `true` | epoch 사이 worker 유지 |
| `prefetch_factor` | `2` | worker당 미리 준비할 batch 수 |
| `save_every` | `50` | 중간 checkpoint 주기 |
| `flow_infer_steps` | `10` | Flow ODE integration step |
| Image backbone | pretrained ResNet18 | `--no_pretrained`로 비활성 |

`force_history_sec`와 `chunk_sec`가 양수이면 실제 step 수는
`round(dataset_hz × seconds)`를 기반으로 계산된다. U-Net action horizon은
down/up sampling을 위해 4의 배수로 맞춘다.

### 5.3 Task별 Flow 기본값

| Parameter | Polishing | Gripper |
|---|---:|---:|
| `batch_size` | `12` | `8` |
| `action_dim` | `9` | `11` |
| `chunk_size` | `128` | `160` |
| `chunk_sec` | `4.27` | `5.33` |
| `use_stain_mask` | `true` | 해당 없음 |
| `use_gripper_history` | 해당 없음 | `true` |
| `gripper_history_sec` | 해당 없음 | `0.5` |
| `gripper_history_len` | 해당 없음 | `15` |
| `gripper_history_hidden_dim` | 해당 없음 | `32` |
| `gripper_history_num_layers` | 해당 없음 | `1` |
| `gripper_history_dropout` | 해당 없음 | `0.0` |

Gripper의 현재 position/current는 각각 MLP로 encoding한다. 최근 0.5초의 causal
`[position, current]` sequence는 Joint GRU가 encoding하고, 두 branch를 fusion해
gripper feature를 만든다. Episode 시작처럼 history가 부족한 구간은 가장 오래된
값으로 left padding한다.

### 5.4 Polishing — Flow 학습

Single camera, 최신 imitation_form 자동 선택:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_single_cam.py
```

특정 dataset:

```bash
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

기본은 stain-mask feature를 사용한다. RGB-only baseline:

```bash
python3 scripts/flow/train_flow_single_cam.py --no_stain_mask
```

Dual camera:

```bash
python3 scripts/flow/train_flow_dual_cam.py
```

특정 dual dataset:

```bash
python3 scripts/flow/train_flow_dual_cam.py \
  --dataset_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form
```

출력:

```text
checkpoints/flow/polishing/<single_cam|dual_cam>/<YYYYMMDD_HHMM>/
├── policy_best.ckpt
├── policy_last.ckpt
└── dataset_stats.pkl
```

### 5.5 Polishing — ACT 학습

ACT는 Polishing single/dual-camera baseline을 제공한다.

```bash
cd ~/nrs_imitation
python3 scripts/act/train_act_single_cam.py
python3 scripts/act/train_act_dual_cam.py
```

ACT도 stain mask를 기본 사용한다. RGB-only baseline은 `--no_stain_mask`를 추가한다.

주요 ACT 기본값:

| Parameter | Default |
|---|---:|
| `batch_size` | `12` |
| `action_dim` | `9` |
| `chunk_size` | `200` |
| `num_epochs` | `500` |
| `lr` | `1e-4` |
| `weight_decay` | `1e-6` |
| `kl_weight` | `10` |
| `hidden_dim` | `512` |
| `nheads` | `8` |
| `enc_layers` | `4` |
| `dec_layers` | `7` |
| `use_force_history` | `true` |
| `force_history_len` | `10` |
| `use_stain_mask` | `true` |

Gripper 전용 ACT wrapper는 현재 없다.

### 5.6 Gripper — Flow 학습

최신 gripper imitation_form 자동 선택:

```bash
cd ~/nrs_imitation/scripts/flow
python3 train_flow_gripper_single_cam.py
```

특정 dataset:

```bash
python3 train_flow_gripper_single_cam.py \
  --dataset_dir /home/eunseop/nrs_imitation/datasets/gripper/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

History 없는 MLP-only ablation:

```bash
python3 train_flow_gripper_single_cam.py --no_gripper_history
```

출력:

```text
checkpoints/flow/gripper/single_cam/<YYYYMMDD_HHMM>/
├── policy_best.ckpt
├── policy_last.ckpt
└── dataset_stats.pkl
```

Dual-camera gripper 실험은 generic entrypoint로 실행할 수 있다.

```bash
cd ~/nrs_imitation/scripts/flow
python3 train_flow_gripper.py \
  --obs_mode dual_cam \
  --camera_names cam0 cam1 \
  --dataset_dir /home/eunseop/nrs_imitation/datasets/gripper/dual_cam/<YYYYMMDD_HHMM>/imitation_form
```

이 명령은 모델 학습 경로만 제공하며 현재 대응하는 dual-camera gripper ROS
inference launch는 없다.

### 5.7 학습 완료 및 checkpoint load 확인

Flow 학습 로그에서 다음을 확인한다.

```text
[INFO] Best epoch
[INFO] Best val loss
[INFO] Best ckpt path
[INFO] Last ckpt path
```

최신 Polishing checkpoint load 확인:

```bash
cd ~/nrs_imitation/scripts/flow
python3 train_flow_single_cam.py --eval
```

최신 Gripper checkpoint load 확인:

```bash
cd ~/nrs_imitation/scripts/flow
python3 train_flow_gripper_single_cam.py --eval
```

Gripper eval은 model과 checkpoint를 `strict=True`로 load한다.
`missing=0, unexpected=0`이어야 하며 gripper-history 사용 여부와 길이도 checkpoint
metadata와 일치해야 한다.

## 6. Inference

Inference는 실제 robot을 움직인다. 처음에는 robot speed를 낮추고 emergency stop이
가능한 상태에서 workspace와 command topic을 확인한다.

### 6.1 Polishing — single camera

최신 Flow checkpoint:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
```

특정 checkpoint:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  ckpt_dir:=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>
```

기본 Flow 학습은 stain mask를 사용하지만 inference launch의 mask는 안전하게
비활성화되어 있다. Mask checkpoint를 실행할 때:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  use_stain_mask:=true \
  auto_stain_mask:=true
```

RGB-only checkpoint는 `use_stain_mask:=false`를 유지한다.

ACT:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  policy_class:=ACT
```

### 6.2 Gripper — single camera

최신 checkpoint:

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py
```

특정 checkpoint:

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py \
  ckpt_dir:=/home/eunseop/nrs_imitation/checkpoints/flow/gripper/single_cam/<YYYYMMDD_HHMM>
```

현재 history checkpoint의 주요 launch 기본값:

| Parameter | Default |
|---|---:|
| `control_hz` | `125.0` |
| `infer_hz` | `10.0` |
| `use_force_history` | `true` |
| `force_history_len` | `30` |
| `use_gripper_history` | `true` |
| `gripper_history_len` | `15` |
| `gripper_history_hz` | `30.0` |
| `gripper_history_sync_slop_sec` | `0.020` |
| `gripper_history_max_age_sec` | `0.20` |
| `use_temporal_agg` | `true` |
| `temporal_agg_mode` | `exp` |
| `temporal_agg_tau_steps` | `20.0` |
| `max_plans` | `6` |
| `gradcam_enable` | `true` |

따라서 최신 기본값을 사용할 때는 `ckpt_dir` 외 파라미터를 반복해서 적을 필요가
없다. Checkpoint가 MLP-only로 학습됐다면 명시적으로 history를 끈다.

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py \
  ckpt_dir:=/home/eunseop/nrs_imitation/checkpoints/flow/gripper/single_cam/<OLD_CHECKPOINT> \
  use_gripper_history:=false
```

Gripper output:

```text
action[0:9]   -> /ur10skku/cmdMotion
action[9]     -> /gripper/command
action[10]    -> /gripper/goal_current_mA
```

Gripper safety 기본값:

| Parameter | Default |
|---|---:|
| `gripper_command_min_tick` | `-653` |
| `gripper_command_max_tick` | `733` |
| `gripper_command_deadband_tick` | `2` |
| `gripper_command_slew_per_sec` | `1000` |
| `gripper_command_step_cap_tick` | `200` |
| `gripper_goal_current_min_mA` | `0` |
| `gripper_goal_current_max_mA` | `1345` |
| `gripper_goal_current_deadband_mA` | `5` |
| `gripper_cmd_safety_max_tick_from_present` | `1500` |
| `tau_sec` | `0.8` |
| `startup_ramp_sec` | `3.0` |

### 6.3 Polishing — dual camera

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
```

특정 checkpoint:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py \
  ckpt_dir:=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/dual_cam/<YYYYMMDD_HHMM>
```

ACT:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py \
  policy_class:=ACT
```

### 6.4 Grad-CAM

Grad-CAM과 `rqt_image_view`는 inference launch에서 기본 활성화된다.

```text
Polishing single   /inference_single_cam/gradcam_overlay
Polishing dual     /inference_dual_cam/gradcam_overlay
Polishing global   /inference_dual_cam/gradcam_overlay_global
Gripper single     /inference_gripper_single_cam/gradcam_overlay
```

비활성화:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  gradcam_enable:=false \
  visualize:=false
```

## 7. 시간축과 동기화 주의사항

- Sensor publish rate는 서로 달라도 된다. Recorder의 최종 dataset timebase는
  Cam0 기준 30 Hz다.
- Pose/force는 image timestamp에 interpolation하고, gripper position/current는
  nearest sample을 사용한다.
- `dataset_hz=30`은 ROS control Hz가 아니라 학습 HDF5 row rate다.
- `control_hz=125`는 UR command loop 주기이고 `infer_hz=10`은 새 policy plan 생성
  주기다.
- `flow_infer_steps=10`은 noise를 action으로 적분하는 횟수이며 ROS Hz가 아니다.
- `action_hz`와 `force_history_hz`는 현재 inference launch parameter가 아니다.
- Gripper history는 runtime에서 position/current pair를 approximate synchronization해
  30 Hz schema와 길이를 검증한다.
- Online force history는 현재 force callback을 buffer에 쌓는다. Force topic이
  500 Hz라면 `force_history_len=30`의 실제 시간 폭이 training의 1초와 다를 수
  있으므로 추후 time-based resampling이 필요한 알려진 제한이다.

## 8. 데이터와 실행 상태 점검

필수 topic 목록:

```bash
ros2 topic list
```

주요 publish rate:

```bash
ros2 topic hz /calibrated_pose
ros2 topic hz /ftsensor/measured_Cvalue
ros2 topic hz /realsense/vr/color/image_raw
ros2 topic hz /gripper/present_position
ros2 topic hz /gripper/present_current_mA
```

HDF5 RGB jitter 시각화:

```bash
python3 source/custom/visualize_hdf5_rgb_jitter.py \
  --input_h5 <HDF5_PATH> \
  --camera_name cam0
```

Camera serial 확인:

```bash
python3 source/custom/check_cam_serial.py
```

Gripper tick range 변환 utility:

```bash
python3 source/custom/convert_gripper_tick_range.py --help
```

모든 변환/학습 옵션:

```bash
python3 source/custom/demo_data_imitation_form_single_cam.py --help
python3 source/custom/gripper_data_imitation_form_single_cam.py --help
python3 scripts/flow/train_flow_single_cam.py --help
python3 scripts/flow/train_flow_gripper_single_cam.py --help
python3 scripts/act/train_act_single_cam.py --help
```

모든 inference launch argument:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py --show-args
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py --show-args
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py --show-args
```

## 9. 설치된 주요 ROS 2 entrypoint

Recording:

```text
hdf5_recorder_single_cam
hdf5_recorder_single_cam_stain_mask
hdf5_recorder_dual_cam
gripper_hdf5_recorder_single_cam
gripper_hdf5_recorder_dual_cam
vr_stage1_hdf5_recorder
vr_stage1_episode_pusher
vr_demo_txt_recorder
gripper_demo_txt_recorder
```

Inference/debug:

```text
inference_single_cam
inference_dual_cam
inference_gripper_single_cam
stain_mask_publisher
```

Launch files:

```text
vr_demo_joy_controller.launch.py
inference_gradcam_single_cam.launch.py
inference_gradcam_gripper_single_cam.launch.py
inference_gradcam_dual_cam.launch.py
```

## 10. 최소 실행 체크리스트

### Polishing single-camera

```text
[ ] ROS workspace build/source
[ ] UR robot state 또는 teaching pose source 확인
[ ] Vive tracker와 VR calibration 확인
[ ] F/T sensor와 Cam0 실행
[ ] hdf5_recorder_single_cam(_stain_mask) 실행
[ ] episode start/demonstration/end 반복
[ ] demo_data_imitation_form_single_cam.py 실행
[ ] train_flow_single_cam.py 실행
[ ] --eval로 checkpoint load 확인
[ ] inference 전 robot safety와 topic 확인
```

### Gripper single-camera

```text
[ ] Polishing 공통 장치 준비
[ ] umi_grp.launch.py 실행
[ ] gripper position/current가 약 30 Hz인지 확인
[ ] gripper_hdf5_recorder_single_cam 실행
[ ] episode start/gripper demonstration/end 반복
[ ] gripper_data_imitation_form_single_cam.py 실행
[ ] train_flow_gripper_single_cam.py 실행
[ ] --eval에서 strict=True, missing=0, unexpected=0 확인
[ ] inference checkpoint의 force/gripper history schema 확인
[ ] inference 전 robot/gripper safety와 topic 확인
```
