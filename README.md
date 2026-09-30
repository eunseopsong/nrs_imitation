# nrs_imitation 통합 README

`scripts/README.md`를 제외한 README 원문을 원래 위치별로 통합한 문서입니다.
각 절의 내용은 원문 그대로 보존했습니다. 상대경로 링크와 실행 위치는 해당 원본 문서의 위치 및 안내를 기준으로 해석하세요.
별도 문서: [scripts/README.md](scripts/README.md)

## 원본 문서 목차

- [README.md](#readme-source-01)
- [.pytest_cache/README.md](#readme-source-02)
- [behavior_ws/src/dynamixel_joy_controller/README.md](#readme-source-03)
- [behavior_ws/src/nrs_ft_aq2/README.md](#readme-source-04)
- [behavior_ws/src/nrs_imitation/.pytest_cache/README.md](#readme-source-05)
- [behavior_ws/src/nrs_imitation/README.md](#readme-source-06)
- [behavior_ws/src/stain_relative_frame/.pytest_cache/README.md](#readme-source-07)
- [behavior_ws/src/stain_relative_frame/README.md](#readme-source-08)
- [behavior_ws/src/umi_ros2/README.md](#readme-source-09)
- [behavior_ws/src/vr_calibration/README.md](#readme-source-10)
- [datasets/polishing/single_cam/20260909_90deg_only_rel_fzobs/imitation_form/README.md](#readme-source-11)
- [experiments/e1_force_observation_20260916/README.md](#readme-source-12)
- [experiments/e1_inference_logging_20260917/README.md](#readme-source-13)
- [experiments/e1_rt_common_20260927/README.md](#readme-source-14)
- [reports/20260926_C5_log_audit/README.md](#readme-source-15)
- [reports/20260926_C_execution_patch/README.md](#readme-source-16)
- [reports/20260926_C_execution_patch/before/experiments/rtc_launch_20260926/README.md](#readme-source-17)
- [reports/20260926_C_vibration/README.md](#readme-source-18)
- [reports/20260926_R5_log_audit/README.md](#readme-source-19)
- [reports/20260926_T5_log_audit/README.md](#readme-source-20)
- [reports/20260926_e1_rtc_audit/README.md](#readme-source-21)
- [reports/20260926_rtc_disconnect_diagnosis/README.md](#readme-source-22)
- [reports/20260927_E1_RT_common_patch/README.md](#readme-source-23)
- [reports/20260927_E1_RT_common_patch/before/experiments/rtc_launch_20260926/README.md](#readme-source-24)
- [reports/20260927_E2_A_preparation/README.md](#readme-source-25)
- [reports/20260927_E2_A_protection_patch/install/Y2RobMotion/share/Y2RobMotion/txtcmd/README.txt](#readme-source-26)
- [reports/20260927_E2_F0_baseline_review/README.md](#readme-source-27)
- [reports/20260927_E2_F0_candidate/README.md](#readme-source-28)
- [reports/20260927_E2_F0_evidence/README.md](#readme-source-29)
- [reports/20260927_E2_F0_observation_setup/README.md](#readme-source-30)
- [reports/20260927_E2_F0_offline_assumption/README.md](#readme-source-31)
- [reports/20260928_E2_A_restore/README.md](#readme-source-32)
- [results/20260920/README.md](#readme-source-33)
- [results/20260926/E1/README.md](#readme-source-34)
- [results/20260926/E1/audit/C/README.md](#readme-source-35)
- [results/20260926/E1/audit/C_patch/README.md](#readme-source-36)
- [results/20260926/E1/audit/R/README.md](#readme-source-37)
- [results/20260926/E1/audit/T/README.md](#readme-source-38)
- [results/20260926/E2/README.md](#readme-source-39)
- [results/table/20260920/README.md](#readme-source-40)
- [results/table/20260920/ral/README.md](#readme-source-41)
- [results/table/20260920/slides/README.md](#readme-source-42)


---

<a id="readme-source-01"></a>

# 원본: `README.md`

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
Start         gripper close, -653 tick
Back          gripper open, 733 tick
RT            close by 50 tick
LT            open by 50 tick
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

FLOW/DINOv3용 RGB recorder:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam
```

기존 ACT stain-mask 실험을 재현할 때만:

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
| `stain_mask_mode` | single camera: `none` | DINOv3 FLOW는 RGB만 저장; 고정 TCP ROI는 encoder 내부 생성 |
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
| Image backbone | frozen pretrained DINOv3 ViT-S/16 | `--image_backbone resnet18`로 기존 ResNet 선택 |

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
| `use_tcp_roi` | `true` | 해당 없음 |
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

DINOv3 ViT-S/16 + TCP ROI patch-attention 학습:

```bash
python3 -m pip install 'timm>=1.0.20,<2'
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260801_2233/imitation_form \
  --image_backbone dinov3 \
  --dino_roi_pooling attention \
  --freeze_image_backbone
```

DINOv3가 기본 backbone이므로 `--image_backbone dinov3`,
`--dino_roi_pooling attention`, `--freeze_image_backbone`은 생략할 수 있다.
기존 ResNet18 baseline은 `--image_backbone resnet18`로 선택한다.

`240x424` RGB는 encoder 내부에서 오른쪽에 8 pixel을 padding한 뒤 `15x27`
DINO patch grid로 변환한다. `(253,120)` 중심의 10% 고정 TCP interaction ROI도
encoder 내부에서 생성하며 HDF5에는 저장하지 않는다. 이 ROI는 patch 경계 기준
`7x7=49` token, 전체 405 token의 약 12.1%와 겹친다. Pretrained DINO backbone은
기본적으로 freeze하고 ROI attention/projection과 FLOW policy만 학습한다.

TCP ROI를 제거하고 DINO global feature만 사용하는 baseline:

```bash
python3 scripts/flow/train_flow_single_cam.py --no_tcp_roi
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

기본 FLOW/DINOv3 checkpoint는 RGB만 입력받고 고정 TCP ROI를 encoder 내부에서
생성하므로 stain-mask topic이나 publisher가 필요 없다. ROI 좌표는 checkpoint
metadata에서 자동 복원된다.

Polishing single-camera 추론의 Grad-CAM 기본값은 `false`다.
진단 시에만 `gradcam_enable:=true`로 켠다.

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

Polishing single-camera launch는 Grad-CAM과 `rqt_image_view`를 기본으로 끈다.
`inference_clean_single_cam.launch.py`는 flow-vector와 modality 영상을 창 없이
`~/Videos/Screencasts/*.webm`에 자동 저장한다 (`overlay_record_enable:=true`).
실시간 확인이 필요한 경우에만 `visualize_flow_vector:=true` 또는
`visualize_modality_importance:=true`로 해당 창을 연다.
기록 부하를 줄이기 위해 영상과 overlay 발행은 기본 10fps
(`overlay_record_fps`), 인코더는 1 thread/낮은 CPU 우선순위를 사용한다.
`metrics_log_enable:=true`에서도 추가 2kHz FT/제어 상태 토픽 구독은 기본 OFF
(`metrics_extra_telemetry_enable:=false`)이며, 기존 위치·힘 입력과 비교 CSV는
최대 20Hz로 기록한다 (`metrics_sample_hz:=20.0`, `0.0`은 전부 기록).
전송 명령·이벤트는 샘플링하지 않으며, 정책 입력·힘 history·제어 주기는 그대로다.
샘플링 생략 수는 `sampling_summary` 이벤트에 남고, 20Hz 기록으로 고주파 힘 피크를 평가할 수는 없다.
`use_stain_mask:=false`이면 불필요한 mask publisher도 실행하지 않는다.
Dual-camera와 gripper launch는 기존 시각화 기본값을 사용한다.

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


---

<a id="readme-source-02"></a>

# 원본: `.pytest_cache/README.md`

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.


---

<a id="readme-source-03"></a>

# 원본: `behavior_ws/src/dynamixel_joy_controller/README.md`

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


---

<a id="readme-source-04"></a>

# 원본: `behavior_ws/src/nrs_ft_aq2/README.md`

# NRS_FT_AQ
NRS FT sensor data aquisition (eCAN)


---

<a id="readme-source-05"></a>

# 원본: `behavior_ws/src/nrs_imitation/.pytest_cache/README.md`

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/cache.html) for more information.


---

<a id="readme-source-06"></a>

# 원본: `behavior_ws/src/nrs_imitation/README.md`

# nrs_imitation

이 저장소의 기본 학습 흐름은 아래 4단계입니다.

```text
HDF5 recording -> imitation_form 변환 -> Flow/ACT train -> ROS2 inference
```

현재 일반 데이터 파이프라인은 `single_cam`과 `dual_cam`으로 분기되어 있습니다.

## 0. 공통 준비

ROS2 node 실행:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
```

Python script 실행:

```bash
cd ~/nrs_imitation
```

## 1. HDF5 Recording

recording 결과는 merged HDF5로 저장됩니다.

```text
single_cam        : ~/nrs_imitation/datasets/polishing/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
gripper single_cam: ~/nrs_imitation/datasets/gripper/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/gripper_hdf5_recorder_single_cam_*.hdf5
dual_cam          : ~/nrs_imitation/datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
```

### Single Cam

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam
```

stain mask를 변환 단계에서 생성하려면 stain-mask recorder를 사용합니다.

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam_stain_mask
```

이 경우 `ep_0000`은 깨끗한 표면 reference입니다. 카메라를 움직이며 recording할 때는 `ep_0000`도 같은 이동 경로로 깨끗한 표면을 먼저 녹화해야 합니다. 고정 reference 한 장면만 있으면 이동 중 배경/경계가 stain으로 잘못 잡힐 수 있습니다.

빛반사 하이라이트를 줄인 `cam0` RGB를 저장:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam --ros-args \
  -p image_preprocess_mode:=highlight_attenuate \
  -p image_specular_mask_mode:=bright \
  -p image_specular_v_thresh:=220 \
  -p image_specular_dilate_px:=2 \
  -p image_specular_attenuate_gain:=0.35
```

기본 입력:

```text
position = /calibrated_pose
force    = /ftsensor/measured_Cvalue
cam0     = /realsense/vr/color/image_raw
```

### Gripper Single Cam

```bash
ros2 run nrs_imitation gripper_hdf5_recorder_single_cam
```

기본 입력:

```text
position         = /calibrated_pose
force            = /ftsensor/measured_Cvalue
cam0             = /realsense/vr/color/image_raw
gripper position = /gripper/present_position       # std_msgs/msg/Int32
gripper current  = /gripper/present_current_mA     # std_msgs/msg/Float32
```

### Dual Cam

```bash
ros2 run nrs_imitation hdf5_recorder_dual_cam
```

기본 입력:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/vr/color/image_raw
cam1     = /realsense/global/color/image_raw
```

### Start / End

녹화 시작:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: start_recording}"
```

녹화 종료:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: end_recording}"
```

조이스틱 컨트롤러:

```bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

기본 F710/XInput 매핑:

```text
RB: recorder start_recording
LB: recorder end_recording
Start: gripper close (/gripper/command = -653)
Back : gripper open  (/gripper/command = 733)
RT   : gripper close by 50 tick
LT   : gripper open by 50 tick
```

## 2. Imitation Form 변환

변환 결과는 기본적으로 같은 run directory 아래 `imitation_form/`에 저장됩니다.

```text
single_cam        : ~/nrs_imitation/datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
gripper single_cam: ~/nrs_imitation/datasets/gripper/single_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
dual_cam          : ~/nrs_imitation/datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
```

변환 후 HDF5 구조는 학습에 필요한 항목만 남깁니다.

```text
observations/position
observations/force
observations/images/cam0
observations/images/cam1   # dual_cam only
action/position
action/force
```

gripper recorder로 생성한 데이터는 gripper 변환 스크립트를 사용하면 아래 항목이 추가됩니다.

```text
observations/gripper/present_current_mA
observations/gripper/present_position
action/gripper_goal_current_mA
action/gripper_present_current_mA
action/gripper_present_position
```

`marker`, `qpos`, `action_flat`, `meta`, `is_pad`는 일반 imitation form에서 쓰지 않습니다.

### Single Cam

최신 single-cam recording을 자동 선택:

```bash
python3 source/custom/demo_data_imitation_form_single_cam.py --write_summary
```

moving-camera stain mask 변환은 current episode를 clean reference episode에 monotonic pose-sequence DTW로 먼저 매칭한 뒤 homography 정렬, top-k reference consensus, pose-distance guard, reference-diff core mask와 제한된 dark-prior 보강, temporal gap filling, temporal pruning을 사용합니다. 변환 로그에 `temporal-filled`가 나오면 짧게 누락된 mask frame을 인접 frame에서 복구한 것이고, `temporal-pruned`가 나오면 인접 정렬 frame의 support가 없는 고립 component를 제거한 것입니다. `had no close clean reference`가 많이 나오면 같은 이동 경로의 clean reference sweep을 다시 녹화하거나, overlay를 확인한 뒤 `--stain_reference_max_pose_dist`를 조정합니다.

특정 파일을 지정:

```bash
python3 source/custom/demo_data_imitation_form_single_cam.py \
  --input_h5 datasets/polishing/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

최신 gripper single-cam recording을 자동 선택:

```bash
python3 source/custom/gripper_data_imitation_form_single_cam.py --write_summary
```

특정 gripper 파일을 지정:

```bash
python3 source/custom/gripper_data_imitation_form_single_cam.py \
  --input_h5 datasets/gripper/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/gripper_hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/gripper/single_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

### Dual Cam

최신 dual-cam recording을 자동 선택:

```bash
python3 source/custom/demo_data_imitation_form_dual_cam.py --write_summary
```

특정 파일을 지정:

```bash
python3 source/custom/demo_data_imitation_form_dual_cam.py \
  --input_h5 datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

## 3. Train

Flow/ACT train은 모두 `single_cam`과 `dual_cam`으로 분리되어 있습니다.  
gripper single-cam은 Flow 전용 entrypoint가 별도로 있습니다.
`--dataset_dir`를 생략하면 각 dataset root 아래 최신 `imitation_form/episode_*.hdf5`를 자동 선택합니다.

checkpoint 기본 저장 위치:

```text
Flow single_cam        : ~/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>/
Flow gripper single_cam: ~/nrs_imitation/checkpoints/flow/gripper/single_cam/<YYYYMMDD_HHMM>/
Flow dual_cam          : ~/nrs_imitation/checkpoints/flow/polishing/dual_cam/<YYYYMMDD_HHMM>/
ACT single_cam         : ~/nrs_imitation/checkpoints/act/polishing/single_cam/<YYYYMMDD_HHMM>/
ACT dual_cam           : ~/nrs_imitation/checkpoints/act/polishing/dual_cam/<YYYYMMDD_HHMM>/
```

### Single Cam

```bash
python3 scripts/flow/train_flow_single_cam.py
python3 scripts/act/train_act_single_cam.py
```

특정 imitation_form 지정:

```bash
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form

python3 scripts/act/train_act_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

### Gripper Single Cam

최신 gripper imitation_form을 자동 선택:

```bash
python3 scripts/flow/train_flow_gripper_single_cam.py
```

특정 gripper imitation_form 지정:

```bash
python3 scripts/flow/train_flow_gripper_single_cam.py \
  --dataset_dir datasets/gripper/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

이 policy는 `cam0 + qpos(position, force) + gripper state(position, current)`를 observation으로 사용합니다. gripper state encoder는 MLP이고, action target은 `position(6) + force(3) + gripper_present_position(1) + gripper_goal_current_mA(1)`의 11D입니다. `gripper_goal_current_mA`는 recorded `present_current_mA`의 magnitude에서 만들어지며, signed `action/gripper_present_current_mA`는 분석/호환용으로 보존됩니다.

### Dual Cam

```bash
python3 scripts/flow/train_flow_dual_cam.py
python3 scripts/act/train_act_dual_cam.py
```

특정 imitation_form 지정:

```bash
python3 scripts/flow/train_flow_dual_cam.py \
  --dataset_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form

python3 scripts/act/train_act_dual_cam.py \
  --dataset_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form
```

## 4. Inference

inference도 `single_cam`과 `dual_cam`으로 분리되어 있습니다.  
`ckpt_dir`를 생략하면 아래 경로에서 최신 `policy_best.ckpt`를 자동 선택합니다.

```text
Flow single_cam: ~/nrs_imitation/checkpoints/flow/polishing/single_cam/*/policy_best.ckpt
Flow dual_cam  : ~/nrs_imitation/checkpoints/flow/polishing/dual_cam/*/policy_best.ckpt
Flow gripper   : ~/nrs_imitation/checkpoints/flow/gripper/single_cam/*/policy_best.ckpt
ACT single_cam : ~/nrs_imitation/checkpoints/act/polishing/single_cam/*/policy_best.ckpt
ACT dual_cam   : ~/nrs_imitation/checkpoints/act/polishing/dual_cam/*/policy_best.ckpt
```

### Single Cam

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
```

ACT checkpoint를 사용할 때:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py policy_class:=ACT
```

기본 입력:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/vr/color/image_raw
```

특정 checkpoint 지정:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  ckpt_dir:=/home/nrs_display/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>
```

ACT checkpoint를 직접 지정할 때는 `policy_class:=ACT`와 ACT checkpoint 경로를 같이 지정합니다.

### Gripper Single Cam

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py
```

기본 입력/출력:

```text
position         = /ur10skku/currentP
force            = /ur10skku/currentF
cam0             = /realsense/vr/color/image_raw
gripper position = /gripper/present_position
gripper current  = /gripper/present_current_mA
robot command    = /ur10skku/cmdMotion
gripper position = /gripper/command
gripper current  = /gripper/goal_current_mA
heatmap overlay  = /inference_gripper_single_cam/gradcam_overlay
```

`ckpt_dir`를 생략하면 `checkpoints/flow/gripper/single_cam` 아래 최신 checkpoint를 자동 선택합니다. 이 node는 polishing inference와 같은 shared control loop를 사용해서 policy action `[0:9]`를 `/ur10skku/cmdMotion`으로 publish하고, 추가로 action `[9]`의 `gripper_present_position`을 `std_msgs/Int32`로 `/gripper/command`, action `[10]`의 `gripper_goal_current_mA`를 `std_msgs/Float32`로 `/gripper/goal_current_mA`에 publish합니다. stain-mask node/option은 사용하지 않습니다.

11D action은 checkpoint의 `dataset_stats.pkl` 범위로 denormalize합니다. robot motion `[0:9]`는 polishing inference의 temporal aggregation, anchor, stage, startup ramp, `step_cap_pos_mm`, `step_cap_ang_rad`, `step_cap_fz`, `cmd_safety_max_xyz_from_current_mm` 경로를 그대로 통과합니다. gripper position action `[9]`는 같은 `tau_sec:=0.8`, `startup_ramp_sec:=3.0`을 사용하고, 기본 `gripper_command_step_cap_tick:=200.0`, `gripper_command_slew_per_sec:=1000.0`, `gripper_cmd_safety_max_tick_from_present:=1500.0` guard를 추가로 통과합니다. gripper current action `[10]`은 기본 `0..1345mA`로 clip되어 publish됩니다. driver의 close-current-stop은 `max(close_current_stop_mA, goal_current_mA)` 기준으로 latch되며, 실제 파지력 안전 상한은 `dxl.goal_current_max_mA`로 제한합니다.

tracker pose/force topic 기준으로 실행할 때:

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py \
  pose_topic:=/calibrated_pose \
  force_topic:=/ftsensor/measured_Cvalue \
  force_msg_type:=wrench
```

특정 checkpoint 지정:

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py \
  ckpt_dir:=/home/eunseop/nrs_imitation/checkpoints/flow/gripper/single_cam/<YYYYMMDD_HHMM>
```

### Dual Cam

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
```

ACT checkpoint를 사용할 때:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py policy_class:=ACT
```

기본 입력:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
cam1     = /realsense/global/color/image_raw
```

특정 checkpoint 지정:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py \
  ckpt_dir:=/home/nrs_display/nrs_imitation/checkpoints/flow/polishing/dual_cam/<YYYYMMDD_HHMM>
```

ACT checkpoint를 직접 지정할 때는 `policy_class:=ACT`와 ACT checkpoint 경로를 같이 지정합니다.

Polishing single-camera launch는 `rqt_image_view` 창을 기본으로 열지 않습니다.
`inference_clean_single_cam.launch.py`는 flow-vector와 modality 영상을 창 없이
`~/Videos/Screencasts/*.webm`에 자동 저장합니다. 필요할 때만
`visualize_flow_vector:=true` 또는 `visualize_modality_importance:=true`로 창을 엽니다.
영상/overlay 기본값은 10fps (`overlay_record_fps`)이며, 인코더는 1 thread와 낮은
CPU 우선순위를 사용합니다. `metrics_log_enable:=true`에서도 추가 고주파 토픽 구독은
기본 OFF (`metrics_extra_telemetry_enable:=false`), 위치·힘/비교 CSV 기록은 최대
20Hz (`metrics_sample_hz:=20.0`, `0.0`은 전부 기록)입니다. 전송 명령과 이벤트는
샘플링하지 않고, 정책 입력·힘 history·로봇 제어 주기도 바꾸지 않습니다.
샘플링된 기록은 고주파 힘 피크 분석용이 아니며, 생략 수는 `sampling_summary`에 남습니다.
Dual-camera launch는 기존 시각화 기본값을 사용합니다.

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
```

visualization topics:

```text
single_cam: /inference_single_cam/gradcam_overlay       # local Grad-CAM overlay
single_cam: /inference_single_cam/stain_mask_overlay    # generated stain-mask overlay when auto_stain_mask=true
dual_cam  : /inference_dual_cam/gradcam_overlay         # local Grad-CAM overlay
dual_cam  : /inference_dual_cam/gradcam_overlay_global  # global Grad-CAM only
```

Grad-CAM 시각화는 끌 수 있습니다.

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py gradcam_enable:=false
```

## Stage-1 VR Workflow

Stage-1 VR episode recorder는 일반 imitation-form pipeline과 별도입니다. 기본 저장 위치는 이제 `datasets/stage1`입니다.

```text
~/nrs_imitation/datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes/episode_*.hdf5
```

녹화:

```bash
ros2 run nrs_imitation vr_stage1_hdf5_recorder
```

최신 stage1 episode directory를 자동 선택해서 robot playback PC로 push:

```bash
ros2 run nrs_imitation vr_stage1_episode_pusher
```

특정 episode directory 지정:

```bash
ros2 run nrs_imitation vr_stage1_episode_pusher --ros-args \
  -p episode_dir:=/home/nrs_display/nrs_imitation/datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes
```


---

<a id="readme-source-07"></a>

# 원본: `behavior_ws/src/stain_relative_frame/.pytest_cache/README.md`

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.


---

<a id="readme-source-08"></a>

# 원본: `behavior_ws/src/stain_relative_frame/README.md`

# stain_relative_frame

Express the TCP trajectory **relative to the stain centre** instead of in
absolute robot base coordinates, so that absolute position stops being a
shortcut for predicting the defect type.

The package is a chain of gates. Each stage refuses to run until the previous
one has passed, and each writes a JSON report that the next stage reads.

```
[0a] home_pose_repeatability   -> is the camera pose reproducible?      GATE
[0b] clean_reference_capture   -> clean-specimen reference + lighting
[1]  homography_collect
     homography_pick_pixels
     homography_fit            -> pixel -> robot mm, held-out <= 2 mm    GATE
[2]  stain_origin_offline      -> per-episode stain_origin + stability
     stain_origin_node         -> the same thing, online, latched once
[3]  dataset_relativize        -> rewrite the dataset, recompute stats
[4]  validate_shortcut         -> shortcut-leakage gate                  GATE
[5]  audit_inference_path      -> training and inference share one path  GATE
     run_pipeline              -> drives [2]-[5] and prints the summary
```

## The two rules that shape the design

**The origin is an episode constant.** It is resolved once, at the home pose,
before the episode starts, and frozen. `StainOrigin.freeze()` raises on any
later write; `stain_origin_node` destroys its camera subscription the moment
it resolves, so there is no code path that could drift. `audit_inference_path`
fails the build if anything reaches a detector from a callback.

**Translation only — never rotation.** Aligning the frame to the stain's
principal axis would map 0 / 30 / 60 / 90 deg onto the same input and delete
the learning target. `relative_frame.to_relative()` takes no angle argument
and there is nowhere to add one without changing every call site.

## Quick start

```bash
colcon build --packages-select stain_relative_frame
source install/setup.bash
```

### [0] Preconditions — hardware required

```bash
# where is the arm now? (seeds --home_pose)
ros2 run stain_relative_frame home_pose_repeatability -- --measure_only

# GATE: 10 returns to home, xy std <= 1 mm, rotation std <= 0.5 deg
ros2 run stain_relative_frame home_pose_repeatability -- \
    --home_pose 420.34 345.54 211.86 -0.009 0.164 2.444 --trials 10
# ... or --manual to move the arm by hand between trials

# clean specimen at the home pose, 40 frames averaged
ros2 run stain_relative_frame clean_reference_capture -- \
    --lighting "ring light 60%, blinds closed"
```

The lighting note is mandatory: step [2] subtracts this reference from the
episode frames, so it is only valid under the illumination it was shot in.

### [1] Homography — once, offline

```bash
ros2 run stain_relative_frame homography_collect      # 8 touches + home image
ros2 run stain_relative_frame homography_pick_pixels  # click, or --method aruco
ros2 run stain_relative_frame homography_fit -- --all_splits
```

Fits on 6 points, holds out 2, gates at 2 mm. On failure: spread the points
wider and re-collect; if it still fails, calibrate the lens, put
`camera_matrix` / `dist_coeffs` in the config and re-run with `--undistort`.
`--all_splits` reports every possible hold-out split, which shows whether one
bad point is carrying the result.

### [2]-[5] Offline

```bash
ros2 run stain_relative_frame run_pipeline -- \
    --dataset_dir datasets/polishing/single_cam/<run>/imitation_form \
    --out_dir     datasets/polishing/single_cam/<run>/imitation_form_rel \
    --use_relative_position
```

or one stage at a time (`stain_origin_offline`, `dataset_relativize`,
`validate_shortcut`, `audit_inference_path`). Every stage takes `--config`.

### Inference

```bash
ros2 launch stain_relative_frame stain_origin_online.launch.py
```

Start it **before** the inference stack. It resolves the origin from the
home-pose view, latches it on `/stain_relative_frame/stain_origin`
(transient local) and releases the camera. A new episode needs an explicit
`~/new_episode` service call.

`nrs_imitation`'s `inference_core.py` already wires this in (see
`_srf_observation_pose6` / `_srf_command_seq` / `_absolutize_demo_start_pose`):

* it reads `use_relative_position` from the **checkpoint's**
  `dataset_stats.pkl` -- not a launch argument -- so an absolute-trained
  policy can never be driven through the relative path and vice versa;
* with the flag off, `self._srf` is `None` and every hook short-circuits:
  an absolute-frame run is byte-for-byte unchanged;
* with the flag on it subscribes to `stain_origin_topic`
  (`:=` overridable, default `/stain_relative_frame/stain_origin`), blocks
  demo-start alignment and inference until the origin is latched, converts
  the measured TCP pose to the stain frame before the policy sees it, and
  converts the predicted trajectory (and `demo_start_pose_mean`) back to
  absolute base coordinates before the robot does.

`inference_clean_single_cam.launch.py` **auto-launches `stain_origin_node`**
when the checkpoint is stain-relative (`stain_origin_autostart:=auto`, the
default; `true` forces it, `false` disables it). It reads the step-[2]
`detect_params` from the checkpoint stats' `stain_origin_report` so the online
detector matches the trained frame. So the only operator step is:

> **park the arm at the home / viewing pose, draw the stain, then launch the
> inference stack as usual.**

The origin node collects ~10 frames (<1 s), resolves + latches, and releases
its subscriptions; the inference node then does its demo-start alignment.
`method=dark` (auto-selected for a `depth_extrinsic` homography) needs the
pose topic for the per-frame homography.

To run the origin node yourself instead (`stain_origin_autostart:=false`):
`ros2 launch stain_relative_frame stain_origin_online.launch.py
detect_params:=<report.json>` before the inference stack.

Checkpoints trained before `flow_train_core.carry_forward_relative_frame_stats`
existed do not carry the flag in their own stats;
`scripts/stain_relative_frame/patch_checkpoint_stats.py <ckpt_dir>` backfills
it from the converted dataset's stats (inference also falls back to the
dataset stats named in `dataset_dir` automatically).

The manual form, if you wire your own node:

```python
from stain_relative_frame.inference_adapter import StainOriginClient

client = StainOriginClient(self, stats_path=f"{ckpt_dir}/dataset_stats.pkl")
client.wait_until_ready(timeout_sec=30.0)     # once, before the episode

qpos_rel = client.observation(qpos_abs)       # per step: policy input
cmd_abs  = client.command(action_rel)         # per step: robot command
```

## Live diagnostics (not gated)

Two bring-up aids for the real rig. Neither is part of the acceptance
pipeline; both need `homography.json` from `homography_depth_calibrate`
(method=depth_extrinsic) and use the reference-free dark-blob detector with a
per-frame homography (`homography.per_frame_homography`).

```bash
# watch relative = to_relative(TCP, stain_origin) update live as you draw the
# stain or jog the arm. --freeze (default) latches the origin like the node
# does; --redetect re-runs the detector every frame (good for tuning the ROI).
ros2 run stain_relative_frame live_relative_check -- \
    --plate_roi 200 68 340 140 --tool_box 205 100 300 240 --dark_thresh 70

# PTP the arm through a +/-12 mm XY star and check the relative position
# follows. Z and orientation are held, offsets are clamped to --max_offset_mm,
# and the arm returns to the recorded home pose in a finally block. The gate is
# on the frozen-origin path (must track d_TCP within --track_tol_mm); the
# per-frame redetect drift is reported for information only.
ros2 run stain_relative_frame ptp_relative_test        # -> checkpoints/.../ptp_relative_test.json
```

The `live_relative_check` prints `to_relative` output directly and
`ptp_relative_test` drives it through `RelativeFrameAdapter`, so both exercise
the same transform the converter and the inference adapter use -- the [5]
audit still sees one preprocessing path.

## The `use_relative_position` flag

Default **False**. With it off, `dataset_relativize` copies every dataset
verbatim and then re-opens both files and checks `np.array_equal` on all of
them — the "identical to the old path" guarantee is verified per run, not
asserted. Verified on 84 real episodes: 0 mismatches.

With it on:

| field | treatment |
| --- | --- |
| `observations/position[:, :2]` | `- stain_origin` |
| `action/position[:, :2]` | `- stain_origin` |
| z, rx, ry, rz | untouched (checked) |
| `observations/force`, `action/force` | untouched (checked) |
| `analysis/absolute/*` | absolute trajectories preserved, not policy inputs |
| `dataset_stats.pkl` | recomputed on the relative coordinates |

Recomputing the stats is not optional: on a real run the x range went from
142 mm wide (absolute) to 20 mm wide (relative). Reusing absolute statistics
would put the normaliser off by the whole workpiece offset.

## The [4] gate

Four checks on the **first frame** of each episode, split 8:2 by episode,
stratified, repeated over many splits:

* **(a)** relative position -> defect type — must be near chance
* **(b)** force -> defect type — must be near chance (removing position can
  push the shortcut onto force)
* **(c)** label-shuffle control — must land on chance, or the split leaks and
  (a) means nothing
* **(d)** scatter of relative start positions per type — must overlap

A result fails only when it is **both** above `chance + margin` **and**
outside its own permutation null (`p < alpha`). At these episode counts a test
split holds ~17 episodes, so one extra correct episode moves the accuracy by
6 points and a fixed margin alone fires on noise. The null is measured from
200 label shuffles of this dataset, so the gate calibrates itself to the
sample size. Both numbers are always printed.

The absolute-position baseline is reported alongside as `info`. It is expected
to be high — it is the shortcut being removed. If it is already at chance, the
conversion is not what is being measured.

## Config

`config/stain_relative_frame.yaml` holds every threshold the gates are judged
against; each report records the values it ran with. Pass `--config` to any
executable, or edit the installed copy.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_pipeline.py -q
```

37 tests over a synthetic fixture with a known homography, known stain
positions and a type-independent approach offset — so the gate is tested
against data whose answer is known, including datasets with an injected force
confound and an injected position confound that it must reject.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed because ROS Humble's
`launch_testing` pytest plugin is incompatible with the newer pytest in the
conda env; this is a pre-existing workspace quirk, not a package issue.

## Artifacts

Everything lands in `checkpoints/stain_relative_frame/` by default:

```
home_pose_repeatability.json    [0a] per-trial poses, std, verdict
clean_reference.npz             [0b] mean gray reference + lighting note
homography_points.npz           [1a/1b] home image, robot pts, pixel pts
homography.json                 [1c] H, held-out errors, verdict
stain_origin_stability.json     [2]  per-episode origin, spread, failures
validation_gate.json            [4]  accuracies, nulls, p-values, verdict
validation_scatter.png          [4d] relative vs absolute start positions
inference_path_audit.json       [5]  findings
```


---

<a id="readme-source-09"></a>

# 원본: `behavior_ws/src/umi_ros2/README.md`

# umi_ros2

`umi_ros2`는 Dynamixel 기반 UMI 그리퍼를 ROS 2에서 제어하기 위한 Python 패키지입니다. 현재 패키지에는 세 가지 실행 방식이 포함되어 있습니다.

- `umi_gripper_pico`: 키보드 입력과 `/gripper/command` 토픽을 이용한 그리퍼 제어
- `umi_gripper_sub`: `/gripper/command` 토픽과 조이스틱 토픽을 이용한 그리퍼 제어
- `umi_gripper_sub_pwm`: 터미널 키보드 입력을 이용한 PWM 모드 테스트

기본적으로 Dynamixel을 Current-based Position Control Mode(Mode 5)로 사용하며, 현재값과 위치를 ROS 토픽으로 publish합니다.

## 1. 패키지 위치

이 패키지는 워크스페이스 루트의 `src/umi_ros2` 아래에 두면 됩니다.

```bash
<workspace_root>/src/umi_ros2
```

## 2. 의존성

`setup.py` 기준 Python 의존성:

- `dynamixel-sdk`
- `opencv-python`
- `pyserial`

ROS 2 의존성:

- `rclpy`
- `std_msgs`
- `launch`
- `launch_ros`
- `ament_index_python`

## 3. 빌드

워크스페이스 루트에서 빌드합니다.

```bash
cd <workspace_root>
colcon build --packages-select umi_ros2
source install/setup.bash
```

## 4. 실행 가능한 노드

`setup.py`에 등록된 ROS 2 실행 엔트리는 아래 세 개입니다.

- `umi_gripper_pico`
- `umi_gripper_sub`
- `umi_gripper_sub_pwm`

직접 실행:

```bash
ros2 run umi_ros2 umi_gripper_pico
ros2 run umi_ros2 umi_gripper_sub
ros2 run umi_ros2 umi_gripper_sub_pwm
```

런치 파일 실행:

```bash
ros2 launch umi_ros2 umi_grp.launch.py
ros2 launch umi_ros2 umi_grp_sub.launch.py
```

## 5. 각 노드 설명

### 5.1 `umi_gripper_pico`

관련 파일:

- [umi_ros2/umi_grp.py](umi_ros2/umi_grp.py)
- [config/umi_grp.yaml](config/umi_grp.yaml)
- [launch/umi_grp.launch.py](launch/umi_grp.launch.py)

특징:

- 기본 노드 이름: `umi_gripper`
- 기본 시리얼 포트: `/dev/ttyUSB0`
- 기본 보드레이트: `57600`
- 기본 그리퍼 ID: `0`
- `/gripper/command` (`std_msgs/Int32`)를 subscribe
- `/gripper/present_current_mA` (`std_msgs/Float32`) publish
- `/gripper/present_position` (`std_msgs/Int32`) publish
- 키보드 입력으로 open/close 및 미세 조정 가능

기본 키보드 조작:

- `o` 또는 `O`: open
- `c` 또는 `C`: close
- `+` 또는 `=`: 목표 tick 증가
- `-` 또는 `_`: 목표 tick 감소
- `ESC`: 종료

실행 예:

```bash
ros2 launch umi_ros2 umi_grp.launch.py
```

직접 노드 실행 예:

```bash
ros2 run umi_ros2 umi_gripper_pico --ros-args --params-file \
  $(ros2 pkg prefix umi_ros2)/share/umi_ros2/config/umi_grp.yaml
```

### 5.2 `umi_gripper_sub`

관련 파일:

- [umi_ros2/umi_grp_sub.py](umi_ros2/umi_grp_sub.py)
- [config/umi_grp_sub.yaml](config/umi_grp_sub.yaml)
- [launch/umi_grp_sub.launch.py](launch/umi_grp_sub.launch.py)

특징:

- 기본 노드 이름: `umi_gripper_sub`
- 기본 시리얼 포트: `/dev/ttyUSB0`
- 기본 보드레이트: `57600`
- 기본 그리퍼 ID: `0`
- `/gripper/command` (`std_msgs/Int32`)를 subscribe
- `/ur10skku/joy_move` (`std_msgs/Float64MultiArray`)를 subscribe
- `/gripper/present_current_mA` (`std_msgs/Float32`) publish
- `/gripper/present_position` (`std_msgs/Int32`) publish
- 조이스틱 축 값을 그리퍼 tick 범위로 매핑

조이스틱 기본 파라미터:

- `joystick.enabled: true`
- `joystick.command_topic: /ur10skku/joy_move`
- `joystick.axis_index: 5`

실행 예:

```bash
ros2 launch umi_ros2 umi_grp_sub.launch.py
```

직접 노드 실행 예:

```bash
ros2 run umi_ros2 umi_gripper_sub --ros-args --params-file \
  $(ros2 pkg prefix umi_ros2)/share/umi_ros2/config/umi_grp_sub.yaml
```

## 6. 주요 토픽

### Subscribe

- `/gripper/command` (`std_msgs/Int32`)
  - 목표 그리퍼 위치 tick 전달
- `/ur10skku/joy_move` (`std_msgs/Float64MultiArray`)
  - `umi_gripper_sub`에서 사용
  - 지정한 `axis_index` 값을 `gripper.min_tick` ~ `gripper.max_tick`으로 매핑

예시:

```bash
ros2 topic pub /gripper/command std_msgs/msg/Int32 "{data: 1500}" -1
```

### Publish

- `/gripper/present_current_mA` (`std_msgs/Float32`)
- `/gripper/present_position` (`std_msgs/Int32`)

모니터링 예시:

```bash
ros2 topic echo /gripper/present_position
ros2 topic echo /gripper/present_current_mA
```

## 7. 파라미터 파일

기본 파라미터 파일:

- [config/umi_grp.yaml](config/umi_grp.yaml)
- [config/umi_grp_sub.yaml](config/umi_grp_sub.yaml)

런치 파일은 기본적으로 위 YAML 파일을 사용하며, `config_file` 인자로 다른 파일을 넘길 수 있습니다.

예시:

```bash
ros2 launch umi_ros2 umi_grp.launch.py config_file:=./config/custom.yaml
ros2 launch umi_ros2 umi_grp_sub.launch.py config_file:=./config/custom.yaml
```

주요 파라미터 예:

- `dxl.port`: Dynamixel 연결 포트
- `dxl.baud`: 보드레이트
- `dxl.gripper_id`: Dynamixel ID
- `gripper.min_tick`, `gripper.max_tick`: 동작 범위
- `dxl.goal_current_mA`: 목표 전류
- `gripper.close_current_stop_mA`: 물체 파지 시 정지 기준 전류
- `gripper.command_topic`: 명령 토픽 이름

`umi_gripper_pico` 전용 파라미터:

- `trigger.min_tick`, `trigger.max_tick`
- `gripper.invert`
- `monitor.enabled`
- `monitor.print_period_sec`
- `keyboard.step_size`

`umi_gripper_sub` 전용 파라미터:

- `gripper.close_increases_tick`
- `joystick.enabled`
- `joystick.command_topic`
- `joystick.axis_index`

## 8. PWM 테스트 노드

[umi_ros2/umi_grp_sub_pwm.py](umi_ros2/umi_grp_sub_pwm.py)는 PWM 모드 테스트용 노드입니다.

주의:

- 터미널 키 입력을 사용하므로 실행 터미널이 TTY여야 합니다.
- 기본 파라미터는 코드 내부 기본값을 사용합니다.

실행:

```bash
ros2 run umi_ros2 umi_gripper_sub_pwm
```

기본 키:

- `o`: open
- `c`: close
- `s`: stop
- `q`: quit

## 9. 실행 전 확인 사항

- Dynamixel 장치가 `dxl.port`에 실제로 연결되어 있어야 합니다.
- 장치 ID와 보드레이트가 YAML 설정과 일치해야 합니다.
- 현재 설정은 기본적으로 `/dev/ttyUSB0`, ID `0`, baud `57600`을 가정합니다.
- 시리얼 장치 권한 문제로 실행이 실패할 수 있습니다.
- 전류 제한, 위치 제한, stop threshold는 실제 하드웨어 기준으로 반드시 재확인해야 합니다.

## 10. 빠른 시작

토픽 기반 제어를 가장 빠르게 시험하려면:

```bash
cd <workspace_root>
colcon build --packages-select umi_ros2
source install/setup.bash
ros2 launch umi_ros2 umi_grp_sub.launch.py
```

다른 터미널에서:

```bash
cd <workspace_root>
source install/setup.bash
ros2 topic pub /gripper/command std_msgs/msg/Int32 "{data: 1500}" -1
```


---

<a id="readme-source-10"></a>

# 원본: `behavior_ws/src/vr_calibration/README.md`

# vr_calibration

## 바로 실행

orientation 오차가 크지 않을 때, 기존 `T_SA`를 유지하고 position/base calibration만 갱신:

```bash
cd <your_ros2_workspace>
source install/setup.bash
ros2 run vr_calibration vr_calibration
```

orientation 오차가 클 때, `/calibrated_pose` 기준 rotation도 다시 맞춤:

```bash
cd <your_ros2_workspace>
source install/setup.bash
ros2 run vr_calibration vr_calibration --ros-args \
  -p t_sa_mode:=update \
  -p t_sa_max_delta_deg:=180.0
```

`vr_calibration`은 UR robot EE pose와 Vive tracker raw pose를 같은 target waypoint에서 수집한 뒤, `vive_tracker_ros2` 런타임이 사용할 calibration YAML을 생성하는 ROS 2 패키지다.

이 문서는 `nrs_imitation` 전체가 아니라 `behavior_ws/src/vr_calibration` 패키지 기준으로만 정리한다.

## 입력과 출력

입력 topic:

- `/ur10skku/currentP`: `Float64MultiArray`, robot current pose `[x y z wx wy wz]`
- `/raw_pose`: `PoseStamped`, Vive tracker raw pose
- `/calibrated_pose`: `Float64MultiArray`, `T_SA` update 모드에서 현재 calibrated rotation을 읽기 위해 사용
- `/ur10skku/ftdata_tcp_raw`: `WrenchStamped`, Y2FT_AQ가 publish하는 보상 전 TCP-frame robot FT 값

주요 파일:

- `vr_calibration/txt/for_vr_calibration_point_v8.txt`: 320 mm EE-to-TCP spindle용 기본 target waypoint 파일
  (v7 대비: home pose가 workpiece 중심(445, 394.5, 220)으로 바뀌면서, 위치용/평면 포인트 14개를
  workpiece 쪽으로 절반 정도 옮기고 z를 표면(170mm) 위 220mm로 맞췄다. 회전 다양성용 24개는 그대로
  두고, workpiece 사각형 코너 4개+중심 1개는 유지한 채 변 중점 4개를 추가했다. v7은 이전 버전으로 보존.)
- `vr_calibration/txt/ur10_ee.txt`: 캡처된 EE pose 기록
- `vr_calibration/txt/ur10_vr.txt`: 캡처된 VR pose 기록
- `vive_tracker_ros2/yaml/calibration_matrix.yaml`: 최종 calibration YAML
- `nrs_ft_aq2/config/spindle_gravity.yaml`: robot FT로 식별한 공통 spindle 중력보상 행렬

## Spindle 중력보상 동시 calibration

이 모드에서는 spindle을 로봇에 장착하고 **로봇 FT만 실행**한다. 교시장치의 `nrs_ft_aq2`는 실행하지 않는다.
Y2FT_AQ는 센서 zero 이후의 값을 TCP 축으로만 회전한 `/ur10skku/ftdata_tcp_raw`를 제공하고,
VR calibration 노드는 정지한 각 capture pose의 wrench 중앙값을 함께 저장한다.

여러 자세에 대해 다음 모델을 식별한다.

```text
wrench_tcp = bias + G_spindle(6x3) * gravity_tcp
```

runtime에 저장되는 `G_spindle`은 자유 6x3 회귀값을 그대로 쓰지 않고, 식별한 질량과 3축 CoM으로 만든
물리적으로 일관된 행렬이다. STL 형상, STL density, STL CoM은 사용하지 않는다. 센서 zero로 제거된 상수값은
회귀의 `bias`가 흡수하며 runtime은 zero 자세와 현재 자세의 `delta gravity`에만 행렬을 적용한다.

기본 v7 waypoint는 `EE2TCP` 길이 320 mm를 기준으로 만든다. 처음 8개 capture pose는 수직 spindle로
중앙 작업공간의 위치 분포를 만들고, 나머지 24개는 EE를 안전한 고점 부근에 유지한 채 기존 v6에서 사용한
자세를 회전 변화가 작은 순서로 배치한다. TCP/EE 끝점 범위와 중력 방향 분포를 수치 검증했지만, 이는 로봇 및
주변 설비의 실제 collision model을 대신하지 않는다. 최초 실행은 반드시 저속/수동 정지 준비 상태에서 확인한다.

실행 순서:

```bash
# nrs_forcecon@192.168.0.151
cd /home/nrs_forcecon/dev_ws
source install/setup.bash
ros2 run Y2FT_AQ FTGetMain

# eunseop_nrs3 (Vive tracker/robot waypoint 노드가 준비된 뒤)
cd /home/eunseop/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run vr_calibration vr_calibration
```

성공 조건을 모두 만족할 때만 로컬 `nrs_ft_aq2/config/spindle_gravity.yaml`을 교체하고 기존 파일은
`.bak`으로 보존한다. 기본 설정에서는 passwordless SSH로 다음 원격 파일도 같은 YAML로 교체한다.

```text
nrs_forcecon@192.168.0.151:/home/nrs_forcecon/dev_ws/src/y2_ur10skku_control/Y2FT_AQ/config/spindle_gravity.yaml
```

원격 파일도 `.bak`으로 보존된다. Y2FT_AQ는 YAML을 시작할 때 읽으므로 calibration이 끝난 뒤 로봇 FT 노드를
재시작해야 새 행렬이 적용된다. 교시장치 FT는 calibration 중 꺼져 있었으므로, 나중에 실행할 때 새 로컬 YAML을 읽는다.

주요 성공 로그:

```text
[GRAVITY_CAPTURE] ...
[GRAVITY_SAVED] n=... cond=... mass=... com=... rms=...
[GRAVITY_REMOTE] updated ...
```

`[GRAVITY_REJECTED]`가 나오면 기존 gravity YAML은 유지된다. 대표적인 거부 조건은 자세 방향 rank 부족,
condition number 초과, 비현실적인 질량/CoM, force/torque residual RMS 초과다.

기본 설정에서는 32개 gravity pose를 모두 최종 fit에 쓰지 않는다. 먼저 전체 pose로 1차 fit을 수행해
pose별 force/torque residual을 계산하고, residual이 작은 good-quality pose만 골라 다시 fit한다.
`gravity_quality_min_pose_samples`개 이상이 남고 최종 RMS/condition/mass/CoM 검사를 통과할 때만
`spindle_gravity.yaml`을 저장한다.

생성되는 YAML 행렬:

- `T_AD`: Vive world/raw frame을 robot base frame으로 올리는 base calibration
- `T_BC`: robot EE에서 tracker/tool frame까지의 offset
- `T_FIX`: z-plane residual을 줄이기 위한 left-multiplied rigid correction
- `POSITION_RESIDUAL`: `T_FIX` 뒤에도 남는 xy 위치별 (dx,dy,dz) 오차를 하나로 묶어 보정하는 quadratic_xy 모델
  (예전에는 z만 보정하는 `Z_RESIDUAL`, xy만 보정하는 `XY_RESIDUAL` 두 모델로 나뉘어 있었다. 둘 다 같은
  정규화(center/scale)에 같은 quadratic_xy basis를 쓰는 사실상 같은 모델이었고 굳이 나눠 저장/적용할
  이유가 없어서 하나로 합쳤다.)
- `T_CE`: final constant offset. `T_CE[2,3]` is stored as a positive z correction knob.
- `T_SA`: orientation display/alignment용 right-multiplied rotation correction

## 기본 실행

빌드:

```bash
cd <your_ros2_workspace>
colcon build --packages-select vr_calibration
source install/setup.bash
```

캘리브레이션 실행:

```bash
ros2 run vr_calibration vr_calibration
```

현재 기본값은 다음과 같다.

```text
t_sa_mode = update
t_sa_max_delta_deg = 180.0
capture_hold_time_s = 1.5
capture_min_hold_time_s = 0.8
capture_window_s = 0.5
capture_min_clean_samples = 20
vr_capture_age_s = 0.2
max_capture_sync_dt_s = 0.05
capture_max_vr_std_mm = 10.0
gravity_quality_select_enable = true
gravity_quality_min_pose_samples = 16
gravity_quality_max_pose_samples = 24    # 0 또는 음수면 good-quality pose 전체 사용
gravity_quality_force_residual_max_n = 3.0
gravity_quality_torque_residual_max_nm = 0.35
handeye_outlier_reject_enable = true
handeye_outlier_max_reject = 2
handeye_outlier_abs_mm = 15.0
handeye_outlier_mad_sigma = 4.0
z_fix_enable = true
position_residual_enable = true
position_residual_max_correction_mm = 10.0
max_calib_position_rms_mm = 50.0
```

`T_SA` update는 기본값으로 켜져 있으므로 별도 옵션 없이 실행하면 된다.

## 캡처 로직

노드는 waypoint 파일에서 `holding_time_s > 0`인 point만 target으로 사용한다. 각 target마다 robot이 다음 조건을 만족하면 hold 상태로 들어간다.

- position error <= `pos_enter_mm_`
- orientation error <= `ori_enter_deg_`
- robot linear velocity <= `vel_thresh_mms_`
- robot angular velocity <= `angvel_thresh_dps_`

패치 이후에는 hold가 끝나는 순간의 단일 샘플을 바로 쓰지 않는다. hold 중 다음 조건을 만족하는 clean sample만 buffer에 쌓는다.

- `/ur10skku/currentP`가 fresh
- `/raw_pose`가 `vr_capture_age_s` 이내
- `abs(currentP_time - raw_pose_time) <= max_capture_sync_dt_s`
- robot이 target region 안에 있음
- robot이 stopped 상태임

그 뒤 clean sample이 최소 `capture_min_clean_samples`개 이상이고, buffer 시간 폭이 `capture_window_s` 이상이면
buffer 안에서 가장 안정적인 `capture_window_s` 구간을 골라 평균 pose를 하나 만든다.
best window는 VR position std, robot linear/angular velocity, target dist/angle을 함께 점수화해서 선택한다.

- robot pose: clean sample 평균
- VR position: clean sample 평균
- VR orientation: quaternion sign-align 평균
- VR position std가 `capture_max_vr_std_mm`를 넘으면 캡처를 보류
- target을 떠나는 순간에도 clean buffer가 이미 유효하면 `[OUT_CAPTURE]`로 그 window를 저장하고 다음 target으로 진행한다.
- hand-eye 초벌 solve 뒤 residual이 큰 sample은 `[OUTLIER]`로 최대 `handeye_outlier_max_reject`개까지 제외하고 다시 solve한다.
- `T_FIX` 뒤에도 XY 위치별 (dx,dy,dz) 오차가 남으면 `POSITION_RESIDUAL` quadratic_xy 모델을 저장한다
  (샘플 8개 이상 필요, fit RMS가 개선될 때만 저장). runtime은 `T_FIX` 적용 직후 x,y,z에
  `x += fx(x,y)`, `y += fy(x,y)`, `z += fz(x,y)`를 더하며, `(dx,dy,dz)` 벡터 크기를
  `position_residual_max_correction_mm`로 clamp한다. `T_FIX`는 rigid(기울기+z-offset)만
  보정하므로, hand-eye solve에 남는 매끄러운 비강체 왜곡(트래커 스케일 오차, 추적 볼륨
  비선형성 등)은 이 모델이 없으면 그대로 데이터셋에 남는다.

캡처 로그 예:

```text
[CLEAN_CAPTURE] averaged 42 samples over 0.510s | dist=0.03mm ang=0.01deg vr_std=1.25mm
[CAPTURE] target 12/32 ...
```

## Calibration 계산 순서

캡처된 sample은 내부적으로 다음 의미를 가진다.

```text
T_AB[i] = robot base(A) -> EE(B)
T_DC[i] = VR world(D) -> tracker(C)
```

전체 계산 흐름:

```text
1. clean sample set 수집
2. hand-eye solve로 T_BC 계산
3. 각 sample에서 T_AD_i = T_AB[i] * T_BC * inv(T_DC[i]) 계산
4. T_AD_i 평균으로 T_AD 생성
5. T_FIX 계산
6. POSITION_RESIDUAL 계산 (T_FIX 이후 (dx,dy,dz) 잔차, 하나의 joint 모델)
7. runtime-chain residual 검증
8. YAML 저장
```

과거에는 이 앞에 position-cloud 기반 `R_Adj`(VR/robot point cloud를 Kabsch로 정렬하는 전역 회전 보정)
단계가 있었다. `T_DC_adj[i] = T_Adj * T_DC[i]`처럼 모든 샘플에 동일한 고정 회전을 좌곱하는 형태였는데,
hand-eye solve는 연속/전체 샘플 쌍의 상대운동(`inv(T0)*T1`)만 사용하므로 이 고정 회전은 그 차분에서
정확히 상쇄되고, 이어지는 `T_AD` 평균 단계도 동일한 고정 회전을 정확히 역보정하는 방식으로 흡수한다.
즉 `R_Adj` 값이 무엇이든 최종 calibrated pose(`M_cal`)는 수학적으로 완전히 동일했다 (직접 코드 상에서
치환해 확인: `T_FIX*T_AD*T_Adj*raw*inv(T_BC) = T_FIX*(T_AD_noRadj*inv(T_Adj))*T_Adj*raw*inv(T_BC)
= T_FIX*T_AD_noRadj*raw*inv(T_BC)`). 실제 운영 YAML도 이미 `R_Adj=Identity`였다. 아무 효과가 없는데
파라미터/YAML 행렬/런타임 곱셈만 늘리고 있어 `R_Adj`/`radj_enable`/`radj_sample_count`를 완전히 제거했다.

## Runtime pose 의미

`vr_calibration`은 YAML만 만든다. `/calibrated_pose`를 어떤 의미로 publish할지는 `vive_tracker_ros2/vive_tracker_node.py`의 `tool_correction_mode`가 결정한다.

```text
tool_correction_mode=none
  -> calibrated tracker/world pose publish
  -> EE와 tracker 사이 offset이 position에 남아 있음

tool_correction_mode=t_bc
  -> T_BC inverse를 적용해서 EE/TCP pose publish
  -> robot currentP와 position이 거의 같아지는 것이 정상

tool_correction_mode=t_ce
  -> legacy T_CE offset 사용
```

현재 기본값은 `none`이다. 따라서 아무 인자 없이 `vive_tracker_node`를 실행하면 `/calibrated_pose`는 robot EE pose가 아니라 tracker pose로 나온다. EE/TCP pose가 필요하면 명시적으로 `t_bc`를 켠다.

`apply_T_CE_extra=true`이면 `T_CE`가 최종 단계에서 추가 적용된다. YAML의 `T_CE[2,3]` 값을 `+dz`만큼 키우면 published z가 대략 `dz`만큼 내려간다. `vive_tracker_node`는 YAML 변경을 감지해서 실행 중에도 `T_CE`를 다시 로드한다.

```bash
ros2 run vive_tracker_ros2 vive_tracker_node --ros-args \
  -p tool_correction_mode:=t_bc
```

## 확인 포인트

캘리브레이션이 정상적으로 끝나면 다음 로그를 확인한다.

```text
[HAND_EYE] using all-pairs motions K=... from N=... samples
[HAND_EYE] sign=... fit_rms=...mm alt=...
[T_FIX] rx=... ry=... tz=...mm
[POS_RES] rms ... -> ...mm
[VALID] rms=...mm max=...mm
[YAML_SAVED] ...
```

`T_SA` update를 켠 경우에는 다음 로그가 있어야 한다.

```text
[T_SA_DONE] delta=...deg
[T_SA] pre-capture done
```

다음 로그가 반복되면 clean sample 조건이 너무 빡빡한 것이다.

```text
[WAIT] hold=.../...s n=.../...
[WAIT] vr_std ... > ...mm
```

이 경우 먼저 `/raw_pose` publish rate와 tracking 상태를 확인하고, 필요하면 `max_capture_sync_dt_s`, `capture_min_clean_samples`, `capture_max_vr_std_mm`를 완화한다.


---

<a id="readme-source-11"></a>

# 원본: `datasets/polishing/single_cam/20260909_90deg_only_rel_fzobs/imitation_form/README.md`

# 20260909_90deg_only_rel_fzobs / imitation_form

`datasets/polishing/single_cam/20260821_90deg_only/imitation_form` (left
unchanged) rewritten for the stain task-frame generalization track (see memory
`stain-task-frame-generalization-plan`). Built 2026-09-09.

1. **absolute position -> stain-relative position** (translation only, no
   rotation). `observations/position` and `action/position` columns x,y are
   shifted by the per-episode stain origin; z, rx, ry, rz untouched. The
   original absolute trajectories are kept at `analysis/absolute/*` (not read
   by the dataloader).
2. **force observation = fz only.** `observations/force[:, 0:2]` (fx, fy) set
   to 0. Shape stays (T,3) so the training dataloader (`force[:, :3]`,
   3-wide force_history) needs no change; fx, fy are constant 0 -> degenerate
   after minmax -> no signal. `action/force` is UNCHANGED (observation only).

42/42 episodes, all kept (`--keep_unstable`). 8 episodes had a stain-origin
per-frame spread of 6-13 mm (episode_4 worst at 13.5 mm); see the origin
report. Median spread 3.0 mm, p95 7.2 mm.

## Provenance / reproduce

```bash
# [2] per-episode stain origin (reference-free dark-blob + per-frame homography)
ros2 run stain_relative_frame stain_origin_offline -- \
  --dataset_dir datasets/polishing/single_cam/20260821_90deg_only/imitation_form \
  --method dark --plate_roi 130 58 346 122 --tool_box 258 115 292 122 --dark_thresh 65 \
  --out checkpoints/stain_relative_frame/stain_origin_90deg_only.json

# [3] relativize position  (the converter's out_dir was later moved to the
#     20260909_90deg_only_rel_fzobs/imitation_form path this file sits in)
ros2 run stain_relative_frame dataset_relativize -- \
  --dataset_dir datasets/polishing/single_cam/20260821_90deg_only/imitation_form \
  --out_dir    datasets/polishing/single_cam/20260909_90deg_only_rel_fzobs/imitation_form \
  --origins    checkpoints/stain_relative_frame/stain_origin_90deg_only.json \
  --use_relative_position --keep_unstable --overwrite

# fx, fy -> 0 in observations/force, recompute dataset_stats.pkl
python3 zero_obs_force_xy.py \
  datasets/polishing/single_cam/20260909_90deg_only_rel_fzobs/imitation_form
```

Homography used: `checkpoints/stain_relative_frame/homography.json`
(method=depth_extrinsic, calibrated 2026-09-08).

## Per-episode HDF5 attrs added

- `use_relative_position = 1`
- `relative_transform_version = stain_relative_v1`
- `stain_origin_xy_mm = [x, y]`  (robot base mm, episode constant)
- `rotation_aligned = 0`  (translation only, always)
- `observation_force_xy_zeroed = 1`
- `observation_force_xy_preserved = 0`
- `observation_force_channels = "fz_only (fx,fy zeroed)"`

## dataset_stats.pkl

Recomputed on the converted coordinates. qpos = [pos(6), force(3)];
fx, fy ranges are [0, ~0] (sanitised to +1e-6). x,y ranges are now centred
near 0 (x in [-102, 79], y in [-68, 49]).


---

<a id="readme-source-12"></a>

# 원본: `experiments/e1_force_observation_20260916/README.md`

# E1 force-observation ablation

`FORCE_OBS_ON` supplies measured `observations/force` as `qpos[6:9]` and as the
causal `force_history`. `FORCE_OBS_OFF` keeps the same 9-D interface and model
size but replaces only those measured-force observation channels with zeros
after normalization, for every history row. It does not remove or zero
`action/force`: both policies learn and output the 3-D target force action.

The robot force controller, sensor acquisition, safety monitoring, and logs are
unchanged. The 42 episodes are split by episode with seed 0 (38 train / 4 val);
the loader fits stats on train episodes only. Existing checkpoints/results are
not reused or overwritten. Full training is intentionally not started here.

Measured-force-derived features found in this path: current `qpos[6:9]`, the
30-row `force_history`, and phase-resampling's internal `Fz` used only to select
training start points. The latter is not returned to the policy and remains a
shared sampling rule in both conditions. No force norm/contact feature is fed
to the model.

Run from the repository root:

```bash
bash experiments/e1_force_observation_20260916/run_check.sh
bash experiments/e1_force_observation_20260916/run_smoke.sh
bash experiments/e1_force_observation_20260916/train_off.sh
bash experiments/e1_force_observation_20260916/train_on.sh
```

Outputs go to `experiments/e1_force_observation_20260916/results/` for checks
and to the new `checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/{on,off}`
directories for future full-training results. The scripts only prepare/launch
training; they do not claim a checkpoint exists before training completes.

For hardware-free inference after training, use the existing ROS entrypoint
with the matching checkpoint and launch argument:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  ckpt_dir=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/off \
  use_force_observation:=false
```

The node validates this value against checkpoint metadata. Force sensor input
must remain enabled for control/safety/logging in both cases.


---

<a id="readme-source-13"></a>

# 원본: `experiments/e1_inference_logging_20260917/README.md`

# E1 실행 로그 (2026-09-17)

ON/OFF는 **힘 관측값을 제공해 학습한 체크포인트 / 힘 관측값을 가리고 학습한 체크포인트**를 뜻한다. 각각의 학습 설정에 맞춰 추론에서도 ON은 measured Fx/Fy/Fz history를 정책에 제공하고, OFF는 **정규화 후 qpos[6:9]와 모든 force history를 0**으로 만든다. 둘 다 위치·자세·목표 힘 action을 출력한다. 기준은 성공한 `20260910_90deg_rel_single/20260910_1606`의 90deg 데이터·얼룩 중심 상대 위치·모델 구성이다. 로깅은 정책으로 값을 돌려주지 않는다.

실제 로봇/센서로 검증하지 않았다. 아래 launch는 로봇을 움직일 수 있으므로 운영자가 기존 준비·안전 절차 후 **한 조건씩** 실행한다. agent는 실행하지 않았다.

## 실행

```bash
cd /home/eunseop/nrs_imitation
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source behavior_ws/install/setup.bash
E1_CKPT=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916
read -r -p '동일하게 사용할 실제 설정 RPM (미확인이면 Enter): ' E1_RPM
E1_RPM_ARGS=()
if [[ -n "$E1_RPM" ]]; then
  E1_RPM_ARGS+=("metrics_rpm_setpoint:=$E1_RPM")
fi
```

`E1_RPM`은 **로그 전용 수동 설정값**이며 spindle 명령/실측 RPM이 아니다. 숫자를 추정하지 않는다. 빈 값이면 위 Bash 배열은 RPM 인자를 생략하고, launch 기본값에 따라 null+경고로 기록한다. `metrics_rpm_setpoint:=`처럼 빈 값을 CLI 인자로 직접 전달하면 ROS 2가 `malformed launch argument`로 거부한다. 실측 없이 일정 회전으로 해석할 경우만 `metrics_rpm_assumed_constant:=true`를 명시한다. 기본 false. RPM이 실제로 동일했는지는 운영자가 확인해야 한다.

ON:

```bash
ros2 launch nrs_imitation inference_clean_single_cam.launch.py \
  ckpt_dir:="$E1_CKPT/on/20260916_1531" use_force_observation:=true \
  use_stain_mask:=false stain_canon_enable:=false inference_mode:=service_stream gradcam_enable:=false \
  metrics_log_enable:=true metrics_run_tag:=E1_on_r01 \
  metrics_repeat_id:=r01 \
  "${E1_RPM_ARGS[@]}" \
  overlay_record_enable:=true removal_viz_enable:=false \
  visualize_flow_vector:=false visualize_modality_importance:=false
```

OFF (ON 종료 후):

```bash
ros2 launch nrs_imitation inference_clean_single_cam.launch.py \
  ckpt_dir:="$E1_CKPT/off/20260916_1531" use_force_observation:=false \
  use_stain_mask:=false stain_canon_enable:=false inference_mode:=service_stream gradcam_enable:=false \
  metrics_log_enable:=true metrics_run_tag:=E1_off_r01 \
  metrics_repeat_id:=r01 \
  "${E1_RPM_ARGS[@]}" \
  overlay_record_enable:=true removal_viz_enable:=false \
  visualize_flow_vector:=false visualize_modality_importance:=false
```

2026-09-17 실행 로그 조사 후 위 명령의 실행 모드를 정정했다. 성공 기준인 9월 10일 `20260910_90deg_rel_single` 실행 로그들은 `PTP9D_STREAM_START`(연속 `service_stream`)와 `gradcam_enable=0`을 사용했다. 앞서 안내한 `service_call`은 구간 완료 응답을 기다리는 별도 실행 방식이다. ON/OFF 모두 위 두 옵션을 동일하게 사용한다. 2026-09-20에 polishing single-cam launch와 실행 entrypoint의 Grad-CAM 기본값도 `false`로 맞췄다. `inference_mode:=service_stream`은 위 명령처럼 명시한다.

E1 ON은 같은 데이터·모델 구성으로 재학습한 체크포인트이며 9월 10일 파일과 동일하지 않다. `policy_best.ckpt`의 저장된 epoch는 기존 19 / E1 ON 39이며, 모델 state 336개 중 174개 tensor 값이 다르다. 정규화 수치와 상대좌표 설정은 같다. 기존 성공 체크포인트 재현과 E1 ON/OFF 비교는 실행 태그로 구분한다.

시편 ID는 ON/OFF 모두 기본값 `default`로 기록하므로 지정할 필요 없다. 다음 반복은 `E1_on_r02`/`E1_off_r02`, `r02`. 실시간 뷰어는 기본으로 끄고, `overlay_record_enable:=true`로 flow-vector와 modality 영상을 `~/Videos/Screencasts/*.webm`에 자동 저장한다. 녹화는 두 영상 토픽의 첫 프레임을 모두 받은 뒤 시작하고, 종료 시 파일을 마무리한다. 창을 열지 않아도 녹화되며 영상 생성·인코딩 연산은 계속 수행된다. `removal_viz_enable:=false`는 별도의 Preston 결과 생성을 끈다. 이 옵션들은 로봇 제어 설정이 아니다.

현재 설치는 source-linked이다. 다른 설치에서 새 모듈이 import되지 않으면 `cd behavior_ws` 후 기존 절차대로 `colcon build --symlink-install --packages-select nrs_imitation` 및 `source install/setup.bash`가 필요하다.

## 저장·점검

출력의 `[METRICS] run directory:`를 복사한다. tag가 같아도 timestamp(ns)+임시 디렉터리 난수로 **새 폴더만** 만들며 이전 로그/체크포인트는 덮어쓰지 않는다.

```text
logs/inference_metrics/<tag>_<timestamp_ns>_<unique>/
  metadata.json        # 실제 node parameters/resolved 설정, checkpoint 식별, config
  wrench.csv           # source별 실제 수신 힘/토크, OFF에서도 비제로 원값
  tcp_pose.csv         # 실제 joint-feedback FK TCP; 명령 아님
  commands.csv         # prediction / node_sent / controller_reported_target
  events.jsonl         # inference/phase/reference/send/response/error/drop
  roi.json             # 최초 고정 reference, 초기/종료 image 정보
  legacy.csv           # 기존 비교 CSV 열 + 시간/age; 최신값 snapshot일 뿐
  logger_status.json   # 실행 중 enqueue/write/drop/error
  summary.json         # 종료 drain 결과; 강제 종료시 없어도 raw 파일은 읽을 수 있음
  snapshots/           # 초기·종료 RGB, 기존 mask가 수신된 경우 최초 mask
  artifacts/           # normalizer·소스·기존 calibration/config 사본 및 SHA256
  code_changes.patch   # 작업트리 변경 식별; 미추적 신규 logger도 artifacts에 복사
```

```bash
read -r -p '[METRICS]가 출력한 실행 폴더 절대경로: ' E1_RUN_DIR
watch -n 2 "wc -l '$E1_RUN_DIR/wrench.csv' '$E1_RUN_DIR/tcp_pose.csv' '$E1_RUN_DIR/commands.csv'; cat '$E1_RUN_DIR/logger_status.json'"
# 종료 후 (ROS 환경 불필요)
/usr/bin/python3 scripts/check_inference_log.py "$E1_RUN_DIR"
```

검사기는 source별 rate/median·maximum gap, 결측, NaN/Inf, 시각 역행, source stamp 반복, drop/error, run_id 일치를 보고한다. **수신 이벤트 수 ≠ 센서 취득 횟수**(`/currentF`는 재발행). middleware 손실은 unknown이다. gap/stale 합격 기준은 기본적으로 설정하지 않는다. 실제 관측 주기와 실험 허용오차를 정한 뒤 `--max-gap-ms`(수신 시각 기준, 모든 source 공통), `--max-age-ms`(legacy cached 값의 동일 host monotonic age)에 명시한다. 보간·삭제하지 않는다. 종료 코드 2는 로그 오류/명시 기준 위반이며, 0도 Preston 분석 준비 완료를 뜻하지 않는다.

주기가 다른 source는 `--source-max-gap-ms SOURCE=MS`를 반복 지정해 각각 판단할 수 있다(예: SOURCE는 실제 `/ur10skku/ftdata`, MS는 운영자가 정한 허용값). 원본 acquisition age를 모르는 wrench/pose의 stale 판정은 여전히 null이다.

## Offline 테스트·예시

```bash
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source behavior_ws/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/behavior_ws/src/nrs_imitation:$PWD/behavior_ws/src/stain_relative_frame:${PYTHONPATH}" \
  /usr/bin/python3 -m pytest -q behavior_ws/src/nrs_imitation/test/test_execution_metrics.py
/usr/bin/python3 scripts/make_inference_log_example.py --force-observation OFF
/usr/bin/python3 scripts/make_inference_log_example.py --force-observation ON
```

예시의 `SYNTHETIC_*`는 실제 실행·가공 결과가 아니다. 테스트는 실제 callback·정규화·history masking·Flow sampler·전송 계산을 작은 untrained ResNet Flow로 실행하며 ROS transport만 mock한다. 생산 DINOv3 checkpoint의 하드웨어 실행/실시간 주기 검증은 아니다. pytest의 설치된 외부 plugin 버전 충돌 때문에 autoload를 끈다.

추가 로그 전용 옵션: `metrics_queue_size`(양수, 기본 8192), `metrics_snapshot_enable`(기본 true), `metrics_context_file`(JSON 문서의 수동 calibration/시편/공구 메모를 `operator_context`에 보존; 제어/변환에 적용하지 않음). 유효 접촉 면적/반경, 외경, Preston 계수, 현재 표면 법선·압축력 부호, 접촉점 offset, 회전 상태는 확인 전 null이다. 수동 자료는 출처/단위/좌표계/측정 날짜를 함께 남긴다.

상세 경로·schema·미확인 사항: [DATA_PATHS.md](DATA_PATHS.md). 테스트 결과: [TEST_RESULTS.md](TEST_RESULTS.md).


---

<a id="readme-source-14"></a>

# 원본: `experiments/e1_rt_common_20260927/README.md`

# E1 R/T 재실행 — 2026-09-27

어제 E1 C 5회의 설정과 저장된 실행기 코드를 기준으로 R/T에 같은 위치·자세 후처리를 적용했다.
아래 명령의 **새 config 경로**를 사용한다. 예전 config 경로는 예전 R/T 동작을 유지한다.
어제 C 결과와 기존 R/T 기록은 보존한다. 새 R/T와 어제 C를 비교 대상으로 묶으며, 예전 R/T는 중복 표본으로 합치지 않는다.

| 항목 | 오늘 R/T에 적용한 값 |
|---|---|
| 위치/자세 평활 | C와 같은 35점 XYZ 이동평균·quaternion 평균 |
| 평활용 시간격자 | 30 Hz, 128점 구간, 120점마다 다음 구간 연결 |
| 시작/구간 연결 | C와 같은 0.5초 quintic 연결 |
| 축별 가속도 | 위치 25 mm/s², 회전 100 deg/s² |
| 공통 명령 발행 | 125 Hz; gain 15/s |
| 공통 속도·힘 변화율 | 위치 10 mm/s, 회전 40 deg/s, 힘 30 N/s |
| 공통 접촉 판정 | ON 3 N / OFF 1.2 N |
| R | 기존 +18 N recipe, 편도 1회; 가공 후 기존 home 복귀 |
| T | 기존 episode_29의 시간·힘·위상 보존 |

R은 원본 약 125 Hz, T는 원본 약 30 Hz 교시 시각이다. 둘 다 힘과 종료 시각은 원본에서 읽고,
위치·자세만 C와 같은 30 Hz 구간으로 처리한다. 마지막 평활 창의 끝점 패딩은 실행시간을 늘리지 않는다.
C 정책·모델·힘 관측·접촉 판정·실행 결과는 수정하지 않았다.

## 실행

기존 로봇/카메라 환경과 시편 준비는 어제와 동일하게 한다. 한 번에 한 방법만 실행한다.

```bash
conda activate nrs_imitation
source /opt/ros/humble/setup.bash
source ~/dev_ws/install/setup.bash
source ~/nrs_imitation/behavior_ws/install/setup.bash

# R — 한 번 실행
ros2 launch nrs_imitation rtc_timed.launch.py method:=R mode:=run \
  config:=/home/eunseop/nrs_imitation/experiments/e1_rt_common_20260927/config.json

# T — 한 번 실행
ros2 launch nrs_imitation rtc_timed.launch.py method:=T mode:=run episode:=episode_29 \
  config:=/home/eunseop/nrs_imitation/experiments/e1_rt_common_20260927/config.json
```

각 명령을 필요한 횟수만큼 별도로 실행한다. 실행 ID는 `RTC_R_20260927T...` 또는
`RTC_T_20260927T...`로 자동 생성된다. 같은 초에 연달아 실행하는 방식은 사용하지 않는다.
콘솔의 `Shared E1 R/T/C profile; pose grid=30 Hz; reference=C 2026-09-26` 표시를 확인한다.
`mode:=check`로 바꾸면 로봇 노드를 시작하지 않는 설정 검사다.

R은 유한 작업 후 기존 접촉 해제/home 복귀 절차, T는 교시 시간 끝에서 자동 정지를 요청한다.
정지 확인 후 Ctrl+C로 기록을 마감한다. 중단이 필요하면 기존 명령을 쓴다.

```bash
ros2 service call /e2_executor/abort std_srvs/srv/Trigger '{}'
```

작업자 정상 종료는 `/e2_executor/finish`, 가공 구간 표시는
`/e2_executor/processing_start`와 `/e2_executor/processing_end`를 같은 Trigger 형식으로 사용한다.
processing_end 자체는 정지 명령이 아니다. R/T 자동 종료를 기다리면 평소 finish 호출은 필요 없다.

CSV·이벤트는 `logs/inference_metrics`, 영상은 `logs/polishing_removal`, 실행 설정은
`logs/rtc_launch_context`에 새 ID로 저장된다. 실제 실행 후 검사·정리할 때 오늘 날짜의 결과로 구분한다.

## 확인된 비교 범위

공통 위치·자세 후처리와 전달 경로를 맞췄다. R의 규칙 경로, T의 고정 교시,
C의 관측 기반 재계획은 각 방법의 차이로 남는다. R의 가공 후 복귀 시간은 가공 성능 비교에서 분리한다.
서로 다른 날의 실제 시편 상태·장비 응답까지 자동으로 같다고 증명한 것은 아니다.

T는 원래도 마지막 복귀 구간이 공통 속도 제한보다 빨라, 교시 시간 종료가 마지막 위치 도착을 뜻하지 않는다.
동일한 입력의 명령 계산에서 종료 위치 오차는 기존 15.09 mm, 패치 후 13.29 mm였다.
이는 로봇 실측 오차가 아니며, 도착하지 않은 실행을 도착 완료로 기록하도록 바꾸지 않았다.
힘 게이트 전환이나 실측 접촉 응답은 실행 후 기록으로 확인한다.

검증: 168개 회귀 검사 통과, R/T 실제 ROS launch `mode:=check` 통과.
어제 C 5회 입력 9,662틱에서 당시 코드와 현재 코드의 요청·후처리·접촉 게이트·발행 9차원 명령이
모두 바이트 단위로 일치했다. R/T의 원래 힘 값·게이트·힘 발행·시간축도 회귀 검사로 확인했다.
에이전트가 실제 로봇을 실행하지 않았다.

상세 검증과 변경 내역: [패치 보고서](../../reports/20260927_E1_RT_common_patch/README.md).


---

<a id="readme-source-15"></a>

# 원본: `reports/20260926_C5_log_audit/README.md`

2026-09-26 C 5회 확인 및 비정상 로그 정리
======================================

패치 후 C 5회 모두 기록 검사를 통과했고, 작업자의 `operator_finish` 요청 이후 `normal_completion`, `controller_hold_verified=true`, `queue_cancel_verified=true`가 확인됐다. 사용자는 다섯 실행에서 의도하지 않은 동작이 없었다고 확인했다. 이 5회는 보존하고, 실행 시작 전 중단 2회와 패치 전 진동으로 중단한 1회의 로그를 삭제했다.

현재 오늘의 보존 실행은 **R 5회 + T 5회 + C 5회 = 15회**다. R/T의 기록과 설정은 변경하지 않았다.

보존된 C 실행
------------

| 실행 태그 시각 | 실행 시간(초) | TCP/힘 행 | 실제 발행 명령 | 계획 수 | 영상(초) | 결과 |
|---|---:|---:|---:|---:|---:|---|
| 18:43:57 | 16.937 | 445 / 445 | 2,117 | 5 | 17.8 | 정상 기록·종료·정지 확인 |
| 18:45:21 | 13.635 | 399 / 399 | 1,704 | 4 | 14.9 | 정상 기록·종료·정지 확인 |
| 18:46:22 | 16.030 | 436 / 436 | 2,003 | 4 | 16.7 | 정상 기록·종료·정지 확인 |
| 18:47:24 | 15.216 | 413 / 413 | 1,901 | 4 | 16.0 | 정상 기록·종료·정지 확인 |
| 18:48:20 | 15.504 | 423 / 423 | 1,937 | 4 | 16.3 | 정상 기록·종료·정지 확인 |

검사 내용
---------

각 실행의 provider/executor 세션 ID가 일치하며 다섯 세션은 서로 다르다. 실행 설정 해시는 다섯 실행에서 동일하고, `c_pose_conditioning_20260926_v1` 및 패치된 코드 해시도 일치한다. CSV 행 구조·유한값·시간 순서·summary 기록 수를 검증했다. 로거 drop, 쓰기 오류, 종료 시 미처리 큐는 모두 0이며 기록이 완료됐다.

C 명령의 네 단계 `time_sampled` → `pose_conditioned` → `contact_gated` → `node_sent`가 빠짐없이 대응한다. 각 예측 계획은 원본 및 후처리 단계에 128점씩 기록됐다. 명령 CSV 전체 행 수는 실제 발행 횟수의 4배다.

다섯 실행의 최대 명령 발행 기록 간격은 8.694 ms였다. 위치 속도는 축별 10 mm/s, 위치 명령 가속도는 축별 25 mm/s² 제한을 지켰고, 회전 속도·가속도 및 힘 변화율 제한도 통과했다. 실행 중 TCP/힘 관측의 최대 공백은 약 138.876 ms로 기존 200 ms freshness 범위 안이었다. 이는 로거 검사 결과이며 헤더 없는 원센서·미들웨어 손실량까지 확인한 것은 아니다.

영상 5개를 전체 디코딩했고 오류가 없었다. 그래프 20개와 시작/종료 이미지 10개도 읽기 검사를 통과했다. 각 launch의 5개 노드가 모두 clean exit로 종료됐다. provider의 `run_interrupted`는 executor 정상 종료 확인 후 Ctrl-C로 기록 프로세스를 마감한 것으로, 가공 도중 중단된 실패 실행과 구분했다.

접촉 판정 전환은 원본 그대로 보존했다. 첫 4초 전환 횟수는 순서대로 0, 58, 40, 4, 8회이며, 사용자의 관찰 확인을 반영해 이 값만으로 실행을 제외하지 않았다. 실측 base-frame Fz 최대치는 순서대로 84.411, 51.789, 65.338, 81.195, 82.984 N이었다. 이 지표들은 모델/실행 동작 분석용이며, 이번 기록 검사에서 별도 원인 분리를 수행한 것은 아니다.

삭제 내역
---------

| 제외 태그 | 이유 |
|---|---|
| `RTC_C_20260926T175117` | 실행 시작 전 수동 중단; 추종 명령 없음 |
| `RTC_C_20260926T175131` | 패치 전 심한 진동을 보고한 실행; 수동 중단 |
| `RTC_C_20260926T184718` | 실행 시작 전 수동 중단; 추종 명령 없음 |

위 3회에 직접 연결된 metrics 폴더 4개, 실행 context 3개, 그래프 폴더 3개, 영상 1개, ROS launch 폴더 3개와 해당 노드 로그 15개를 삭제했다. 합계 **29개 경로, 109개 파일, 14,450,113 bytes**다. 다른 날짜의 실험과 공유 원격 제어기 로그는 삭제 범위에 포함하지 않았다.

삭제 직전 파일 목록·크기·inode·수정 시각·SHA-256을 대조했다. 삭제 후 C 보존 파일 285개와 기존 R/T 파일 580개의 SHA-256이 삭제 전과 같음을 확인했다. 총 865개 보존 파일 검증을 통과했다.

기록 해석과 자료 위치
--------------------

여기서 정상 종료는 작업자 종료 요청 후 제어기의 정지 유지가 확인됐다는 의미다. C에는 자동 접촉 해제·홈 복귀 완료가 기록되지 않으며, 독립적인 가공 품질/제거량 성공 판정은 별도다. 작업자 `processing_start/end` 마커도 없어 표의 실행 시간을 순수 가공 시간으로 해석하지 않는다.

기존 진동 진단과 패치 분석 보고서는 보존했다. 사용자 요청에 따라 해당 보고서의 입력이었던 패치 전 C 원본 로그를 삭제했으므로, 그 과거 재생 스크립트는 삭제된 입력 없이 재실행할 수 없다. 기존 분석 결과와 원본 해시·이번 삭제 이력은 남아 있다.

- [보존 실행 목록·로그 경로 CSV](retained_runs.csv)
- [전체 검사·삭제 목록·보존 해시](cleanup_manifest.json)
- [읽기 전용 검사 및 삭제 목록 생성 스크립트](audit.py)
- [정확한 삭제 목록 적용 스크립트](cleanup.py)


---

<a id="readme-source-16"></a>

# 원본: `reports/20260926_C_execution_patch/README.md`

2026-09-26 C 실행 패치
====================

C에만 궤적 평활, 재계획 연결, 위치/회전 가속도 제한을 적용했다. 사용자가 오늘 측정한 R/T를 유지하도록 요청했으므로 R/T의 경로와 실행 조건은 보존했다. 기존 C 실행 명령에서 새 패치가 로드되는 것을 무구동 check 모드로 확인했다. 실제 로봇은 실행하지 않았으며 진동 해소와 힘 응답의 하드웨어 검증은 남아 있다.

적용 내용
---------

현재 [config.json](../../experiments/e2_rule_replay_20260920/config.json)의 `il.pose_conditioning`에 `c_pose_conditioning_20260926_v1`을 추가했다. `common`, `executor`, `recipe`, `replay`와 기존 `il` 항목은 수정 전 및 오늘 실행 로그의 설정과 일치한다. 공유 실행기의 `execution_method=il`에서만 새 처리를 활성화한다.

| C 전용 처리 | 적용 값/동작 |
|---|---|
| 위치 평활 | 기존 서비스 C의 35점 중심 이동평균과 같은 XYZ 처리; 예측 구간 내부에서 계산 |
| 자세 평활 | 같은 35점의 quaternion 평균; 회전벡터 ±π 경계에서 잘못된 회전 방지 |
| 최초 연결 | 실제 시작 자세에서 새 목표까지 0.5초 quintic 가중치로 연결 |
| 재계획 연결 | 이전 기준 궤적의 위치·속도에서 0.5초 동안 새 기준 궤적으로 연결; 연결 도중 새 계획이 와도 연속 상태 유지 |
| 위치 명령 가속도 | 축별 25 mm/s², 정지 상태에서 기존 10 mm/s까지 최소 0.4초 |
| 회전 명령 가속도 | 축별 100 deg/s², 기존 축별 40 deg/s 속도 한계 유지 |
| 작업영역 검사 | 원본 목표, 평활·연결한 목표, 가속도 제한 후 발행할 명령 모두 기존 공통 범위 검사 |

공통 125 Hz 발행, gain 15/s, 축별 위치 속도 10 mm/s, 회전 속도 40 deg/s, 힘 변화율 30 N/s를 유지했다. 힘 목표의 부호·값·시간축, 접촉 ON/OFF 3.0/1.2 N, 힘 클리핑 설정, 피드백 freshness/watchdog와 정지 확인 절차도 유지했다. 학습 모델·정규화·force observation은 변경하지 않았다.

C의 원본 30 Hz/128점 시간축과 재계획 주기를 보존한다. 이번 패치는 원본 시간축을 늘리는 거리 기반 retiming을 추가하지 않는다. 원본 예측은 그대로 기록하고, 발행 전에 C의 위치/자세 명령만 안정화한다. 가속도 제한에 따른 추가 추종 지연과 경로 차이는 C 처리의 일부다.

검증 결과
---------

관련 테스트 93개가 통과했다. C 평활·연결·속도/가속도 제한, 반복 재계획, 정지 목표 수렴, 회전 경계, 힘 처리 불변, 작업영역/피드백 중단, R/T 유한 시간축, R 복귀, ROS launch 인자 전달과 기록을 검사했다. 노드나 프로세스를 실행하지 않는 launch 확장 테스트도 포함한다.

오늘 R 5회와 T 5회의 저장된 계획·시간·접촉 상태를 이전 코드와 새 코드에 동일하게 입력했다. R은 회당 4,085틱, T는 회당 1,864틱으로 **총 29,745틱의 원본/게이트 후/발행 9차원 명령이 바이트 단위로 동일**했다. R의 별도 복귀 구현은 그대로이며 기존 복귀 테스트도 통과했다. 원본 로그를 수정하거나 삭제하지 않았다.

진동 C `RTC_C_20260926T175131`의 1,776틱 입력 재생 결과:

| 항목 | 패치 전 | 패치 후 |
|---|---:|---:|
| 첫 4초 X 방향 전환 | 13회 | 1회 |
| 첫 4초 Y 방향 전환 | 12회 | 2회 |
| 최대 XYZ 축별 명령 가속도 | 약 2,500 mm/s² | 25 mm/s² |
| 재계획 접수 시 기준 위치의 불연속 | 원본 요구 궤적에서 최대 33.3 mm 변화가 관측됨 | 세 경계 모두 0 mm |
| 축별 위치 속도 상한 | 10 mm/s | 10 mm/s |
| 힘 목표·게이트·힘 발행 결과 | 기준 | 동일 |

방향 전환은 속도가 +1 mm/s와 -1 mm/s를 오갈 때 집계했다. 영점 부근의 작은 변화는 제외한다. 이전 진단의 X 10회/Y 8회는 인접 두 틱에서 바로 반전한 횟수이므로 위의 전체 방향 전환 횟수와 정의가 다르다. 그래프는 실제 로봇 가속도나 패치 후 센서 데이터가 아닌, 계산된 명령을 비교한다. 경계의 0 mm 역시 기준 궤적의 연속성 검사다.

수정 후 실행기 연산 시간은 로컬 오프라인 재생에서 틱 p99 약 0.20 ms, 새 계획 준비 약 0.46 ms였다. 이 수치는 실제 ROS 부하나 원격 전송의 시간 보장이 아니다. 설치된 Python 모듈과 launch는 수정한 소스에 연결된 symlink임을 확인했으며, 설치 환경의 C `mode:=check`에서 새 프로필 이름과 검사 통과가 출력됐다. 추가 패키지 설치나 제어 PC 변경은 필요하지 않았다.

남은 확인과 비교 해석
--------------------

접촉 판정과 힘 제어 조건은 R/T 기준을 유지하기 위해 바꾸지 않았다. 오프라인 재생에는 기록된 접촉 상태를 그대로 입력했으므로, 진동 분석에서 보인 접촉 게이트 반복 전환과 큰 실측 힘 피크가 해소됐다는 뜻은 아니다. 자세 명령이 달라졌을 때 실제 접촉과 힘 응답이 어떻게 바뀌는지는 첫 C 검증 실행에서 확인해야 한다. 첫 실행부터 정상 데이터로 확정하지 말고, 반복 진동이 나타나면 기존 중단 절차로 종료한다.

R/T와 C는 같은 전달 방식·공통 제어 상수를 쓰지만, **C에만 추가한 평활·연결·가속도 제한은 비교 방법에 명시해야 한다.** 기존 R/T를 보존한 채 C의 전체 구성 성능을 비교할 수 있으며, 차이를 모델 자체의 효과만으로 분리해 해석할 수는 없다. `execution_equivalence=unverified`는 유지했다. 기존 R/T의 완료 상태를 이번 패치로 재분류하지 않았다.

실행
----

기존 ROS 환경 터미널에서 같은 C 명령을 사용한다. 콘솔에 `C pose conditioning: c_pose_conditioning_20260926_v1`이 표시된다. `mode:=check`는 노드를 시작하지 않는 검사이며, 아래 `mode:=run`은 사용자가 실행하는 로봇 구동 명령이다.

```sh
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash

ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=C \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=run \
  checkpoint:=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/on/20260916_1531/policy_best.ckpt
```

R/T 명령은 기존대로다. 요청에 따라 R/T 재측정이나 자동 로봇 구동은 수행하지 않았다. 기존 가공 구간 표시·finish·abort 사용법은 [실행 안내](../../experiments/rtc_launch_20260926/README.md)를 따른다.

로그와 재현
-----------

C executor metadata/`plan_received` 이벤트에 `c_pose_conditioning`을 기록하고, 명령 CSV에 `pose_conditioned` 단계를 추가했다. C는 `time_sampled` → `pose_conditioned` → `contact_gated` → `node_sent` 네 단계다. R/T는 기존 세 단계를 유지한다. C 명령 행의 details에는 `conditioning_profile`이 기록된다.

- [명령 비교 그림](command_comparison.png), [PDF](command_comparison.pdf)
- [재생 수치와 원본 파일 해시](recorded_validation.json)
- [검증 요약 및 적용 파일 해시](validation.json), [패치 diff](changes.patch)
- [오프라인 재현 스크립트](validate_recorded.py): `OPENBLAS_NUM_THREADS=2 /usr/bin/python3 reports/20260926_C_execution_patch/validate_recorded.py`
- [수정 전 파일 해시](before_sha256.json), [수정 전 소스·설정](before/)
- [C 회귀 테스트](../../behavior_ws/src/nrs_imitation/test/test_c_pose_conditioning.py)

후속 기록: 패치 후 사용자가 C 5회를 실행했고, [C5 검사 및 정리](../20260926_C5_log_audit/README.md)에서 기록 완료·작업자 종료·정지 유지를 확인했다. 사용자 요청으로 패치 전 비정상 C 원본 입력 로그는 삭제했다. 위 오프라인 비교 결과와 당시 원본 해시는 보존했지만, `validate_recorded.py`는 삭제된 과거 입력 없이 재실행할 수 없다. C의 접촉 판정 전환은 분석 지표로 유지했다.


---

<a id="readme-source-17"></a>

# 원본: `reports/20260926_C_execution_patch/before/experiments/rtc_launch_20260926/README.md`

# R/T/C를 ros2 launch로 실행

`rtc_timed.launch.py`는 기존 `e2_timed_topic_v1` 실행기의 직접 launch 진입점이다.
기존 shell 실행 스크립트 없이 method=R/T/C를 선택할 수 있다.
공통 service·controller interpolation을 사용하는 새 E1 실행기는 아직 구현하지 않았다.
이 파일 추가로 실행 동등성·힘 법선 보정·recipe validation이 완료되는 것은 아니다.

현재 프로젝트에서 패키지 빌드·설치를 마쳤다. 각 터미널에서 ROS 환경을 읽는다.

```sh
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash
```

**점검: 로봇·추론·카메라 노드를 시작하지 않는다**

```sh
ros2 launch nrs_imitation rtc_timed.launch.py --show-args

ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=R \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=check
```

`mode`를 생략해도 check이다. config는 필수이고 자동 선택하지 않는다.
T는 `episode`, C는 `checkpoint`까지 명시해야 check/run이 통과한다.
검사 실패는 노드를 시작하기 전에 종료한다. 기존 preflight의 통과는 하드웨어 또는 새 E1 준비 완료 판정이 아니다.

**기존 timed 실행기 수동 실행 — 한 번에 한 방법**

다음 명령은 현재 config를 사용하기로 선택하는 경우의 예시다.
R은 기존 18 N, peak 10 mm/s, 직선 편도 1회와 접촉 해제·교시 시작 자세 복귀 설정이다.
T는 episode_29, C는 명시된 9/16 ON1531 checkpoint를 선택하는 명령이다.
새 실험의 recipe/교시/model을 자동으로 확정한 것은 아니다. 다른 선택은 검토한 config와 launch 인자가 함께 일치해야 한다.
현재 config의 RPM은 unknown이며 표면 법선·배포 버전 확인도 별도이다.

```sh
# R
ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=R \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=run

# T — episode_29를 선택하는 경우
ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=T \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  episode:=episode_29 mode:=run

# C — 아래 9/16 ON checkpoint를 선택하는 경우
ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=C \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=run \
  checkpoint:=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/on/20260916_1531/policy_best.ckpt
```

먼저 같은 명령에서 `mode:=run`을 `mode:=check`로 바꿔 점검할 수 있다.
원하는 로그 태그는 `run_tag:=RTC_R_r01`처럼 추가한다. 생략하면 방법과 시각으로 생성한다.
기존 로봇 driver·제어기·카메라 실행 환경은 사용자가 준비해야 한다. 이 launch는 스핀들을 켜지 않는다.
세 방법은 각각 독립 작업 영역에서 실행하며 동일 영역을 순차 가공한 결과를 독립 비교로 세지 않는다.

**가공 구간 기록·종료·중단**

다른 ROS 환경 터미널에서 기존 실행기의 실제 service를 사용한다.

```sh
ros2 service call /e2_executor/processing_start std_srvs/srv/Trigger '{}'
ros2 service call /e2_executor/processing_end std_srvs/srv/Trigger '{}'

# 작업자가 정상 종료를 요청할 때, 특히 C
ros2 service call /e2_executor/finish std_srvs/srv/Trigger '{}'

# 수동 중단
ros2 service call /e2_executor/abort std_srvs/srv/Trigger '{}'
```

R/T는 유한 궤적 끝에서 자동 정지를 요청한다. R만 기존 설정에 따라 home 복귀를 포함한다.
정지 상태를 확인한 뒤 launch 터미널에서 Ctrl-C로 기록을 마감한다.
실행 도중 Ctrl-C는 manual_abort이며 service ACK 자체는 물리 정지 확인이 아니다.
기존 제어기는 통신 단절 시 물리 정지를 보장하는 장치가 아니므로 기존 현장 중단 절차를 따른다.

원본 config를 덮어쓰지 않는다. `mode:=run`은 `logs/rtc_launch_context/RTC_<method>_*/config.json`에
실행용 복사본과 paper_experiment=E1, 구번호 source config 경로/hash, 명시 선택 인자,
execution_equivalence=unverified를 기록한다. provider/executor는 이 동일 복사본을 사용한다.
CSV/event는 `logs/inference_metrics`, 영상은 기존 `logs/polishing_removal`에 남는다.

**검증 범위**

선택 누락·OFF/설정 불일치 차단, check 모드 무노드·무파일, R/T/C 전달 인자,
run config 복사와 원본 보존을 노드 실행 없이 검사했다.
사용자의 16:01 실행에서 발견된 하위 launch의 `config` 인자 충돌을 수정했다.
RTC 진입점 전용 인자는 그룹 안에서 제거하고, 실험 JSON은 `e2_config`로 전달한다.
원점 감지 노드는 자기 패키지의 `stain_relative_frame.yaml`을 읽고, 동결된 감지 설정은 기존 `detect_params`로 받는다.
실제 하위 launch를 노드 실행 직전까지 펼쳐 파라미터를 평가하는 R/T/C 회귀 검사도 추가했다.
수정 후 총 12개 검사를 통과했다. 에이전트는 실제 run launch·service 호출·로봇·스핀들을 실행하지 않았다.
하위 힘 제어·보간·게인·안전한계는 변경하지 않았다. 새 E1 준비의 남은 사항은
[사전 감사](/home/eunseop/nrs_imitation/reports/20260926_e1_rtc_audit/README.md)를 참조한다.


---

<a id="readme-source-18"></a>

# 원본: `reports/20260926_C_vibration/README.md`

2026-09-26 C 진동 진단
=====================

현재 C의 가장 유력한 진동 원인은, 기존 C 서비스 실행에 있던 궤적 평활·거리/속도 기반 시간 조정을 우회하면서 거친 예측 궤적과 재계획 경계의 불연속을 추종하는 실행 방식이다. 첫 4초부터 실제 발행 위치 명령의 X/Y 진행 방향이 반복 반전했다. 힘 명령을 허용하는 접촉 판정도 반복 전환되어 하위 힘 제어와의 상호작용이 추가 기여했을 가능성이 있다. 코드 차이와 명령의 반복 반전은 확인했지만, 기록만으로 단일한 물리 원인을 확정하지는 않는다.

9/20에도 같은 문제를 분석한 [기존 진단](../20260920_e2_c_vibration/diagnosis.md)이 있었으며, 당시 진동 C와 이번 C의 정책 체크포인트·정규화·정책 실행 설정·공통 실행기 설정이 같다. 관련 Python 실행 코드 해시도 일치한다. 앞서 이 미해결 이력을 확인하지 않고 현재 설정으로 실행을 이어가도 된다고 안내한 판단은 부정확했다. 수정과 오프라인 검증 전에는 C 재실행을 보류해야 한다.

조사 대상과 종료 상태
--------------------

- 대상: `RTC_C_20260926T175131`, 17:51:39.610 실행 시작, 17:51:53.822 수동 중단 요청. 실행 약 14.212초, 4개 계획, 실제 발행 명령 1,776개.
- 실행기: [executor 원본 로그](../../logs/inference_metrics/RTC_C_20260926T175131_executor_20260926T175131_1790412691920690153_40pdakyf/).
- 계획 제공자: [provider 원본 로그](../../logs/inference_metrics/RTC_C_20260926T175131_20260926T175134_1790412694442221395_2dzi0tlx/).
- 17:51:54.923에 `manual_abort`, `controller_hold_verified=true`, `queue_cancel_verified=true`가 기록됐다. 이는 제어기의 정지 유지 확인이며, 공구의 접촉 해제나 스핀들 정지 확인은 아니다.
- 직전 `RTC_C_20260926T175117`은 계획/추종 명령 발행 전 중단된 별도 시도다.
- 실행기와 계획 제공자 로그는 모두 쓰기 오류·큐 유실 0, 종료 시 기록 완료 상태다. 이번 C는 정상 완료 실험으로 분류할 수 없다.

확인한 실행 차이
----------------

| 항목 | 기존 서비스 C | 현재 공통 timed executor의 C |
|---|---|---|
| 위치/자세 예측 처리 | 35점 이동평균 후 경로 큐에 추가 | 시간 보간 후 공통 gain/rate 제한 |
| 경로 진행 | 거리/목표 속도에 따른 구간 시간, 큐 끝 부근에 이어 붙임 | 30 Hz 예측의 고정 시간축, 새 계획으로 교체 |
| 힘 목표 선택 | 측정 위치에 가까운 예측점의 힘을 별도 서비스로 갱신 | 위치와 같은 경과 시간의 힘을 선택 |
| 공통으로 남은 제한 | gain 및 축별 위치/회전/힘 변화율 제한 | 같은 제한 적용; 추가 가속도/jerk 제한 없음 |

현재 R/T/C가 같은 실행기를 사용한다는 설명 자체는 맞지만, 그것만으로 기존 C의 진동 억제 기능까지 유지되거나 실제 접촉 작업이 안정적이라고 볼 수 없다. 체크포인트·정규화·정책 실행 설정은 기존 서비스 C와도 일치하므로, 다른 모델을 잘못 불러온 정황은 없다.

수치 근거
---------

| 이번 C의 기록 | 값 | 해석 |
|---|---:|---|
| 첫 예측 계획의 인접 XYZ 간격 | 중앙값 4.896 mm, p95 9.621 mm, 최대 14.506 mm | 30 Hz 입력에 점간 변화가 큼 |
| 같은 계획에 기존 35점 평활을 오프라인 적용 | 중앙값 0.617 mm, p95 1.171 mm, 최대 1.263 mm | 평활의 효과를 보여 주는 수학적 비교; 하드웨어 수정 검증은 아님 |
| 첫 4초의 발행 명령 방향 반전 | X 10회, Y 8회, Z 0회 | 인접 속도가 서로 반대이며 양쪽 모두 절댓값 1 mm/s 초과인 경우 집계 |
| 첫 4초 접촉 게이트 전환 | 23회 | 실제 표면 접촉 횟수가 아니라 힘 명령 허용 상태 전환 횟수 |
| 전체 접촉 게이트 전환 | 38회 | 접촉 상태가 반복 바뀜 |
| 발행 Fz 목표 범위 | -4.285 ~ 25.569 N | 게이트와 부호 있는 힘 목표가 하위 제어에 입력됨 |
| 실측 base-frame Fz 최대 | 127.250 N, 실행 후 4.700초 | 큰 힘 피크가 관측됨; 진동의 단독 원인 증거는 아님 |
| 발행 명령 기록의 최대 간격 | 8.408 ms | 약 125 Hz 발행 지속, 기록된 deadline fault 없음 |

재계획 경계에서 시간 보간한 요구 XYZ가 다음과 같이 변했다. 이는 요구 궤적의 불연속이며 로봇이 그 거리만큼 순간 이동했다는 뜻이 아니다.

| 실행 경과 | 계획 교체 | 요구 XYZ 변화 | 해당 틱 실제 발행 XYZ 변화 |
|---|---|---:|---:|
| 4.096초 | 1 → 2 | 20.560 mm | 0.139 mm |
| 8.304초 | 2 → 3 | 33.257 mm | 0.139 mm |
| 12.496초 | 3 → 4 | 14.721 mm | 0.113 mm |

첫 재계획 전인 4초 안에도 방향 반전이 나타나므로, 재계획 경계만을 원인으로 볼 수는 없다. 속도 제한이 한 틱의 이동량은 줄이지만, 잡음에 따른 반복 반전이나 새 목표로 향하는 속도 변화까지 평활하게 만드는 구조는 아니다.

비교용으로 오늘 보존된 첫 T와 첫 R을 같은 방식으로 분석했다. 첫 4초 X/Y/Z 명령 방향 반전은 두 실행 모두 0회였다. 접촉 게이트 전환은 T 10회, R 0회였다. 이는 일부 실행의 기술적 비교이며, 전체 R/T/C 실험의 통계적 결과는 아니다. T에서도 첫 14초 내 base Fz 116.486 N이 기록되어, C의 힘 피크나 음수 목표 하나만으로 C에만 있는 원인이라고 결론 내리지 않는다.

하위 힘 제어와 통신 확인
-----------------------

제어 PC에서 읽은 설치 설정은 `CONTROL_PERIOD=0.008`, TCP 힘 좌표계, `Force_Con_Mode=3`이었다. 하위 힘 제어 소스에는 목표 힘 절댓값 0.01 N과 실측 힘 절댓값 1.50 N에 따른 상태 분기, 목표만 활성일 때 부호를 유지한 15 N precontact 입력이 있다. 상위 게이트가 반복 전환하면 이 분기와 상호작용할 수 있다. 다만 하위 제어기의 매 틱 분기·실제 적용 힘은 이번 데이터에 없으므로 증폭 메커니즘은 아직 가설이다.

제어 PC의 설정 파일과 `force_control.cpp`, `robot_motion.cpp`, `robot_command.cpp` 해시는 검사한 로컬 파일과 일치했다. 이는 실행 중인 바이너리까지 동일함을 증명하지는 않는다. 실행 구간의 UR 드라이버 로그에는 연결 단절/재연결 이벤트가 없었고, 운동 노드는 Force 제어 초기화를 기록했다. 초기 정렬과 시작/종료의 서비스 호출은 보였지만 추종 중 경로 APPEND 서비스는 없었다. 이번 조사 범위에서는 명령 발행 지연이나 연결 단절을 주원인으로 지목할 근거가 부족하다. 로그에 이벤트가 없다는 사실로 모든 통신·스케줄링 문제를 배제할 수는 없다.

해석의 한계와 다음 수정 범위
--------------------------

실측 힘 로그는 base 좌표계이고 하위 제어 설정은 TCP 좌표계를 사용하므로, 기록된 Fz 목표와 실측 base Fz를 단순히 빼서 힘 추종 오차로 제시하지 않았다. 약 20 Hz 피드백만으로 고주파 기계 진동이나 힘 센서의 실제 취득 시각을 완전히 복원할 수도 없다.

수정 시에는 공통 실행 단계에서 예측 궤적 평활, 새 계획의 연속 연결, 속도 및 가속도 변화, 접촉 전환과 힘 좌표계 처리를 검토해야 한다. T/C 비교를 유지하려면 공통 처리의 정의와 T 원본 시간·위치·힘 의미 보존을 함께 검증해야 한다. C만 예전 서비스 방식으로 되돌리는 것은 기존 비교 조건을 바꾸므로 별도 설계 판단이 필요하다. 먼저 저장된 로그로 명령 연속성·변화율·접촉 전환을 검증할 수 있으며, 이것도 실제 진동이 해소됐다는 하드웨어 검증을 대신하지는 않는다.

이번 작업은 기존 로그·소스·설정의 읽기 및 오프라인 분석에 한정했다. 로봇을 재실행하거나 제어 설정을 변경하지 않았고, C 로그를 보존했다.

산출물 및 코드 위치
------------------

- [진단 그래프 PNG](diagnostic.png), [PDF](diagnostic.pdf)
- [수치·종료 이벤트·비교 해시·원본 해시](evidence.json), [제어 PC 읽기 결과](remote_readback.json)
- [재현 가능한 오프라인 분석](analyze.py): 이 디렉터리에서 `python3 analyze.py`로 수치와 그래프를 다시 생성한다. ROS 명령을 보내지 않는다.
- [기존 C 평활 및 별도 힘 갱신](../../behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py): `_ptp9d_stream_topup`, `_ptp9d_stream_update_force`; 현재 timed transport에서는 해당 경로를 사용하지 않는다.
- [현재 공통 실행기](../../behavior_ws/src/nrs_imitation/nrs_imitation/e2_timed_execution.py): `TimedExecution.accept`, `TimedExecution.tick`.
- 제어 PC와 해시를 비교한 로컬 소스: `/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/src/robot_command.cpp`의 `streamLoop`, 같은 디렉터리 `force_control.cpp`의 힘 활성 상태 분기.

자료 정리 이력: 이후 사용자의 비정상 로그 삭제 요청에 따라 이 보고서의 패치 전 C 원본 실행 로그·영상·실행별 그래프를 삭제했다. 본 진단 결과와 수치·해시·진단 그림은 보존했다. 삭제 범위와 패치 후 보존 C 5회는 [C5 검사 보고서](../20260926_C5_log_audit/README.md)에 기록했다. 삭제된 원본을 읽는 분석 스크립트는 현재 파일 구성에서 과거 결과를 재생성할 수 없다.


---

<a id="readme-source-19"></a>

# 원본: `reports/20260926_R5_log_audit/README.md`

2026-09-26 R 실행 로그 정리 결과

총 20회 중 가공 provider 궤적을 끝까지 실행하고 가공 구간의 힘·TCP 로그가 완전한 5회를 보존했다. 시작 실패·가공 전 중단·접근 중 중단 15회의 관련 파일은 사용자 요청에 따라 삭제했다.

보존 기준은 가공 구간의 기록 완전성이다. 다섯 실행 모두 이후 retract arrival/contact-release verification timeout으로 종료되었으며, 홈 복귀를 포함한 normal_completion은 0회다.

| 실행 태그 | 가공 구간(초) | 전체 TCP/힘 행 | 가공 구간 TCP/힘 행 | 로그 경로 |
|---|---:|---:|---:|---|
| RTC_R_20260926T170642 | 17.104237 | 982/982 | 306/306 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T170642_executor_20260926T170643_1790410003762802082_hjyzh_8_) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T170642_20260926T170644_1790410004435588336_fhh5hznt) |
| RTC_R_20260926T171118 | 17.103958 | 941/941 | 310/310 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171118_executor_20260926T171119_1790410279662742419_mayax94c) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171118_20260926T171120_1790410280349607268_043tznwz) |
| RTC_R_20260926T171328 | 17.103937 | 919/919 | 306/307 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171328_executor_20260926T171329_1790410409421232182_9nhzzix_) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171328_20260926T171330_1790410410101483480_p54gawqx) |
| RTC_R_20260926T171650 | 17.104136 | 1041/1041 | 308/308 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171650_executor_20260926T171650_1790410610994845562_axfuiaxp) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171650_20260926T171651_1790410611685746589_edv9q_l7) |
| RTC_R_20260926T172034 | 17.103830 | 909/908 | 306/306 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T172034_executor_20260926T172035_1790410835446352027_7n9a8ps6) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T172034_20260926T172036_1790410836108075672_6_j6yp4s) |

다섯 실행 모두 동일 config SHA256, 서로 다른 session ID. CSV 파싱/유한값/기록 수/시간 순서를 검증했고 로거 drop, 쓰기 오류, 미처리 큐는 모두 0이다. 가공 구간의 TCP/힘 최대 관측 간격은 61.85 ms 이내이며, 경계 포함 100 ms 이상 공백은 없다. 각 실행의 processing 단계에서 실제 발행된 명령 기록은 2,138개이고 contact gate는 모두 true다. 전체 commands.csv는 time_sampled/contact_gated/node_sent의 3단계 기록이므로 행 수를 실제 발행 횟수로 해석하면 안 된다.

영상 5개를 ffprobe로 디코딩 확인했고, 각 실행의 그래프 4개와 시작/종료 이미지 2개 및 두 NPZ 계획 파일의 유한값/시간 증가를 확인했다. 삭제 후 보존 대상 295개 파일의 SHA256이 삭제 전과 같음을 확인했다.

processing_start/end 작업자 이벤트는 없어, 분석 구간은 provider_phase processing부터 force_ramp_out까지의 대용 구간이다. 실제 가공 성공이나 시편/영역 5개의 독립성을 판정한 것이 아니다. 헤더 없는 센서 피드백이므로 미들웨어/원센서 손실량은 확정할 수 없다.

삭제 범위: 오늘 RTC R의 제외된 실행에 직접 연결된 metrics 폴더 28개, run context 15개, 그래프 폴더 15개, 영상 15개, ROS launch 폴더 15개와 노드 로그 73개. 총 811개 파일, 67,679,774 bytes.

[보존 실행 목록 CSV](retained_runs.csv) · [삭제 목록 및 검증 결과](cleanup_manifest.json)


---

<a id="readme-source-20"></a>

# 원본: `reports/20260926_T5_log_audit/README.md`

2026-09-26 T 5회 로그 확인 결과

T 실행 5회 모두 재생 궤적을 끝까지 소비하고 정지 hold를 확인했으며, 데이터 기록 검사를 통과했다. 오늘 T의 시작 실패/중도 중단 기록은 없어 삭제 대상은 0개다. 기존 R 5회와 T 5회를 보존했다.

CSV 행 구조, 유한값, 시간 순서, summary 기록 수, 로거 drop/쓰기 오류/미처리 큐를 확인했다. 영상 5개 전체 프레임을 ffprobe로 읽고 그래프 20개와 시작/종료 이미지 10개도 검증했다. 각 T의 provider_prediction/postprocessed는 443행씩, 실제 node_sent 명령은 1,864개다. 모든 launch의 5개 노드가 정상 종료했다.

| 시작 시각 | TCP 행 | 힘 행 | 재생 길이(초) | 영상(초) | 최종 목표 오차(mm) | 실행 로그 |
|---|---:|---:|---:|---:|---:|---|
| 17:28:27 | 514 | 513 | 14.910 | 29.2 | 14.991 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T172827_executor_20260926T172828_1790411308738816535__tr5tupl) |
| 17:41:11 | 452 | 451 | 14.910 | 25.9 | 19.498 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174111_executor_20260926T174112_1790412072875639201_iupu_04x) |
| 17:42:16 | 445 | 445 | 14.910 | 25.3 | 20.823 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174216_executor_20260926T174217_1790412137101159951_6tdsli24) |
| 17:43:11 | 457 | 456 | 14.910 | 25.9 | 20.187 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174311_executor_20260926T174312_1790412192846237668_x7gp49fu) |
| 17:44:14 | 489 | 489 | 14.910 | 27.8 | 19.522 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174414_executor_20260926T174415_1790412255162873318_n99zdbbj) |

다섯 실행 모두 final_target_reached=false이고 normal_completion 이벤트가 없다. 따라서 판정은 데이터 기록 완료이며 물리 작업의 정상 완료가 아니다. 가공 시작/끝 작업자 마커가 없으므로 14.910초 전체 재생 길이를 순수 가공 시간으로 해석하지 않는다.

실행 중 executor TCP/힘 관측 간격은 최대 60.903 ms, provider 관측 간격은 최대 70.858 ms였다. 실제 발행 명령 간격은 최대 9.222 ms였다. 관측된 로거 누락/쓰기 오류는 0이며, 헤더 없는 센서 피드백의 미들웨어 손실량은 별도로 확정할 수 없다.

기존 R 보존 파일 295개의 SHA256이 이전 정리 때와 일치함을 확인했다.

[보존 실행 CSV](retained_runs.csv) · [상세 검사 JSON](audit.json)


---

<a id="readme-source-21"></a>

# 원본: `reports/20260926_e1_rtc_audit/README.md`

# E1 R / T / C 사전 감사 — 2026-09-26

현재 단계는 **실행 경로 감사 완료, 공통화 구현 승인 대기**이다. 실험 준비 전체가 완료된 상태가 아니다.
로봇 launch, 로봇 명령 service, 스핀들, 학습·추론 실행은 수행하지 않았다.
기존 실행 코드·설정·데이터·checkpoint·결과를 변경하지 않고 이 감사 폴더만 추가했다.

| 산출물 | 내용 |
|---|---|
| [실행 경로 감사표](execution_path_audit.md) | R/T/C 함수 경로, 실제 launch·로그, 서비스·보간·힘·중단 차이와 근거 |
| [최소 변경 계획](minimum_change_plan.md) | 재사용 범위, 승인 필요한 시간 명세 확장, 구현·검증·운영 산출물 |
| [체크포인트 후보](checkpoint_candidates.json) | 파일 내부 config·tensor schema, SHA-256, normalizer, ON/OFF 구분 |
| [교시 목록](episode_inventory.csv) | 42개 episode의 원본 시간, 변환·힘 대응 검사, 해시 |
| [데이터 검사](dataset_verification.json) | split 재구성, 정규화 통계 대조, 시간·pose·force 원본 대응 |
| [실제 과거 로그 경로](log_transport_evidence.json) | service_stream C와 timed_topic R/T/C를 구분하는 기록 |
| [소스 해시](source_hashes.json) | 조사한 로컬 소스·설정·입력 프롬프트의 정확한 버전 |
| [감사 manifest](audit_manifest.json) | paper_experiment=E1, 구번호 원래 경로, 미확정 선택과 완료 상태 |
| [기존 git 상태](git_status_at_audit.txt) | 이번 작업 이전부터 존재한 수정·미추적 파일 목록 |

**핵심 판단**

- 기존 `experiments/e2_rule_replay_20260920/config.json`의 R/T/C는 `e2_timed_topic_v1`으로 실행한다. 반면 과거 C의 `service_stream`은 C++ 큐·보간을 거친다. 같은 `cmdMotion` 하위 입력에 도달한다는 사실만으로 두 실행기의 동등성을 주장할 수 없다.
- 기존 `SingleArmCommand.srv`에는 waypoint 시간이 없다. 위치 큐는 거리와 속도로 시간을 다시 만들고, 힘은 별도 즉시 채널을 따른다. 따라서 R/T를 이 서비스에 단순 연결하는 것만으로 교시 시간을 보존할 수 없다.
- 로컬 제어기 설정은 TCP 힘 좌표계이며 `currentF`는 base 좌표이다. 교시 센서·표면 법선·부호·중력 보정 대응은 미확인이다. 법선력 RMSE를 아직 산출할 수 없다.
- 기존 R의 18 N·편도 1회·10 mm/s와 episode_29는 과거 기록이다. 이번 E1의 기본값이나 선택으로 자동 승계하지 않는다.
- C 선택은 미확정이다. 9/16 ON `20260916_1531/policy_best.ckpt`는 내부 ON 플래그가 있고 best epoch는 39다. 9/10 baseline은 best epoch 19, ON/OFF 플래그가 없는 구버전이다. 둘 다 실제 weight 기준 state/action 9차원, history 30, horizon 128이다. 별도 9/16 `1515` 후보도 존재하므로 최신 경로 검색으로 고르지 않는다.

**승인·선택 이후 진행 순서**

1. 최소 변경 계획의 서비스 시간 명세 확장 범위를 승인하고 C checkpoint와 T episode를 명시한다.
2. 독립 reference의 존재, 법선·센서 보정, RPM 출처, calibration recipe·힘/속도 한계·공통 종료 예산을 확정한다. 미확정 필드는 null과 사유로 기록한다.
3. 공통 서비스 연결·logger·오프라인 평가·manifest를 구현하고 무하드웨어 테스트를 수행한다.
4. 배포 버전과 실제 보간·적용 명령·중단을 현장에서 확인한다. 물리 실행은 작업자가 수행한다.
5. 독립 영역에 방법을 배정해 실험하고, 로그 품질 검사 후 힘 프로파일 및 선택적 스타일러스 결과를 분석한다.

기존 `run.sh`는 구번호 실행기를 위한 도구이며 이번 E1 실행 안내로 배포하지 않는다.
새 실기 명령도 아직 존재하지 않으므로 만들어진 것처럼 기재하지 않는다.
정상 완료, 수동 중단, timeout, 안전 중단, 계측 불가의 새 판정·명령은 승인 후 구현 및 `--help` 대조를 거쳐 제공한다.

**완료 상태**

| 구분 | 상태 |
|---|---|
| 코드·설정·기존 로그 감사 | 완료 |
| 데이터·checkpoint 읽기 전용 검사 | 완료; 후보 검사이며 선택·추론 검증 아님 |
| E1 새 구현 / 전체 offline 검증 | 미수행, 감사 후 승인 단계 |
| 실제 하드웨어 확인 / 실험 | 미수행 |
| execution_equivalence | unverified |

승인 경계의 출처는 [입력 프롬프트](/home/eunseop/Downloads/Codex_E1_RTC_Execution_and_Force_Profile_Evaluation_Prompt.txt) 2절이다:
“먼저 감사 결과와 최소 수정 계획을 보고해라.”
“공통 실행 경로를 바꿔야 한다면 사용자 확인을 받은 후 구현하고, 확인 전 로봇 실행 관련 코드를 수정하지 말 것.”

이번 보고서의 수치는 파일·기존 로그에서 확인한 설정 또는 데이터 속성이다. 새 실험의 성공률·RMSE·가공 결과를 생성하지 않았다.


---

<a id="readme-source-22"></a>

# 원본: `reports/20260926_rtc_disconnect_diagnosis/README.md`

# RTC 시작 시 제어 연결 중단 진단 — 2026-09-26

확인된 장애는 UR driver의 `Connection to reverse interface dropped`이다.
오늘 세 차례 모두 RTC 노드 기동 직후 발생했다. `singleArm_cmd`와 UR driver 프로세스 자체는
그 시점에 종료되지 않았으며, 이후 작업자의 Ctrl+C로 종료된 기록이 있다.
오늘 launch 변경으로 모델 연산량이 늘었다는 근거는 발견하지 못했다.
정확한 연결 중단 원인은 아직 확정하지 않았다.

## 패치 전후 비교

- 오늘 15:07 사전 감사의 SHA-256과 비교한 23개 파일이 모두 동일하다. 추론 본체,
  하위 launch, executor, 실험 config, 로컬 제어 소스가 포함된다.
- 제어 PC의 `robot_command.cpp`, `singleArm_cmd.cpp`, `robot_motion.cpp`도 감사 당시 소스 해시와 동일하다.
  이는 소스 비교이며 실행 바이너리의 동일성을 별도로 증명한 것은 아니다.
- 9월 20일 19:58 R 기록과 오늘 R 3회의 `runtime_parameters` 295개를 각각 비교했다.
  차이는 `e2_config`, `e2_session_id`, `metrics_context_file`, `metrics_run_tag` 4개뿐이다.
  실험 config 원본은 동일하고 오늘은 로그 폴더에 실행 문맥을 추가한 복사본을 사용한다.
- `resolved_runtime`, checkpoint/normalizer 메타데이터, sampling은 동일하다.
  제어 125 Hz, infer timer 5 Hz, FLOW steps 10, horizon 128, force history 30,
  metrics 20 Hz가 증가하지 않았다. 기존 shell과 새 launch 모두 OMP/OpenBLAS thread 제한은 2다.
- 9월 20일과 오늘 모두 origin, overlay recorder, removal recorder, executor, inference의 5개 노드를 시작한다.
- 오늘 패치는 RTC 진입점과 하위 `config` 이름 충돌 수정이다. git HEAD 대비 전체 diff에는
  9월 20일의 미커밋 변경도 포함되어 있으므로 그 전체를 오늘 변경으로 해석하지 않았다.

## 과거에도 같은 시작 시점의 연결 중단이 있었다

| 실행 | 노드 시작 (KST) | reverse 연결 중단 (KST) | 기록 시각 차이 |
|---|---|---|---:|
| 9/20 R | 19:57:40.781 | 19:57:41.024 | 243 ms |
| 9/20 T | 20:09:51.397 | 20:09:51.619 | 222 ms |
| 9/26 R 1 | 16:12:43.494 | 16:12:43.728 | 234 ms |
| 9/26 R 2 | 16:13:16.361 | 16:13:16.606 | 245 ms |
| 9/26 R 3 | 16:13:49.358 | 16:13:49.597 | 240 ms |

양쪽 PC는 조회 시 `NTPSynchronized=yes`였다. 표는 서로 다른 호스트의 wall-clock 로그 비교이며,
장애 순간의 시계 오차를 별도로 계측하지 않았으므로 정밀 지연 측정값으로 취급하지 않는다.
9/20 19:58 R 실행에서는 정상적인 provider 준비 및 timed command 기록도 있어, 간헐적 장애 양상이다.

오늘 마지막 실행에서 driver 연결 중단은 16:13:49.597, executor 초기화 로그는 49.892,
추론 노드의 device 로그는 50.481, 첫 current-pose 명령은 50.767,
원점 확정 및 PTP 요청은 52.10 부근이다. 제어 PC가 startup stream-stop을 처리한 기록도
50.171이다. 기록상 연결 중단은 이들 명령보다 먼저 발생했다.

R/T는 `inference_core.py`의 `execution_method != il` 분기에서 신경망 로드를 건너뛴다.
`policy`는 None이며 GradCAM과 modality attribution도 실제 실행 시 꺼진다.
`Using device: cuda`라는 로그만으로 FLOW GPU 추론이 실행됐다고 판단할 수 없다.
오늘 세 실행 모두 `provider_prepared` 이벤트가 없었으며 가공 궤적 준비 전 중단됐다.
Python/torch import, 카메라 처리, ROS 발견 통신 등 기동 부하는 여전히 발생할 수 있다.

## 자원·통신 관련 관측과 해석의 한계

- 제어 PC는 i7-6600U, 2 physical cores / 4 logical CPUs, generic kernel 6.8.0-138이다.
  CPU governor는 powersave, shell RT priority limit은 0이다.
- UR driver에 `Could not enable FIFO RT scheduling policy` 경고가 있고,
  조회한 UR/SingleArm threads는 모두 TS였다. 같은 FIFO 경고는 9월 20일에도 있었다.
- 제어 PC가 로봇 `.47`과 추론 PC `.150`에 접근하는 인터페이스는 모두 `enp2s0`이다.
  링크는 1 Gbps이며 RX dropped 누적 320이 관측됐다. 누적값으로 이번 장애의 패킷 손실을 단정할 수 없다.
- 조회 시 제어 PC available RAM은 약 6.1 GiB, swap 사용량은 0이었다.
  해당 시간대 kernel 로그 조회에서 OOM kill/segfault를 찾지 못했다.
- 추론 종료 후 로컬 GPU는 662 / 16376 MiB, 7%였다. 이는 장애 순간의 사용량 측정이 아니다.
- CPU/통신 burst와 제어 스케줄링 지연, DDS 노드 발견 부하가 우선 확인할 가설이다.
  이 중 어느 것도 로그만으로 원인 확정하지 않았다. 로봇 측 stop reason도 확보하지 못했다.

현재 근거로 모델 크기나 FLOW steps를 줄이는 조치를 정당화하기 어렵다.
다음 재현에서는 노드 기동 순간의 제어 PC CPU scheduling, reverse socket/네트워크,
로봇 측 stop reason을 함께 기록하고, 기록·시각화 노드 분리 여부를 한 번에 하나씩 비교하는 것이 적절하다.
실시간 우선순위 설정도 점검 대상이나 이번 진단에서는 설정을 변경하지 않았다.

## 근거 파일

- [기계 판독 가능한 비교·해시·타임라인](comparison.json)
- [오늘 제어 PC 로그](remote_today_driver_and_control.txt)
- [과거 중단 기록·제어 소스 해시](remote_previous_disconnects_and_source_hashes.txt)
- [제어 PC 스레드·네트워크 상태](remote_resources_and_network.txt)
- [로컬 시간 동기화·네트워크](local_clock_and_network.txt)
- [R 파라미터 차이](runtime_parameter_diff.txt)

원격 작업은 로그·상태 조회만 수행했다. 로봇/추론을 시작하거나 service를 호출하지 않았다.
이번 진단 중 제어 설정, launch, 모델 코드, 기존 실험 로그는 수정하지 않았다.


---

<a id="readme-source-23"></a>

# 원본: `reports/20260927_E1_RT_common_patch/README.md`

# 어제 C 기준 E1 R/T 공통 후처리 패치

사용자 요청에 따라 2026-09-26 C 전용 안정화 처리를 2026-09-27 R/T 재실행에 적용했다.
새 설정은 `experiments/e1_rt_common_20260927/config.json`이며 실행 명령은
[R/T 실행 안내](../../experiments/e1_rt_common_20260927/README.md)에 있다.

기준 C는 `results/20260926/E1/C/`의 정상 5개 실행이다. 각 launch 설정의
common/executor/recipe/replay/il/task가 현재 기존 설정과 모두 같음을 확인하고 복사했다.
새 `rtc_pose_conditioning` 설정에서만 R/T 공통화를 활성화한다.

R/T의 원본 경로는 고정된 채 힘·위상·종료 시각을 제공한다. 위치·자세의 내부 보기만 30 Hz,
128점, 120점 stride로 나누어 **기존 CPoseConditioner 자체**에 전달한다.
이렇게 해야 R 원본 125 Hz에 35점 필터를 그대로 적용하여 C와 평활 시간이 달라지는 문제가 없다.
T는 설정의 resample_hz 표기와 달리 실제 provider에서 원본 443개 시각(약 30 Hz)을 그대로 읽는다.
원본 시간은 바꾸지 않고 C와 같은 규칙 격자의 위치·자세 보기를 만든다.
각 구간은 같은 35점 위치/quaternion 평균, 0.5초 연결, 위치/회전 가속도 제한을 거친다.
힘에는 이 평활을 적용하지 않는다. 유한 경로는 반복·교체되지 않으며 기존 시각에 종료한다.

## 검증

- `full_tests.xml`, `full_tests.log`: 168 passed. 기존 E1/E2 로더·공통 실행·정지·복귀·기록 포함.
- `targeted_tests.xml`: 공통 후처리·C 기존 동작·launch의 40개 검사.
- `recorded_c_validation.json`: 어제 C 5회/9,662틱, 저장된 당시 코드와 현재 코드의
  requested/conditioned/gated/sent 9차원 명령이 모두 바이트 단위로 동일.
- R/T에 같은 원본 경로를 C 형식의 128점 구간으로 따로 공급하여 위치·자세 명령의 일치를 검사.
  원래 힘 값·접촉 게이트·힘 발행·경로 시간 보존, 속도/가속 제한, 유한 종료도 검사.
- `R_launch_check.log`, `T_launch_check.log`: 설치된 ROS launch 검사 통과, 노드는 시작하지 않음.
- `finite_plan_validation.json`: 실제 R/T 계획의 시간축 및 연산 시간 확인. 새 구간을 처리하는
  틱의 로컬 최대 시간은 R 0.84 ms, T 0.89 ms였다. 실제 ROS 부하의 마감시간 보장은 아니다.
- `protected_before.json`, `validation.json`: 어제 C 데이터·영상·소스/설정 사본과
  E1 manifest/PDF 등 288개 파일의 내용 보존 확인.

첫 전체 검사에서 프로젝트 `source` 경로가 없어 모델 import 5개가 실패했다.
명시적 PYTHONPATH와 ROS 환경을 적용한 최종 전체 검사에서 168개가 통과했다.
ROS launch_testing과 현재 pytest의 플러그인 API 충돌을 피하려고 자동 플러그인 로딩을 끈
일반 단위 테스트 환경을 사용했고, launch는 별도로 실제 CLI의 무노드 check를 실행했다.

T 원본 14.9104초는 종료 시점일 뿐 최종 위치 도착 보장이 아니다. 공통 속도 제한을 적용한
명령의 끝점 차이는 기존 15.09 mm, 새 처리 13.29 mm였다. 종료시간 연장이나 완료 판정 완화는 하지 않았다.
R 가공 후 별도 Position 복귀는 기존 동작이며 가공 비교와 분리한다.
C는 정책에서 새 구간을 받고 R/T는 고정 경로를 구간으로 나누므로 제공자 차이는 유지된다.
실제 힘 응답과 날짜 간 작업 조건은 실기 기록과 작업자 절차로 확인한다.

공유 소스 파일 해시가 바뀌므로 E2의 실행 계약/설정 버전 식별자도 근거를 남기고 갱신했다.
`E2_contract_review.json`에서 이전/새 식별자를 확인할 수 있다. E2 모델, 실행 상수,
실험 프로토콜, 보정, F0는 변경하지 않았다. A의 F0 미확정 차단도 유지된다.
기존 결과의 계약 식별자를 소급 변경하지 않았다.

`before/`는 이번 수정 전 소스/설정이고 `changes.patch`는 이번 변경만의 diff다.
실제 run launch, 로봇 service 호출, 스핀들 실행은 수행하지 않았다.


---

<a id="readme-source-24"></a>

# 원본: `reports/20260927_E1_RT_common_patch/before/experiments/rtc_launch_20260926/README.md`

# R/T/C를 ros2 launch로 실행

`rtc_timed.launch.py`는 기존 `e2_timed_topic_v1` 실행기의 직접 launch 진입점이다.
기존 shell 실행 스크립트 없이 method=R/T/C를 선택할 수 있다.
공통 service·controller interpolation을 사용하는 새 E1 실행기는 아직 구현하지 않았다.
이 파일 추가로 실행 동등성·힘 법선 보정·recipe validation이 완료되는 것은 아니다.

2026-09-26 C 진동 패치를 적용했다. 아래의 기존 C 명령은
`c_pose_conditioning_20260926_v1`을 활성화한다. C에만 35점 위치/자세 평활,
0.5초 재계획 연결, 위치/회전 가속도 제한을 적용한다. R/T와 공통 실행 설정,
힘 목표·접촉 판정·힘 변화율은 유지한다. 오늘 R/T 10회의 입력 재생에서도
이전 코드와 같은 9차원 명령이 나왔으며 기존 기록을 보존했다.
첫 C는 1회 검증 실행으로 진동과 힘 응답을 확인한다. 실제 진동 해소는 아직 검증하지 않았다.
패치 수치·비교 조건·테스트는 [C 패치 보고서](../../reports/20260926_C_execution_patch/README.md)에 있다.

현재 프로젝트에서 패키지 빌드·설치를 마쳤다. 각 터미널에서 ROS 환경을 읽는다.

```sh
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash
```

**점검: 로봇·추론·카메라 노드를 시작하지 않는다**

```sh
ros2 launch nrs_imitation rtc_timed.launch.py --show-args

ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=R \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=check
```

`mode`를 생략해도 check이다. config는 필수이고 자동 선택하지 않는다.
T는 `episode`, C는 `checkpoint`까지 명시해야 check/run이 통과한다.
검사 실패는 노드를 시작하기 전에 종료한다. 기존 preflight의 통과는 하드웨어 또는 새 E1 준비 완료 판정이 아니다.

**기존 timed 실행기 수동 실행 — 한 번에 한 방법**

다음 명령은 현재 config를 사용하기로 선택하는 경우의 예시다.
R은 기존 18 N, peak 10 mm/s, 직선 편도 1회와 접촉 해제·교시 시작 자세 복귀 설정이다.
T는 episode_29, C는 명시된 9/16 ON1531 checkpoint를 선택하는 명령이다.
새 실험의 recipe/교시/model을 자동으로 확정한 것은 아니다. 다른 선택은 검토한 config와 launch 인자가 함께 일치해야 한다.
현재 config의 RPM은 unknown이며 표면 법선·배포 버전 확인도 별도이다.

```sh
# R
ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=R \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=run

# T — episode_29를 선택하는 경우
ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=T \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  episode:=episode_29 mode:=run

# C — 아래 9/16 ON checkpoint를 선택하는 경우
ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=C \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=run \
  checkpoint:=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/on/20260916_1531/policy_best.ckpt
```

먼저 같은 명령에서 `mode:=run`을 `mode:=check`로 바꿔 점검할 수 있다.
원하는 로그 태그는 `run_tag:=RTC_R_r01`처럼 추가한다. 생략하면 방법과 시각으로 생성한다.
기존 로봇 driver·제어기·카메라 실행 환경은 사용자가 준비해야 한다. 이 launch는 스핀들을 켜지 않는다.
세 방법은 각각 독립 작업 영역에서 실행하며 동일 영역을 순차 가공한 결과를 독립 비교로 세지 않는다.

**가공 구간 기록·종료·중단**

다른 ROS 환경 터미널에서 기존 실행기의 실제 service를 사용한다.

```sh
ros2 service call /e2_executor/processing_start std_srvs/srv/Trigger '{}'
ros2 service call /e2_executor/processing_end std_srvs/srv/Trigger '{}'

# 작업자가 정상 종료를 요청할 때, 특히 C
ros2 service call /e2_executor/finish std_srvs/srv/Trigger '{}'

# 수동 중단
ros2 service call /e2_executor/abort std_srvs/srv/Trigger '{}'
```

R/T는 유한 궤적 끝에서 자동 정지를 요청한다. R만 기존 설정에 따라 home 복귀를 포함한다.
정지 상태를 확인한 뒤 launch 터미널에서 Ctrl-C로 기록을 마감한다.
실행 도중 Ctrl-C는 manual_abort이며 service ACK 자체는 물리 정지 확인이 아니다.
기존 제어기는 통신 단절 시 물리 정지를 보장하는 장치가 아니므로 기존 현장 중단 절차를 따른다.

launch는 원본 config를 덮어쓰지 않는다. `mode:=run`은 `logs/rtc_launch_context/RTC_<method>_*/config.json`에
실행용 복사본과 paper_experiment=E1, 구번호 source config 경로/hash, 명시 선택 인자,
execution_equivalence=unverified를 기록한다. provider/executor는 이 동일 복사본을 사용한다.
CSV/event는 `logs/inference_metrics`, 영상은 기존 `logs/polishing_removal`에 남는다.

**검증 범위**

선택 누락·OFF/설정 불일치 차단, check 모드 무노드·무파일, R/T/C 전달 인자,
run config 복사와 원본 보존을 노드 실행 없이 검사했다.
사용자의 16:01 실행에서 발견된 하위 launch의 `config` 인자 충돌을 수정했다.
RTC 진입점 전용 인자는 그룹 안에서 제거하고, 실험 JSON은 `e2_config`로 전달한다.
원점 감지 노드는 자기 패키지의 `stain_relative_frame.yaml`을 읽고, 동결된 감지 설정은 기존 `detect_params`로 받는다.
실제 하위 launch를 노드 실행 직전까지 펼쳐 파라미터를 평가하는 R/T/C 회귀 검사도 추가했다.
수정 후 총 12개 검사를 통과했다. 에이전트는 실제 run launch·service 호출·로봇·스핀들을 실행하지 않았다.
하위 힘 제어·보간·게인·안전한계는 변경하지 않았다. 새 E1 준비의 남은 사항은
[사전 감사](/home/eunseop/nrs_imitation/reports/20260926_e1_rtc_audit/README.md)를 참조한다.


---

<a id="readme-source-25"></a>

# 원본: `reports/20260927_E2_A_preparation/README.md`

# E2 A 자동 실행 준비 결과 — 2026-09-27

A 500 epoch 완료 모델의 등록과 정규화 연결, 실제 시작 로더와 모델 추론, 공통 실행 계약 및 ROS launch 파라미터 검사를 완료했다.
F0에 필요한 사용자 사실 확인은 미확정 상태를 유지했다. 실제 A 로봇 실행은 하지 않았다.

- `summary.json`: 완료한 자동 설정과 남은 실제 실행 차단 항목
- `config_before_registration.json`, `config_after_registration.json`: 등록 전후 설정
- `config_after_loader_fix.json`: A 시작 로더 수정 후 현재 설정
- `trained_A_validation.json`: 실제 학습 모델/교시 입력 추론, 힘 관측 변경 불변성, 6차원 출력과 null force 로그
- `tests.log`, `tests.xml`: 시작 로더 수정 전 관련 오프라인 테스트 40개 통과 기록
- `full_checks.log`, `full_tests.xml`: 시작 로더 수정 후 전체 오프라인 테스트 157개 통과
- `runtime_loader_before_fix.log`, `runtime_loader_after_fix.log`: A 시작 오류 재현 및 A/B/C 모델 복원 회귀 검사
- `runtime_loader_review.json`, `runtime_loader_fix.patch`: 변경 범위 검토 및 실행 계약 갱신 근거
- `ros2_launch_check.log`, `preflight.json`: 실제 ros2 launch check 결과; A의 남은 차단 사유는 F0 미확정
- `remaining_F0_inputs.json`: 정상 교시 가공 구간/좌표 보정 또는 이미 검증된 일정 목표 힘에 대한 사용자 정보
- `protected_artifacts_before.json`: E1 및 B/C 결과의 manifest/PDF 식별 해시 보존 확인
- `protected_artifacts_after.json`: B/C 보관 파일 817개와 기존 manifest/PDF 보존 확인
- `installed_source_check.json`: ROS 설치 경로가 수정된 소스를 직접 사용함을 확인

등록용 파일명은 policy_best.ckpt지만 epoch 500의 원본 policy_last.ckpt와 동일한 SHA-256이다.
B/C 모델 및 공통 실행 값은 그대로다. A 모델 등록 후 시작 로더의 pose6 입력 복원 오류를 수정했으며,
코드 변경을 구분하기 위해 실행 계약 및 설정 버전(cohort)을 갱신했다. 사용자 공정 확인 내용은 그대로다.
A와 과거 best B/C의 선택 규칙 차이는 기존 exploratory 비교 분류를 유지한다.

실행 명령과 처리 시작·종료 서비스는 experiments/e2_force_ablation_20260926/A_execution_20260927.md에 정리했다.


---

<a id="readme-source-26"></a>

# 원본: `reports/20260927_E2_A_protection_patch/install/Y2RobMotion/share/Y2RobMotion/txtcmd/README.txt`

# Explanation of "cmd_6D.txt"
-> Each column means x, y, z, wx, wy, wz, desired_lin_vel, desired_ang_vel, holding_time
-> Velocity means the velocity between present and previous points
-> And first row data of velocity will be ignored
-> Unit: mm, degree, mm/s, degree/s, s


# Explanation of "cmd_9D.txt"
-> Each column means x, y, z, wx, wy, wz, fx, fy, fz, desired_lin_vel, desired_ang_vel, holding_time
-> Velocity means the velocity between present and previous points
-> Force means the force between present and previous points
-> And first row data of velocity will be ignored
-> Unit: mm, degree, N, mm/s, degree/s, s

# Explanation of "cmd_continue9D.txt"
-> Each column means x, y, z, wx, wy, wz, fx, fy, fz
-> Force means the force between present and previous points
-> And first row data of velocity will be ignored
-> Unit: mm, rad, N


# NOTE!!
-> holding posture must be same with previous posture!!
-> At holding posture, you can maintain the set force
-> Most high priority setting is holding_time 

---

<a id="readme-source-27"></a>

# 원본: `reports/20260927_E2_F0_baseline_review/README.md`

# E2 F0 비접촉 기록 검토 — 2026-09-27

사용자가 실행한 `unloaded_151625_rn9ptmg4`의 저장 파일 5개 해시를 확인했다.
기록은 정상적으로 마감됐으며 CSV 2,473행과 원본 bag 메시지 수가 일치한다.

| 항목 | 결과 |
|---|---:|
| 힘 기록 구간 | 19.7755초 |
| 힘 / 자세 / 제어 모드 메시지 | 각각 2,473개 |
| 관측 힘 수신 주기 | 약 125.00 Hz |
| 최대 힘 수신 간격 | 11.911 ms |
| 최대 대응 자세 나이 | 9.260 ms |
| TCP Fz 평균 | −0.1506 N |
| TCP Fz 표준편차 | 0.1842 N |
| TCP Fz 최소 / 최대 | −0.8492 / +0.4749 N |
| 처음 대비 최대 위치 이동 | 0.0834 mm |
| 처음 대비 최대 회전 변화 | 0.000147 rad |
| 제어 모드 | 전 구간 Position |
| 기존 base Fz 접촉 ON 3 N에 도달한 비접촉 샘플 | 0개 |

첫 힘 샘플 1개는 이전 자세가 없어서 TCP 변환에 포함하지 않았다. 원본 및 base 힘 통계에는 보존했다.
나머지 2,472개는 기존 0.2초 freshness 기준 안에서 대응한다. 큰 영점 편차나 긴 수신 공백은
이 기록에서 관측되지 않았다. 영점 보정값을 자동 적용하지 않았으며, 물리적 비접촉 여부는 작업자 준비 조건이다.

## 현재 로봇 PC 읽기 전용 확인

- setup_parameters.yaml SHA-256 `18eeb2069eb3f9ae24c3d35299223dd8166e4fcb910192f3a6b83a7fc95ef4a0`:
  E1에서 보존한 설정과 같으며 Force_Con_Coordinate=1(TCP), 125 Hz 설정이다.
- spindle_gravity.yaml SHA-256 `699eb9d59c6776be2336ebf2469780dc9e614e8ac6b9189443239f9cab662cb2`:
  교정 ID 2026.09.01 21:32로 이전 확인본과 일치한다.
- 오늘 FT 시작 로그 `FTGetMain_601830_1790489413523.log`에서 중력 보정 requested=true,
  matrix_loaded=true를 확인했다. 설정 파일 존재 확인과 오늘 시작 로그 확인이며, 새 센서 교정을 시행한 것은 아니다.

상세 근거는 `review.json`, `remote_configuration_readback.json`에 있다.
원본 결과·설정·교정 파일은 수정하지 않았다.

## 다음 적용 검증 상태

사용자는 장비·공구 허용 접촉힘 범위 또는 현장 힘 중단 기준에 대해 **“미확인”**이라고 답했다.
현재 공통 실행 설정의 fz_hard_limit_N=0은 제한 비활성을 뜻하며 허용 범위의 근거가 아니다.
제어기 설정의 FORCE_SWITCH_PRECONTACT_FORCE_HOLD=15 역시 비접촉 시 제어 목표 처리값으로,
장비의 허용 힘이나 일반적인 과부하 중단 한계로 해석하지 않는다.

따라서 다음 단계의 23 N 적용 검증은 보류한다. 원래 E2 지시문 4-2는 검증된 운용 범위의
후보와 별도 검증을 요구한다. 확인 없이 측정된 교시 힘 범위나 기존 R의 18 N을 허용 상한으로 만들지 않는다.
장비·공구 책임자가 확인한 운용 범위/중단 기준과 그 근거가 확보되면 별도 적용 검증을 구성한다.
그 후 힘 축·부호·명령 대응과 실제 반응을 확인하고 사용자가 최종 F0를 확인하면 고정한다.

현재 상태: **비접촉 기준 기록 검토 완료 / 후보 23 N 적용 검증 대기 / A 본시험 F0 미확정**.


---

<a id="readme-source-28"></a>

# 원본: `reports/20260927_E2_F0_candidate/README.md`

# E2 A 외부 힘 후보 — 2026-09-27

**저장된 교시 Fz의 상수 근사값은 23.018 N, 정수 단위 검토 후보는 23 N이다.**
현재 TCP 힘으로 물리 보정·검증·고정한 값이 아니며 A 실기용 F0에는 적용하지 않았다.

사용자 확인: “0819 42개는 모두 정상 작업이야 / 없음”. 선택된 42개가 모두 정상 작업이고,
같은 공정에서 별도로 검증한 상수 힘 recipe는 없다는 뜻으로 기록했다.
성공 여부를 다시 확인하거나 사용자가 F0 숫자를 직접 계산할 필요는 없다.

## 산출 근거

- 원본은 2026-08-19 교시다. 사전에 고정한 학습 38개를 모두 사용했고 검증 4개는 사용하지 않았다.
- `action/force[:, 2]`는 원본 HDF5의 필터링된 `ft[:, 2]`와 전부 일치한다.
  이 신호는 교시 측정 Fz이며 현재 로봇의 실제 applied target 기록이나 검증된 법선력은 아니다.
- 힘 크기를 보지 않고 자세 Z의 접근·가공·이탈 구간을 근사 분할했다. 이후 38개 모두의
  경계 영상과 자세 궤적을 확인했다. 정확한 접촉 순간을 수동 정답으로 확정한 것은 아니다.
- 각 가공 후보 구간을 원래 시간 간격으로 평균한 뒤, 38개 교시에 같은 가중치를 주었다.
  총 구간 길이는 220.36초, 구간 내 샘플은 6,591개다.
- 낮은 힘과 음수도 남겼다. 가중치가 있는 3 N 미만 95개, 0 N 이하 30개가 계산에 포함됐다.
  절댓값이나 새 영점·중력 보정은 적용하지 않았다.
- E1 R의 18 N이나 E2 B/C 본시험 결과로 값을 정하지 않았다.

목적함수는 `mean_episode(integral((Fz - c)^2 dt) / duration)`이다.
독립적인 샘플별 시간 겹침 계산으로 23.0182261091 N을 재현했다. 이 값에서 ±1 N을 움직이면
제곱오차가 각각 정확히 1 N² 증가했다. 기존 B/C 역정규화·후처리 함수의 오프라인 검사에서도
23 N은 실행기 직전까지 같은 수치와 부호로 전달됐다. 이는 소프트웨어 채널 확인이며 물리 보정 증명은 아니다.

## 구간 불확실성

| 구간 선택 | 상수 근사값 |
|---|---:|
| 기본 자세 기준 | 23.02 N |
| 양쪽 경계 0.25초 확장 / 축소 | 21.90 / 23.75 N |
| 양쪽 경계 0.50초 확장 / 축소 | 20.46 / 24.22 N |
| 자세 높이 허용차 0.5 / 1.5 mm | 23.37 / 22.72 N |
| 같은 구간, 사다리꼴 시간 적분 | 23.08 N |

경계를 넓히면 접근·이탈 일부를 포함할 수 있다. 위 값들은 구간 민감도이며 검증된 운용 범위나
실기에서 시험하도록 승인된 후보 목록이 아니다. 개별 교시 평균 19.19–27.93 N 역시 운용 한계가 아니다.

## 실제 A 실행 전 남은 일

1. 이 근사 가공 구간을 최종 recipe의 산출 구간으로 받아들일 수 있는지 검토한다.
   `boundary_review_01.png`부터 `07.png`, `pose_phase_proposals.json`에 모든 구간을 남겼다.
2. 교시의 축·영점·중력 보정과 현재 TCP 목표 힘의 대응을 검증한다.
   앞선 파일 감사에서 보정 코드·YAML 및 로봇 PC 시작 로그는 확보했지만 교시 당시 보정 해시와
   zero 자세가 원본에 기록되지 않아 이 대응까지 확정하지는 못했다.
3. 해당 대응과 실제 운용 범위가 확인된 뒤, 작업자가 참여하는 별도 검증 계획에 후보·설정·
   중단 기준·기록 방법을 명시한다. 본시험 B/C 결과를 이용해 값을 최적화하지 않는다.
4. 검증 결과를 바탕으로 최종 F0와 출처를 사용자 확인 후 고정한다.
   이는 원래 E2 지시문 4-2의 “사용자 확인 후 최종 F0와 그 출처를 freeze한다”에 따른 절차다.

현재 사용자에게 미지의 보정값이나 운용 한계를 추측해서 제공하도록 요구하지 않는다.
위 근거가 없는 부분은 별도 검증 대상으로 남겼다. 23 N에 대한 동의만으로 보정 완료로 바꾸지 않는다.
본 작업에서 로봇 명령은 보내지 않았고, 실행 설정과 기존 E1/E2 결과는 그대로다.

## 파일

- `candidate.json`: 후보 수치·출처·가중·구간 민감도·미확정 항목.
- `episode_means.csv`: 교시별 원본 대응·시작/끝 시각·시간 평균.
- `F0_candidate_review.pdf`: 전체 후보와 38개 교시 힘 그래프(8쪽).
- `visual_review.json`, `boundary_review_*.png`: 경계 검토 기록과 영상.
- `verification.json`: 독립 검산 및 실제 소프트웨어 함수 검사.
- `manifest.json`: 산출물과 입력 확인 파일 해시.

이 자료는 `reports/20260927_E2_F0_evidence/`의 사용자 확인 전 감사에 대한 후속 기록이다.
그 시점의 “정상 작업 여부 미확인”은 이번 사용자 확인으로 해소됐다.


---

<a id="readme-source-29"></a>

# 원본: `reports/20260927_E2_F0_evidence/README.md`

# E2 F0 근거 확인 — 2026-09-27

F0 숫자나 보정 행렬을 사용자가 직접 계산해서 제공할 필요는 없다. 기존 파일과 로봇 PC를 읽어 기술 근거를 확인했다.
현재 자료만으로는 작업 성공 여부와 정확한 가공 구간을 확정할 수 없어 F0를 설정하지 않았다.

| 항목 | 확인 결과 | 의미 |
|---|---|---|
| 교시 원본 | 2026-08-19 촬영, 90도 작업 42개; 학습 38개/검증 4개 | 후보 산출에는 확인된 학습용 교시 사용 |
| 저장/출처 | 42개 모두 save_complete=1, joystick_end; action/force와 원본 ft가 전부 일치 | 기록 완료이며 작업 성공 판정은 아님 |
| 힘 신호 | /ftsensor/measured_Cvalue → EMA(alpha=0.2) → action/force, 시간 대응 보존 | 힘 센서 관측값을 교사 action으로 사용한 것; 실제 controller applied target 기록은 아님 |
| 교시 보정 | zero/축 재배열/중력 변화량 보상 코드와 2026-08-16 22:43 보정 YAML 존재 | 교시 파일에는 당시 로드한 보정 해시·zero 자세가 없음 |
| 로봇 보정 | 원격 PC 현재 YAML: 2026-09-01 21:32; E2 B/C 직전 로그 requested=true, matrix_loaded=true | 로컬 dev_ws의 오래된 복사본으로 원격 상태를 판단하지 않음 |
| 명령 좌표 | 기존 사용자 확인 및 소스상 Force_Con_Coordinate=1, TCP 축 target | 기존 E1 C 설정 확인을 다시 요구할 필요 없음 |
| 가공 구간 | 수동 processing 시작/끝 기록 없음 | contact_on은 힘 threshold 자동 검출; EMA mode의 contact_off=-1은 종료 검출 미구현을 뜻함 |
| 성공/품질 | 성공 판정 annotation 없음 | 사용자 확인 또는 교시 영상 검토가 필요 |

사용자에게 우선 확인한 것은 두 가지다.

1. 선택된 42개 교시를 모두 정상 잉크 제거 교시로 봐도 되는지, 실패/연습이 섞였는지.
2. 같은 장비·공정에서 별도로 검증한 일정 목표 힘이 있는지. 있다면 값과 사용한 작업/설정, 없다면 없다고 답하면 된다.

정확한 구간 초 단위를 사용자가 외워서 제공할 필요는 없다. 영상에서 구간 후보를 정리해 검토할 수 있다.
`teacher_phase_review.png`는 학습 episode_0/38을 동일한 시간 간격으로 뽑은 검토 자료이며 구간 확정이나 성공 판정이 아니다.
`episode_evidence.csv`의 processing_start/end와 success_confirmed는 확인 전이므로 비어 있다.

교시 Fz를 법선력이나 현재 TCP 명령과 자동으로 동일시하지 않았다. 교시 기록 당시의 zero/보정·축 대응을
확인하거나, 이미 검증된 command-target 변환/상수 힘 recipe 근거를 사용해야 한다.
E1 R의 18 N은 자동 복사하지 않으며 E2 B/C 본시험 결과로 F0를 조정하지 않는다.
최종 후보와 근거를 제시한 뒤 사용자가 확인하면 freeze한다(원래 E2 지시문 4-2).

확인 파일은 audit.json, source_recording_metadata.json, remote_spindle_gravity.yaml,
remote_FT_startup_logs.json에 저장했다. 실험 설정, 학습 모델, E1/B/C 결과 및 로봇 동작은 변경하지 않았다.


---

<a id="readme-source-30"></a>

# 원본: `reports/20260927_E2_F0_observation_setup/README.md`

# F0 검증 1단계: 비접촉 기준 기록

기존 E2 launch는 F0가 미확정인 A를 실행하지 않는다. 실제 적용 전 확인 자료를 모으기 위해
힘·자세·제어 모드만 구독하는 별도 rosbag launch를 추가했다. A/B/C 실행이나 힘 목표 설정은 하지 않는다.

작업자가 기존 절차로 로봇을 HOME에 정지시키고, 공구를 시편과 떨어뜨리고 스핀들을 끈다.
로봇 드라이버와 힘 센서 노드는 켜 두며 추론/실험 launch는 종료된 상태에서 아래 명령을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/install/setup.bash
source ~/nrs_imitation/behavior_ws/install/setup.bash
ros2 launch nrs_imitation e2_f0_observe.launch.py mode:=record duration_s:=20
```

약 20초 뒤 recorder에 SIGINT를 보내 정상 마감하고 오프라인 분석을 실행한다.
출력 경로는 `results/YYYYMMDD/E2/F0_validation/unloaded_<시각>_<고유ID>/`다.
`bag/`에 원본 세 토픽, `force_pose.csv`에 실제 pose와 짝지은 힘,
`summary.json`에 base/TCP 힘 평균·표준편차·범위, 기록 간격과 자세 변화량을 저장한다.
`request.json`과 `manifest.json`도 저장한다. 피드백이 없으면 `insufficient_feedback`이며,
기록 자체가 실패하면 `analysis_error.json`을 남긴다. 빈 기록을 0 N으로 처리하지 않는다.

`/ur10skku/currentF`는 확인한 코드의 base 힘 규약을 사용하고, `/ur10skku/currentP`의
실제 회전벡터로 TCP 성분을 계산한다. 같은 bag 수신 시각 기준으로 이전 pose를 연결하며
0.2초보다 오래된 pose에는 TCP 변환값을 만들지 않는다. headerless 메시지여서 센서 취득 시각의
정확한 동기화나 현재 물리 보정을 이 기록만으로 인증할 수는 없다.

이 단계에서는 측정 영점을 자동으로 빼거나 보정 파일을 바꾸지 않는다.
공구 비접촉 여부는 작업자 준비 조건이며 데이터만으로 판정하지 않는다.
후보 23 N의 가공 적용, 힘 축/부호의 독립 검증, 운용 범위 검증 또는 F0 확정을 완료한 것으로
표시하지 않는다. 기준 기록을 검토한 뒤 필요한 보정 및 별도 힘 적용 검증을 진행한다.

노드 없이 구성만 확인하려면 `mode:=check`를 사용한다. 설치 및 오프라인 검사 5개 통과,
실제 launch check 통과. 에이전트는 record 모드나 로봇을 실행하지 않았다.


---

<a id="readme-source-31"></a>

# 원본: `reports/20260927_E2_F0_offline_assumption/README.md`

# 23 N 가정값의 오프라인 검사

사용자의 “그냥 임의의 값으로 채워줘” 요청에 대해, 기존 교시 평균에서 반올림한
23 N을 `offline_assumption.json`에 **미검증 가정값**으로 저장했다.
이 파일은 실기 설정과 별도 스키마이며 로봇 실행 설정으로 사용할 수 없다.

기존 실행기에 23 N 가정값, 정지 자세, 합성 접촉/비접촉 상태를 넣어 450틱을 계산했다.
가공 구간 가정 밖의 목표 0 N, 접촉 게이트, 기존 30 N/s 힘 변화율 제한의 계산을 확인했다.
이 검사는 실제 A 모델 추론, 동역학 시뮬레이션, 실측 힘 추종 검증 또는 허용 힘 검증이 아니다.

실제 로봇 명령은 0회이며 ROS context도 시작하지 않았다. 실기 config.json은 바이트 단위로
동일하다. 검증된 운용 범위·물리 중단 기준은 null이고 F0는 미확정 상태를 유지한다.
새로운 힘 상한이나 보정 완료 이력을 임의로 만들지 않았다.

재현: `OPENBLAS_NUM_THREADS=2 /usr/bin/python3 reports/20260927_E2_F0_offline_assumption/check_assumed_force.py`


---

<a id="readme-source-32"></a>

# 원본: `reports/20260928_E2_A_restore/README.md`

# E2 A 복구 및 검증

소프트웨어 준비 완료, 실기 preflight는 네 항목 차단 상태다.

- 독립 학습 A epoch 500 체크포인트와 6차원 정규화, force observation/history/action 제거된 모델 경로 복구.
- 누락된 추론/공통 실행기/기록 연결과 ROS 실행 항목 복구. 현재 E1 전용 자동 marker 코드는 보존.
- 가공 중 외부 +23 N 요청, 접근/종료 시 0 N 요청, 공통 gate/slew 유지.
- B/C 10개 보관 config와 executor/postprocessor/sampling/task 동일.
- `tests_final.log`: 188 passed, 레거시 rule/replay fixture 의존 3개 제외.
- `build.log`: ROS 패키지 빌드 성공.
- `offline_launch_host.log`: 실제 학습 모델 추론 및 500 tick synthetic executor 검사, launch 정상 종료.
- `launch_check_final.log`: 모델 A/B/C 및 코드 계약 확인. A 실기 차단 근거 4개 명시.
- `restoration.json`, `before/`, `runtime_changes.patch`: 복구 출처와 변경 전 파일 보존.
- `summary.json`: 최종 상태 및 오프라인 결과 경로.

실제 로봇/스핀들을 구동하지 않았다. `limits_evidence`, `sensor_overload_evidence`,
`workspace_evidence`, `approach_exit_evidence`는 기존 확인 기록에서도 미완료다.
새 승인이나 가공 품질 검증값으로 대체하지 않았다. 최종 실행 안내는
`experiments/e2_A_pilot_20260927/RUN_A.md`에 있다.


---

<a id="readme-source-33"></a>

# 원본: `results/20260920/README.md`

# 2026-09-20 B·C 결과 모음

현재 결과표의 B: **16:23:41–16:24:24 — 9/16 E1 OFF**

현재 결과표의 C: **17:17:26–17:18:13 — 9/16 E1 ON, 가장 최근 실행**

- `BC_results_20260920.csv` / `.xlsx`: 갱신된 비교표.
- `B_force_obs_OFF/`: B 원본 CSV·메타데이터·영상·removal 그림.
- `C_force_obs_ON/`: 오늘 9/16 ON C 실행 2회. 비교 대상으로 선택된 것은 `selected_runs.json` 참조.
- `archive/C_baseline_20260910_ckpt/`: 기존 9/10 baseline 기록(현재 C 비교표에서 제외).
- `archive/incomplete_runs/`: 오늘 초기 정렬에 실패해 정책 추론을 시작하지 못한 5회 기록.
- `archive/previous_BC_comparison/`: 갱신 전 B vs 9/10 baseline 표와 분석 보존.
- `runs_index.csv`: 오늘 9개 실행과 선택 여부, 영상 상태.
- `selected_runs.json`: 현재 B/C 원본 경로와 ckpt 및 영상 검사 결과.
- `manifest.json`: 복사 원본·대상 경로, 크기, SHA-256.
- `report.md`, `comparison.json`, `comparison.png`: 최신 비교의 수치·방법·그래프.

영상은 총 4개를 보존했다. B, 최신 C, 17:15 C 영상은 정상 마무리됐다.
기존 16:06 baseline 영상은 원본의 파일 끝부분 불완전 상태를 그대로 보존하고 표시했다.
원본 로그·영상은 그대로 두고 복사했으며, A 실행 결과는 없다.

공통 비교 구간은 첫 plan 이후 0–37.5초, 로그는 최대 20Hz다.
표는 기술적 단일 실행 비교이며 실제 제거율·작업 성공률·통계적 우열을 주장하지 않는다.


---

<a id="readme-source-34"></a>

# 원본: `results/20260926/E1/README.md`

2026-09-26 E1 전체 결과
======================

오늘 보존한 R 5회, T 5회, C 5회(총 15회)의 데이터와 영상 전체 복사본이다.
원본 logs 및 Videos/Screencasts는 그대로 유지했다. 비정상 C 3회는 사용자 요청으로 삭제되어 이 묶음에 포함하지 않았다.

각 방법 폴더의 실행 태그별 구성:

```text
R 또는 T 또는 C/
  RTC_<방법>_20260926T<시각>/
    executor/        실행기 CSV·이벤트·요약·소스/설정 스냅샷
    provider/        예측/경로 CSV·계획 NPZ·센서·시작/종료 이미지·소스 스냅샷
    launch_context/ 실행 설정
    plots/          궤적·힘·제거량 그래프와 summary
    video/          원본 WebM 영상
    ros_logs/       해당 실행의 launch 및 노드 로그
```

[실행 목록 CSV](run_index.csv)에 각 실행의 상대 경로, 영상 경로, 종료 상태, 기록 수와 설정 해시를 기록했다.
[파일 목록·SHA-256](manifest.json)은 복사된 모든 원본 파일의 출처와 상대 경로를 연결한다.
감사 결과와 C 패치 설명은 `audit/`에 함께 저장했다. 원본 audit 문서와 원본 metadata 내부의 절대 경로는 출처 보존을 위해 수정하지 않았다.

| 방법 | 보존 횟수 | 기록/종료 상태 |
|---|---:|---|
| [R](R/) | 5 | 가공 구간 기록 완료; 이후 복귀 timeout. normal_completion은 아님 |
| [T](T/) | 5 | 재생과 정지 기록 완료; 최종 목표 도달 미확인 |
| [C](C/) | 5 | 작업자 finish 후 normal_completion 및 정지 유지 확인 |

C에는 35점 궤적 평활·재계획 연결·가속도 제한 패치가 적용됐다. R/T 실행 조건은 오늘 원본을 유지했다.
사용자는 C 5회에서 의도하지 않은 동작이 없었다고 확인했다. 접촉 판정 전환은 분석 지표로 보존했다.
물리 가공 품질·실험 영역의 독립성·순수 가공 시간은 이 파일 정리 작업에서 새로 판정하지 않았다.
영상과 기록은 성공 여부를 선택적으로 보정하거나 편집하지 않은 원본 복사본이다.


---

<a id="readme-source-35"></a>

# 원본: `results/20260926/E1/audit/C/README.md`

2026-09-26 C 5회 확인 및 비정상 로그 정리
======================================

패치 후 C 5회 모두 기록 검사를 통과했고, 작업자의 `operator_finish` 요청 이후 `normal_completion`, `controller_hold_verified=true`, `queue_cancel_verified=true`가 확인됐다. 사용자는 다섯 실행에서 의도하지 않은 동작이 없었다고 확인했다. 이 5회는 보존하고, 실행 시작 전 중단 2회와 패치 전 진동으로 중단한 1회의 로그를 삭제했다.

현재 오늘의 보존 실행은 **R 5회 + T 5회 + C 5회 = 15회**다. R/T의 기록과 설정은 변경하지 않았다.

보존된 C 실행
------------

| 실행 태그 시각 | 실행 시간(초) | TCP/힘 행 | 실제 발행 명령 | 계획 수 | 영상(초) | 결과 |
|---|---:|---:|---:|---:|---:|---|
| 18:43:57 | 16.937 | 445 / 445 | 2,117 | 5 | 17.8 | 정상 기록·종료·정지 확인 |
| 18:45:21 | 13.635 | 399 / 399 | 1,704 | 4 | 14.9 | 정상 기록·종료·정지 확인 |
| 18:46:22 | 16.030 | 436 / 436 | 2,003 | 4 | 16.7 | 정상 기록·종료·정지 확인 |
| 18:47:24 | 15.216 | 413 / 413 | 1,901 | 4 | 16.0 | 정상 기록·종료·정지 확인 |
| 18:48:20 | 15.504 | 423 / 423 | 1,937 | 4 | 16.3 | 정상 기록·종료·정지 확인 |

검사 내용
---------

각 실행의 provider/executor 세션 ID가 일치하며 다섯 세션은 서로 다르다. 실행 설정 해시는 다섯 실행에서 동일하고, `c_pose_conditioning_20260926_v1` 및 패치된 코드 해시도 일치한다. CSV 행 구조·유한값·시간 순서·summary 기록 수를 검증했다. 로거 drop, 쓰기 오류, 종료 시 미처리 큐는 모두 0이며 기록이 완료됐다.

C 명령의 네 단계 `time_sampled` → `pose_conditioned` → `contact_gated` → `node_sent`가 빠짐없이 대응한다. 각 예측 계획은 원본 및 후처리 단계에 128점씩 기록됐다. 명령 CSV 전체 행 수는 실제 발행 횟수의 4배다.

다섯 실행의 최대 명령 발행 기록 간격은 8.694 ms였다. 위치 속도는 축별 10 mm/s, 위치 명령 가속도는 축별 25 mm/s² 제한을 지켰고, 회전 속도·가속도 및 힘 변화율 제한도 통과했다. 실행 중 TCP/힘 관측의 최대 공백은 약 138.876 ms로 기존 200 ms freshness 범위 안이었다. 이는 로거 검사 결과이며 헤더 없는 원센서·미들웨어 손실량까지 확인한 것은 아니다.

영상 5개를 전체 디코딩했고 오류가 없었다. 그래프 20개와 시작/종료 이미지 10개도 읽기 검사를 통과했다. 각 launch의 5개 노드가 모두 clean exit로 종료됐다. provider의 `run_interrupted`는 executor 정상 종료 확인 후 Ctrl-C로 기록 프로세스를 마감한 것으로, 가공 도중 중단된 실패 실행과 구분했다.

접촉 판정 전환은 원본 그대로 보존했다. 첫 4초 전환 횟수는 순서대로 0, 58, 40, 4, 8회이며, 사용자의 관찰 확인을 반영해 이 값만으로 실행을 제외하지 않았다. 실측 base-frame Fz 최대치는 순서대로 84.411, 51.789, 65.338, 81.195, 82.984 N이었다. 이 지표들은 모델/실행 동작 분석용이며, 이번 기록 검사에서 별도 원인 분리를 수행한 것은 아니다.

삭제 내역
---------

| 제외 태그 | 이유 |
|---|---|
| `RTC_C_20260926T175117` | 실행 시작 전 수동 중단; 추종 명령 없음 |
| `RTC_C_20260926T175131` | 패치 전 심한 진동을 보고한 실행; 수동 중단 |
| `RTC_C_20260926T184718` | 실행 시작 전 수동 중단; 추종 명령 없음 |

위 3회에 직접 연결된 metrics 폴더 4개, 실행 context 3개, 그래프 폴더 3개, 영상 1개, ROS launch 폴더 3개와 해당 노드 로그 15개를 삭제했다. 합계 **29개 경로, 109개 파일, 14,450,113 bytes**다. 다른 날짜의 실험과 공유 원격 제어기 로그는 삭제 범위에 포함하지 않았다.

삭제 직전 파일 목록·크기·inode·수정 시각·SHA-256을 대조했다. 삭제 후 C 보존 파일 285개와 기존 R/T 파일 580개의 SHA-256이 삭제 전과 같음을 확인했다. 총 865개 보존 파일 검증을 통과했다.

기록 해석과 자료 위치
--------------------

여기서 정상 종료는 작업자 종료 요청 후 제어기의 정지 유지가 확인됐다는 의미다. C에는 자동 접촉 해제·홈 복귀 완료가 기록되지 않으며, 독립적인 가공 품질/제거량 성공 판정은 별도다. 작업자 `processing_start/end` 마커도 없어 표의 실행 시간을 순수 가공 시간으로 해석하지 않는다.

기존 진동 진단과 패치 분석 보고서는 보존했다. 사용자 요청에 따라 해당 보고서의 입력이었던 패치 전 C 원본 로그를 삭제했으므로, 그 과거 재생 스크립트는 삭제된 입력 없이 재실행할 수 없다. 기존 분석 결과와 원본 해시·이번 삭제 이력은 남아 있다.

- [보존 실행 목록·로그 경로 CSV](retained_runs.csv)
- [전체 검사·삭제 목록·보존 해시](cleanup_manifest.json)
- [읽기 전용 검사 및 삭제 목록 생성 스크립트](audit.py)
- [정확한 삭제 목록 적용 스크립트](cleanup.py)


---

<a id="readme-source-36"></a>

# 원본: `results/20260926/E1/audit/C_patch/README.md`

2026-09-26 C 실행 패치
====================

C에만 궤적 평활, 재계획 연결, 위치/회전 가속도 제한을 적용했다. 사용자가 오늘 측정한 R/T를 유지하도록 요청했으므로 R/T의 경로와 실행 조건은 보존했다. 기존 C 실행 명령에서 새 패치가 로드되는 것을 무구동 check 모드로 확인했다. 실제 로봇은 실행하지 않았으며 진동 해소와 힘 응답의 하드웨어 검증은 남아 있다.

적용 내용
---------

현재 [config.json](../../experiments/e2_rule_replay_20260920/config.json)의 `il.pose_conditioning`에 `c_pose_conditioning_20260926_v1`을 추가했다. `common`, `executor`, `recipe`, `replay`와 기존 `il` 항목은 수정 전 및 오늘 실행 로그의 설정과 일치한다. 공유 실행기의 `execution_method=il`에서만 새 처리를 활성화한다.

| C 전용 처리 | 적용 값/동작 |
|---|---|
| 위치 평활 | 기존 서비스 C의 35점 중심 이동평균과 같은 XYZ 처리; 예측 구간 내부에서 계산 |
| 자세 평활 | 같은 35점의 quaternion 평균; 회전벡터 ±π 경계에서 잘못된 회전 방지 |
| 최초 연결 | 실제 시작 자세에서 새 목표까지 0.5초 quintic 가중치로 연결 |
| 재계획 연결 | 이전 기준 궤적의 위치·속도에서 0.5초 동안 새 기준 궤적으로 연결; 연결 도중 새 계획이 와도 연속 상태 유지 |
| 위치 명령 가속도 | 축별 25 mm/s², 정지 상태에서 기존 10 mm/s까지 최소 0.4초 |
| 회전 명령 가속도 | 축별 100 deg/s², 기존 축별 40 deg/s 속도 한계 유지 |
| 작업영역 검사 | 원본 목표, 평활·연결한 목표, 가속도 제한 후 발행할 명령 모두 기존 공통 범위 검사 |

공통 125 Hz 발행, gain 15/s, 축별 위치 속도 10 mm/s, 회전 속도 40 deg/s, 힘 변화율 30 N/s를 유지했다. 힘 목표의 부호·값·시간축, 접촉 ON/OFF 3.0/1.2 N, 힘 클리핑 설정, 피드백 freshness/watchdog와 정지 확인 절차도 유지했다. 학습 모델·정규화·force observation은 변경하지 않았다.

C의 원본 30 Hz/128점 시간축과 재계획 주기를 보존한다. 이번 패치는 원본 시간축을 늘리는 거리 기반 retiming을 추가하지 않는다. 원본 예측은 그대로 기록하고, 발행 전에 C의 위치/자세 명령만 안정화한다. 가속도 제한에 따른 추가 추종 지연과 경로 차이는 C 처리의 일부다.

검증 결과
---------

관련 테스트 93개가 통과했다. C 평활·연결·속도/가속도 제한, 반복 재계획, 정지 목표 수렴, 회전 경계, 힘 처리 불변, 작업영역/피드백 중단, R/T 유한 시간축, R 복귀, ROS launch 인자 전달과 기록을 검사했다. 노드나 프로세스를 실행하지 않는 launch 확장 테스트도 포함한다.

오늘 R 5회와 T 5회의 저장된 계획·시간·접촉 상태를 이전 코드와 새 코드에 동일하게 입력했다. R은 회당 4,085틱, T는 회당 1,864틱으로 **총 29,745틱의 원본/게이트 후/발행 9차원 명령이 바이트 단위로 동일**했다. R의 별도 복귀 구현은 그대로이며 기존 복귀 테스트도 통과했다. 원본 로그를 수정하거나 삭제하지 않았다.

진동 C `RTC_C_20260926T175131`의 1,776틱 입력 재생 결과:

| 항목 | 패치 전 | 패치 후 |
|---|---:|---:|
| 첫 4초 X 방향 전환 | 13회 | 1회 |
| 첫 4초 Y 방향 전환 | 12회 | 2회 |
| 최대 XYZ 축별 명령 가속도 | 약 2,500 mm/s² | 25 mm/s² |
| 재계획 접수 시 기준 위치의 불연속 | 원본 요구 궤적에서 최대 33.3 mm 변화가 관측됨 | 세 경계 모두 0 mm |
| 축별 위치 속도 상한 | 10 mm/s | 10 mm/s |
| 힘 목표·게이트·힘 발행 결과 | 기준 | 동일 |

방향 전환은 속도가 +1 mm/s와 -1 mm/s를 오갈 때 집계했다. 영점 부근의 작은 변화는 제외한다. 이전 진단의 X 10회/Y 8회는 인접 두 틱에서 바로 반전한 횟수이므로 위의 전체 방향 전환 횟수와 정의가 다르다. 그래프는 실제 로봇 가속도나 패치 후 센서 데이터가 아닌, 계산된 명령을 비교한다. 경계의 0 mm 역시 기준 궤적의 연속성 검사다.

수정 후 실행기 연산 시간은 로컬 오프라인 재생에서 틱 p99 약 0.20 ms, 새 계획 준비 약 0.46 ms였다. 이 수치는 실제 ROS 부하나 원격 전송의 시간 보장이 아니다. 설치된 Python 모듈과 launch는 수정한 소스에 연결된 symlink임을 확인했으며, 설치 환경의 C `mode:=check`에서 새 프로필 이름과 검사 통과가 출력됐다. 추가 패키지 설치나 제어 PC 변경은 필요하지 않았다.

남은 확인과 비교 해석
--------------------

접촉 판정과 힘 제어 조건은 R/T 기준을 유지하기 위해 바꾸지 않았다. 오프라인 재생에는 기록된 접촉 상태를 그대로 입력했으므로, 진동 분석에서 보인 접촉 게이트 반복 전환과 큰 실측 힘 피크가 해소됐다는 뜻은 아니다. 자세 명령이 달라졌을 때 실제 접촉과 힘 응답이 어떻게 바뀌는지는 첫 C 검증 실행에서 확인해야 한다. 첫 실행부터 정상 데이터로 확정하지 말고, 반복 진동이 나타나면 기존 중단 절차로 종료한다.

R/T와 C는 같은 전달 방식·공통 제어 상수를 쓰지만, **C에만 추가한 평활·연결·가속도 제한은 비교 방법에 명시해야 한다.** 기존 R/T를 보존한 채 C의 전체 구성 성능을 비교할 수 있으며, 차이를 모델 자체의 효과만으로 분리해 해석할 수는 없다. `execution_equivalence=unverified`는 유지했다. 기존 R/T의 완료 상태를 이번 패치로 재분류하지 않았다.

실행
----

기존 ROS 환경 터미널에서 같은 C 명령을 사용한다. 콘솔에 `C pose conditioning: c_pose_conditioning_20260926_v1`이 표시된다. `mode:=check`는 노드를 시작하지 않는 검사이며, 아래 `mode:=run`은 사용자가 실행하는 로봇 구동 명령이다.

```sh
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash

ros2 launch nrs_imitation rtc_timed.launch.py \
  method:=C \
  config:=/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json \
  mode:=run \
  checkpoint:=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/on/20260916_1531/policy_best.ckpt
```

R/T 명령은 기존대로다. 요청에 따라 R/T 재측정이나 자동 로봇 구동은 수행하지 않았다. 기존 가공 구간 표시·finish·abort 사용법은 [실행 안내](../../experiments/rtc_launch_20260926/README.md)를 따른다.

로그와 재현
-----------

C executor metadata/`plan_received` 이벤트에 `c_pose_conditioning`을 기록하고, 명령 CSV에 `pose_conditioned` 단계를 추가했다. C는 `time_sampled` → `pose_conditioned` → `contact_gated` → `node_sent` 네 단계다. R/T는 기존 세 단계를 유지한다. C 명령 행의 details에는 `conditioning_profile`이 기록된다.

- [명령 비교 그림](command_comparison.png), [PDF](command_comparison.pdf)
- [재생 수치와 원본 파일 해시](recorded_validation.json)
- [검증 요약 및 적용 파일 해시](validation.json), [패치 diff](changes.patch)
- [오프라인 재현 스크립트](validate_recorded.py): `OPENBLAS_NUM_THREADS=2 /usr/bin/python3 reports/20260926_C_execution_patch/validate_recorded.py`
- [수정 전 파일 해시](before_sha256.json), [수정 전 소스·설정](before/)
- [C 회귀 테스트](../../behavior_ws/src/nrs_imitation/test/test_c_pose_conditioning.py)

후속 기록: 패치 후 사용자가 C 5회를 실행했고, [C5 검사 및 정리](../20260926_C5_log_audit/README.md)에서 기록 완료·작업자 종료·정지 유지를 확인했다. 사용자 요청으로 패치 전 비정상 C 원본 입력 로그는 삭제했다. 위 오프라인 비교 결과와 당시 원본 해시는 보존했지만, `validate_recorded.py`는 삭제된 과거 입력 없이 재실행할 수 없다. C의 접촉 판정 전환은 분석 지표로 유지했다.


---

<a id="readme-source-37"></a>

# 원본: `results/20260926/E1/audit/R/README.md`

2026-09-26 R 실행 로그 정리 결과

총 20회 중 가공 provider 궤적을 끝까지 실행하고 가공 구간의 힘·TCP 로그가 완전한 5회를 보존했다. 시작 실패·가공 전 중단·접근 중 중단 15회의 관련 파일은 사용자 요청에 따라 삭제했다.

보존 기준은 가공 구간의 기록 완전성이다. 다섯 실행 모두 이후 retract arrival/contact-release verification timeout으로 종료되었으며, 홈 복귀를 포함한 normal_completion은 0회다.

| 실행 태그 | 가공 구간(초) | 전체 TCP/힘 행 | 가공 구간 TCP/힘 행 | 로그 경로 |
|---|---:|---:|---:|---|
| RTC_R_20260926T170642 | 17.104237 | 982/982 | 306/306 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T170642_executor_20260926T170643_1790410003762802082_hjyzh_8_) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T170642_20260926T170644_1790410004435588336_fhh5hznt) |
| RTC_R_20260926T171118 | 17.103958 | 941/941 | 310/310 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171118_executor_20260926T171119_1790410279662742419_mayax94c) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171118_20260926T171120_1790410280349607268_043tznwz) |
| RTC_R_20260926T171328 | 17.103937 | 919/919 | 306/307 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171328_executor_20260926T171329_1790410409421232182_9nhzzix_) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171328_20260926T171330_1790410410101483480_p54gawqx) |
| RTC_R_20260926T171650 | 17.104136 | 1041/1041 | 308/308 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171650_executor_20260926T171650_1790410610994845562_axfuiaxp) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T171650_20260926T171651_1790410611685746589_edv9q_l7) |
| RTC_R_20260926T172034 | 17.103830 | 909/908 | 306/306 | [실행](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T172034_executor_20260926T172035_1790410835446352027_7n9a8ps6) · [provider](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_R_20260926T172034_20260926T172036_1790410836108075672_6_j6yp4s) |

다섯 실행 모두 동일 config SHA256, 서로 다른 session ID. CSV 파싱/유한값/기록 수/시간 순서를 검증했고 로거 drop, 쓰기 오류, 미처리 큐는 모두 0이다. 가공 구간의 TCP/힘 최대 관측 간격은 61.85 ms 이내이며, 경계 포함 100 ms 이상 공백은 없다. 각 실행의 processing 단계에서 실제 발행된 명령 기록은 2,138개이고 contact gate는 모두 true다. 전체 commands.csv는 time_sampled/contact_gated/node_sent의 3단계 기록이므로 행 수를 실제 발행 횟수로 해석하면 안 된다.

영상 5개를 ffprobe로 디코딩 확인했고, 각 실행의 그래프 4개와 시작/종료 이미지 2개 및 두 NPZ 계획 파일의 유한값/시간 증가를 확인했다. 삭제 후 보존 대상 295개 파일의 SHA256이 삭제 전과 같음을 확인했다.

processing_start/end 작업자 이벤트는 없어, 분석 구간은 provider_phase processing부터 force_ramp_out까지의 대용 구간이다. 실제 가공 성공이나 시편/영역 5개의 독립성을 판정한 것이 아니다. 헤더 없는 센서 피드백이므로 미들웨어/원센서 손실량은 확정할 수 없다.

삭제 범위: 오늘 RTC R의 제외된 실행에 직접 연결된 metrics 폴더 28개, run context 15개, 그래프 폴더 15개, 영상 15개, ROS launch 폴더 15개와 노드 로그 73개. 총 811개 파일, 67,679,774 bytes.

[보존 실행 목록 CSV](retained_runs.csv) · [삭제 목록 및 검증 결과](cleanup_manifest.json)


---

<a id="readme-source-38"></a>

# 원본: `results/20260926/E1/audit/T/README.md`

2026-09-26 T 5회 로그 확인 결과

T 실행 5회 모두 재생 궤적을 끝까지 소비하고 정지 hold를 확인했으며, 데이터 기록 검사를 통과했다. 오늘 T의 시작 실패/중도 중단 기록은 없어 삭제 대상은 0개다. 기존 R 5회와 T 5회를 보존했다.

CSV 행 구조, 유한값, 시간 순서, summary 기록 수, 로거 drop/쓰기 오류/미처리 큐를 확인했다. 영상 5개 전체 프레임을 ffprobe로 읽고 그래프 20개와 시작/종료 이미지 10개도 검증했다. 각 T의 provider_prediction/postprocessed는 443행씩, 실제 node_sent 명령은 1,864개다. 모든 launch의 5개 노드가 정상 종료했다.

| 시작 시각 | TCP 행 | 힘 행 | 재생 길이(초) | 영상(초) | 최종 목표 오차(mm) | 실행 로그 |
|---|---:|---:|---:|---:|---:|---|
| 17:28:27 | 514 | 513 | 14.910 | 29.2 | 14.991 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T172827_executor_20260926T172828_1790411308738816535__tr5tupl) |
| 17:41:11 | 452 | 451 | 14.910 | 25.9 | 19.498 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174111_executor_20260926T174112_1790412072875639201_iupu_04x) |
| 17:42:16 | 445 | 445 | 14.910 | 25.3 | 20.823 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174216_executor_20260926T174217_1790412137101159951_6tdsli24) |
| 17:43:11 | 457 | 456 | 14.910 | 25.9 | 20.187 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174311_executor_20260926T174312_1790412192846237668_x7gp49fu) |
| 17:44:14 | 489 | 489 | 14.910 | 27.8 | 19.522 | [열기](/home/eunseop/nrs_imitation/logs/inference_metrics/RTC_T_20260926T174414_executor_20260926T174415_1790412255162873318_n99zdbbj) |

다섯 실행 모두 final_target_reached=false이고 normal_completion 이벤트가 없다. 따라서 판정은 데이터 기록 완료이며 물리 작업의 정상 완료가 아니다. 가공 시작/끝 작업자 마커가 없으므로 14.910초 전체 재생 길이를 순수 가공 시간으로 해석하지 않는다.

실행 중 executor TCP/힘 관측 간격은 최대 60.903 ms, provider 관측 간격은 최대 70.858 ms였다. 실제 발행 명령 간격은 최대 9.222 ms였다. 관측된 로거 누락/쓰기 오류는 0이며, 헤더 없는 센서 피드백의 미들웨어 손실량은 별도로 확정할 수 없다.

기존 R 보존 파일 295개의 SHA256이 이전 정리 때와 일치함을 확인했다.

[보존 실행 CSV](retained_runs.csv) · [상세 검사 JSON](audit.json)


---

<a id="readme-source-39"></a>

# 원본: `results/20260926/E2/README.md`

# 2026-09-26 E2 B/C 결과

B(힘 관측 OFF) 5회, C(힘 관측 ON) 5회, 총 10회의 정상 실행 기록과 원본 영상이다.
두 조건 모두 학습된 힘 출력을 사용하며 E1 C 기반의 동일한 실행·평활·정지 로직을 공유한다.
A는 아직 실행하지 않았으며 이 결과에 포함하지 않았다.

- `B/<실행 ID>/`, `C/<실행 ID>/`: E1과 동일한 조건별 보관 구조
- `executor/`, `provider/`: 명령·센서 CSV, 이벤트, 원본 메타데이터와 이미지
- `launch_context/`: 실행 설정, 시도 정보, 코드 스냅샷, 원본 archive_manifest
- `plots/`, `video/`, `ros_logs/`: 그래프·원본 WebM·관련 ROS 로그
- `run_index.csv`: 10회 목록, 종료 상태, 데이터/영상 경로
- `manifest.json`: 원본 파일 출처·크기·SHA-256 및 보관본 무결성
- `audit/cleanup_manifest.json`: 시도 12회 중 정상 10회 보존, 시작 실패 B 2회 삭제 근거
- `analysis/`: 실행 구간 수치 CSV/JSON 및 보고서 검증
- 상위 날짜 폴더의 `E2_BC_final_report_20260926.pdf`: 단일 통합 보고서

정상 원본은 `E2_BC_01/{B,C}/`에도 유지했다. 원본 파일 내부의 절대 경로는 출처 보존을 위해 수정하지 않았다.
비정상 B 두 시도의 원본 폴더와 해당 전용 ROS launch 로그 폴더는 사용자 요청에 따라 삭제했다.
삭제한 원시 로그는 보관하지 않았고 삭제 이유와 파일 해시 목록만 audit에 남겼다. C에는 비정상 시도가 없다.

10회 모두 operator_finish, normal_completion 및 실제 정지 유지 확인이 있다.
접촉 판정 전환 횟수만으로 실행을 제외하지 않았다. 실행 기록 정상과 물리 가공 품질은 별개다.
processing_start/end 마커가 없어 순수 가공 시간은 미확정이며, 실행 시간은 접근을 포함한다.
힘은 robot-base Fz의 기술 통계이며, 미검증 법선 힘·교사 기준 프로파일·제어기 적용 힘 오차는 NA다.
시편 상태는 사용자 지시에 따라 매회 동일하다고 가정했다. B 5회 뒤 C 5회가 수행되어 순서 효과는 분리할 수 없다.

영상 범위: C5의 녹화 시작은 execution_start보다 4.223초 늦었다. 원본 명령·센서 기록은 정상이며 실행은 보존했다. 나머지 9회의 녹화 시작 지연은 0.139~0.201초다. 전체 영상 파일의 디코딩 성공과 실행 전 구간 촬영 여부는 구분한다.


---

<a id="readme-source-40"></a>

# 원본: `results/table/20260920/README.md`

# 2026-09-20 E1 / E2 — RA-L 형식 결과표

**최신 논문 삽입용 이미지:** [E1_RAL.png](ral/E1_RAL.png), [E2_RAL.png](ral/E2_RAL.png).
`ral/`에는 각 방법·변수 정의를 영문 각주에 포함한 LaTeX 조판 표(600 dpi PNG, 벡터 PDF, 원고용 table* 소스)가 있습니다. 발표 디자인의 `slides/`와 구분됩니다.

저장 경로: `results/table/20260920/` (실험 날짜 기준).

- **A: 보류.** A의 수치나 성공률은 만들지 않았습니다.
- **E1:** 기존에 선택한 16:23 B(9/16 OFF), 17:17 C(9/16 ON), 각 1회.
- **E2:** 19:58 편도 R, 20:10 episode_29 T, 20:13 C 1차, 20:14 C 2차를 모두 포함했습니다.
- 이전 19:38 왕복 R도 `E2_all_started_runs.csv` / Excel 같은 이름의 시트에 수치를 보존했습니다. 편도 R과는 합치지 않았습니다.

## 파일

- `E1_E2_tables.png`: 두 표를 합친 600 dpi 이미지.
- `E1_table.png`, `E2_table.png`: 각각의 논문용 영문 표. PDF/SVG 벡터 파일도 포함.
- `E1_E2_results.xlsx`: E1, E2, 원정밀도 수치, 모든 E2 실행, 실행 이력, 조건, 지표 정의의 7개 시트.
- `E1_results.csv`, `E2_results.csv`, `E1_E2_results.csv`: Excel 호환 UTF-8 BOM CSV, 숫자 원정밀도 유지.
- `run_audit.csv`: E1의 9개 기존 기록과 E2의 8개 시도를 포함한 이력/선택 근거.
- `tables_ieee_ral.tex`: IEEEtran의 `table*`용 표 소스; `booktabs,graphicx` 사용.
- `metrics_and_provenance.json`, `source_manifest.json`, `validation.json`: 상세 수치·종료 이벤트·원본 SHA-256·검증 결과.

## 해석

E1은 첫 정책 plan 이후 공통 0–37.5초를 재계산했으며, 기존 선택 B/C 결과와 수치가 일치합니다.
E2 수치는 실제 execution_start부터 첫 stop_requested까지입니다. R의 복귀 구간은 힘/경로 통계에 포함하지 않되, 복귀 실패는 결과 상태에 반영했습니다.

평균±표준편차는 **한 실행 내 시간 가중 힘 변동**입니다. 반복 실험 간 표준편차, 신뢰구간, 유의성 검정이 아닙니다. E2의 서로 다른/중단된 관측 시간을 완료 작업시간으로 비교할 수 없습니다.

R은 편도 경로 후 수직 상승을 수행했지만 접촉 해제 확인 시간 초과로 교시 시작 XY 복귀를 시작하지 못했습니다. T는 재생 시간 종료와 정지 유지가 확인됐으나 마지막 목표 도달과 접촉 해제는 미확인입니다. C 1차는 수동 중단 후 정지 유지 확인, C 2차는 Force 모드 상실/갱신 중단 후 STOP 응답과 정지 유지 미확인입니다. 두 C 시도는 진동이 보고된 실행입니다.

Fz는 로봇 base Z축의 필터된 측정 힘입니다. 검증된 표면 법선력이나 압력이 아닙니다. 기록은 최대 20 Hz이므로 모든 고주파 피크를 측정했다고 볼 수 없습니다. τ3/τ50는 3/50 N 이상 시간이며 성공/접촉/안전 판정이 아닙니다.

실측 제거량, 얼룩 제거율, 물리 작업 성공률 및 검증된 힘 추종 RMSE는 **미평가**입니다. k=1 removal heatmap을 실제 제거량으로 대체하지 않았습니다. 회전수는 미확인이고 시편/얼룩 초기화 동일성도 검증되지 않았습니다.

E1의 service_stream(35점 평활화)과 E2의 timed_topic 실행부가 다릅니다. 같은 C 체크포인트라도 E1/E2를 동등한 실행 조건으로 합산하거나 성능 우열을 주장하면 안 됩니다. R의 후처리 복귀 절차도 T/C와 다릅니다. 표는 현재 파일럿 실행의 기술적 기록입니다.

## 형식과 재현

RA-L의 IEEE 2단 형식 안내를 참고해 전폭 7.16 inch 영문 표, 상단 캡션, 로마 숫자 표 번호, serif 글꼴과 수평선으로 구성했습니다. 최종 원고에서는 실제 문서의 표 번호/캡션과 함께 검토해야 합니다.
공식 안내: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/

재생성: `python3 scripts/export_ral_e1_e2_20260920.py`
로봇 실행/제어기 변경 없이 저장된 로그만 읽습니다. 기존 로그·CSV·영상·실험 manifest는 수정하지 않았습니다.


---

<a id="readme-source-41"></a>

# 원본: `results/table/20260920/ral/README.md`

# RA-L 원고 삽입용 표 이미지

사용자의 형식 정정을 반영해 논문 표 형태로 다시 제작했습니다.

- **E1_RAL.png**: E1 B/C의 측정 힘 입력 비교. B/C의 뜻, 두 조건의 목표 힘 학습 및 하위 힘 제어 유지, 각 지표 정의를 각주에 포함했습니다. A는 보류입니다.
- **E2_RAL.png**: E2 R(편도 규칙 +18 N), T(episode_29 재생), C1/C2(힘 관측 ON IL 두 시도)의 실행 결과. 각 방법·지표와 종료 상태를 각주에서 설명합니다.

두 이미지는 600 dpi, 흰 배경, Times 계열 serif 서체, 상단 TABLE I/II 캡션 및 수평선으로 구성했습니다. 표는 2단 원고의 전폭 7.16 inch를 기준으로 LaTeX에서 조판했고, 표 주위로 잘랐습니다.

- 논문에는 PNG보다 `E1_RAL.pdf`, `E2_RAL.pdf` 또는 원고용 `E1_E2_IEEE_tables.tex`를 권장합니다.
- 각 `E1_RAL.tex`, `E2_RAL.tex`는 독립 렌더링용 소스입니다. 실제 원고의 표 번호는 원고용 table* 소스를 넣으면 자동으로 결정됩니다.
- 형식 안내: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
- 수치는 상위 폴더의 검증된 CSV를 그대로 사용했습니다. 표의 ±는 시간 변동이며 반복 간 오차가 아닙니다. 중단/미확인 결과를 성공으로 바꾸지 않았습니다.
- τ₃·τ₅₀ 열 머리에 힘 기준 이상의 시간임을 표시했고, 첫 각주에 각 기준값과 누적 시간의 뜻을 풀어 썼습니다. **τ는 초 단위 시간이며 토크가 아님**을 이미지 안에 명시했습니다.

재생성: `python3 scripts/render_ral_paper_tables.py`
표 숫자, PDF 문자, 넘침 여부 및 PNG 크기를 자동 확인했으며 결과는 validation.json에 저장합니다.


---

<a id="readme-source-42"></a>

# 원본: `results/table/20260920/slides/README.md`

# PPT 삽입용 이미지 2장

두 파일 모두 16:9, 3840×2160 PNG입니다. 기존 논문용 원본 표는 그대로 보존합니다.

1. **E1_ppt_summary.png — 힘 observation ablation 설명과 결과**
   - B: 측정 힘 입력 OFF, 학습된 목표 힘 출력 ON.
   - C: 측정 힘 입력 ON, 학습된 목표 힘 출력 ON.
   - A는 보류이며 양쪽 모두 하위 힘 제어를 유지합니다.
   - 선택한 B/C 각 1회, 첫 plan 이후 공통 0–37.5초 수치와 Fz·SD·n·τ3·τ50·XY 거리·추론 지연의 정의를 포함합니다.

2. **E2_ppt_summary.png — R/T/C 방법 정의와 실행 기록**
   - R: 편도 규칙 경로, 가공 기준 힘 18 N.
   - T: episode_29의 위치·힘·시간 재생.
   - C: 측정 힘을 입력받고 동작·목표 힘을 출력하는 9/16 ON IL 정책.
   - R/T 각 1회와 C 두 번을 따로 표기하고, R 복귀 미완료·T 목표/접촉 해제 미확인·C 중단 상태를 표시합니다.

모든 수치는 상위 폴더의 검증된 CSV에서 읽어 소수 둘째 자리로 표시했습니다. 평균±SD는 한 실행 내 시간 변동으로, 반복 간 불확실성이나 통계적 우열을 의미하지 않습니다. 제거율·물리 작업 성공률은 미평가입니다.

PPT에서는 [삽입 → 그림]으로 PNG를 넣고 가로세로 비율을 유지해 크기를 조정하면 됩니다.
재생성: `python3 scripts/render_e1_e2_ppt_slides.py`

