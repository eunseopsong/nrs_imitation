# nrs_imitation

이 저장소의 기본 학습 흐름은 아래 4단계입니다.

```text
HDF5 recording -> imitation_form 변환 -> Flow train -> ROS2 inference
```

현재 일반 데이터 파이프라인은 `single_cam`과 `dual_cam`으로 분기되어 있습니다.  
주의: dual-camera 데이터 폴더명은 recorder 내부 obs mode 기준으로 `multi_cam`입니다.

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
single_cam: ~/nrs_imitation/datasets/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
dual_cam  : ~/nrs_imitation/datasets/multi_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
```

### Single Cam

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam
```

빛반사 하이라이트를 줄인 `cam0` RGB를 저장:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam --ros-args \
  -p image_preprocess_mode:=specular_inpaint \
  -p image_specular_mask_mode:=bright \
  -p image_specular_v_thresh:=220 \
  -p image_specular_dilate_px:=2 \
  -p image_specular_inpaint_radius:=3.0
```

기본 입력:

```text
position = /calibrated_pose
force    = /ftsensor/measured_Cvalue
cam0     = /realsense/vr/color/image_raw
```

### Dual Cam

```bash
ros2 run nrs_imitation hdf5_recorder_dual_cam
```

기본 입력:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
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

## 2. Imitation Form 변환

변환 결과는 기본적으로 같은 run directory 아래 `imitation_form/`에 저장됩니다.

```text
single_cam: ~/nrs_imitation/datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
dual_cam  : ~/nrs_imitation/datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
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

`marker`, `qpos`, `action_flat`, `meta`, `is_pad`는 일반 imitation form에서 쓰지 않습니다.

### Single Cam

최신 single-cam recording을 자동 선택:

```bash
python3 source/custom/demo_data_imitation_form_single_cam.py --write_summary
```

특정 파일을 지정:

```bash
python3 source/custom/demo_data_imitation_form_single_cam.py \
  --input_h5 datasets/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form \
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
  --input_h5 datasets/multi_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

## 3. Train

Flow train도 `single_cam`과 `dual_cam`으로 분리되어 있습니다.  
`--dataset_dir`를 생략하면 각 dataset root 아래 최신 `imitation_form/episode_*.hdf5`를 자동 선택합니다.

checkpoint 기본 저장 위치:

```text
single_cam: ~/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>/
dual_cam  : ~/nrs_imitation/checkpoints/flow/polishing/dual_cam/<YYYYMMDD_HHMM>/
```

### Single Cam

```bash
python3 scripts/flow/train_flow_single_cam.py
```

특정 imitation_form 지정:

```bash
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

### Dual Cam

```bash
python3 scripts/flow/train_flow_dual_cam.py
```

특정 imitation_form 지정:

```bash
python3 scripts/flow/train_flow_dual_cam.py \
  --dataset_dir datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form
```

## 4. Inference

inference도 `single_cam`과 `dual_cam`으로 분리되어 있습니다.  
`ckpt_dir`를 생략하면 아래 경로에서 최신 `policy_best.ckpt`를 자동 선택합니다.

```text
single_cam: ~/nrs_imitation/checkpoints/flow/polishing/single_cam/*/policy_best.ckpt
dual_cam  : ~/nrs_imitation/checkpoints/flow/polishing/dual_cam/*/policy_best.ckpt
```

### Single Cam

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
```

기본 입력:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
```

특정 checkpoint 지정:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  ckpt_dir:=/home/nrs_display/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>
```

### Dual Cam

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
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

위 launch 명령은 inference와 `rqt_image_view`를 같이 실행합니다.

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
```

visualization topics:

```text
single_cam: /inference_single_cam/gradcam_overlay       # local Grad-CAM + xyz trajectory overlay
dual_cam  : /inference_dual_cam/gradcam_overlay         # local Grad-CAM + xyz trajectory overlay
dual_cam  : /inference_dual_cam/gradcam_overlay_global  # global Grad-CAM only
```

시각화 layer는 각각 끌 수 있습니다.

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py gradcam_enable:=false
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py trajectory_overlay_enable:=false
```

xyz 궤적이 너무 크거나 작으면 고정 스케일을 조절합니다.

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py trajectory_overlay_pixels_per_mm:=4.0
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
