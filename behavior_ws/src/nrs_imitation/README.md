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

기본 F710/XInput 매핑:

```text
RB: recorder start_recording
LB: recorder end_recording
A : gripper close (/gripper/command = 2500)
B : gripper open  (/gripper/command = 590)
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

이 policy는 `cam0 + qpos(position, force) + gripper state(position, current)`를 observation으로 사용합니다. gripper state encoder는 MLP이고, action target은 `position(6) + force(3) + gripper_present_position(1)`의 10D입니다. `action/gripper_present_current_mA`는 imitation_form에 보존되지만 command target으로는 사용하지 않습니다.

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
cam0     = /realsense/robot/color/image_raw
```

특정 checkpoint 지정:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  ckpt_dir:=/home/nrs_display/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>
```

ACT checkpoint를 직접 지정할 때는 `policy_class:=ACT`와 ACT checkpoint 경로를 같이 지정합니다.

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

위 launch 명령은 inference와 `rqt_image_view`를 같이 실행합니다.

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
