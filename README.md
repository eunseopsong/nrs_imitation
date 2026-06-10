# nrs_imitation

Robotic polishing imitation-learning workspace.

Main flow:

```text
single_cam: teaching / HDF5 recording -> imitation_form conversion -> Flow/ACT training -> ROS2 inference
dual_cam  : stage1 VR teaching -> robot playback + dual-cam HDF5 recording -> imitation_form conversion -> Flow/ACT training -> ROS2 inference
```

The current polishing pipeline is split into:

- `single_cam`: one RGB camera, `cam0`
- `dual_cam`: stage1 trajectory first, then robot playback data with `cam0` and `cam1`

Folder naming note:

- `single_cam` data is saved under `datasets/single_cam`
- `dual_cam` recorder data is saved under `datasets/multi_cam`

## Quick Start

The commands below are written so that each stage can auto-select the latest recorded / converted / trained result.

### 0. Build

```bash
cd ~/nrs_imitation/behavior_ws
colcon build --packages-select nrs_imitation --symlink-install
source install/setup.bash
```

For Python scripts:

```bash
cd ~/nrs_imitation
```

## Quick Start: Single Cam

Use this when recording only `cam0`.

Expected recording topics:

```text
position = /calibrated_pose
force    = /ftsensor/measured_Cvalue
cam0     = /realsense/vr/color/image_raw
command  = /vr_demo_recorder/command
```

Expected inference topics:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
```

### 1. Start sensor / command inputs

Run the producer nodes you normally use for teaching. If these aliases are configured on the PC, this is the usual setup:

```bash
vive
ft
rsv
```

Optional joystick command node:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

### 2. HDF5 recording

Run this in a dedicated terminal and keep it running while recording episodes:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation hdf5_recorder_single_cam
```

To generate `observations/images/stain_mask` during conversion, use the stain-mask recorder:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam_stain_mask
```

Record `ep_0000` as a clean-surface reference before recording stained episodes. If the camera moves during recording, `ep_0000` must cover the same camera/tool motion on a clean surface; a fixed clean view is not enough for moving-view stain masks.

To record `cam0` after specular-highlight filtering:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam --ros-args \
  -p image_preprocess_mode:=highlight_attenuate \
  -p image_specular_mask_mode:=bright \
  -p image_specular_v_thresh:=220 \
  -p image_specular_dilate_px:=2 \
  -p image_specular_attenuate_gain:=0.35
```

Filter modes:

```text
image_specular_mask_mode:=white   # bright low-saturation white glare
image_specular_mask_mode:=bright  # bright colored glare, such as yellow/green reflection
```

Tuning:

```text
Lower image_specular_v_thresh -> removes more bright pixels
Higher image_specular_v_thresh -> keeps more of the original image
Higher image_specular_dilate_px -> expands the removed highlight region
Lower image_specular_attenuate_gain -> darkens detected highlights more
```

Use `highlight_attenuate` for broad colored reflections. Use `specular_inpaint` only for small white glare spots, because large inpaint masks can distort object and surface geometry.

Start one teaching episode:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: start_recording}"
```

End the episode:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: end_recording}"
```

Repeat start/end for more episodes. When finished, stop the recorder terminal with `Ctrl+C`.

Output:

```text
~/nrs_imitation/datasets/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
```

### 3. Convert to imitation_form

This auto-selects the latest single-cam merged HDF5:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_single_cam.py --overwrite --write_summary
```

For moving-camera stain masks, conversion first matches the current episode to the clean reference episode with monotonic pose-sequence DTW, then uses homography alignment, top-k reference consensus, a pose-distance guard, reference-diff core masking with constrained dark-prior expansion, temporal gap filling, and temporal pruning. If the log says `temporal-filled`, short mask dropouts were repaired from neighboring frames. If it says `temporal-pruned`, isolated mask components without support from nearby aligned frames were removed. If many frames still say `had no close clean reference`, re-record the clean reference sweep with the same motion, or relax `--stain_reference_max_pose_dist` only after checking the overlay.

Output:

```text
~/nrs_imitation/datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
```

### 4. Train policy

This auto-selects the latest single-cam `imitation_form`:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_single_cam.py
```

By default, Flow training uses `observations/images/stain_mask` and adds VIOLA-style masked feature pooling on top of the `cam0` RGB feature map. To train the RGB-only baseline instead:

```bash
python3 scripts/flow/train_flow_single_cam.py --no_stain_mask
```

ACT uses the same single-cam dataset root and common training arguments. It also uses `stain_mask` by default; add `--no_stain_mask` for RGB-only ACT:

```bash
cd ~/nrs_imitation
python3 scripts/act/train_act_single_cam.py
```

Checkpoint output:

```text
~/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>/
~/nrs_imitation/checkpoints/act/polishing/single_cam/<YYYYMMDD_HHMM>/
```

### 5. Inference

This auto-selects the latest single-cam `policy_best.ckpt` and opens the Grad-CAM viewer:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
```

Default inference uses Flow checkpoints under `checkpoints/flow/polishing/single_cam`. For ACT checkpoints, add:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py policy_class:=ACT
```

Checkpoints trained with the default stain-mask pooling require inference to receive a live `stain_mask` for `cam0`:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  use_stain_mask:=true \
  stain_mask_topic:=<live_cam0_stain_mask_topic>
```

If you trained with `--no_stain_mask`, keep inference RGB-only with `use_stain_mask:=false`.

Visualization topics:

```text
/inference_single_cam/gradcam_overlay  # local Grad-CAM + xyz trajectory overlay
```

Disable either visualization layer when needed:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py gradcam_enable:=false
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py trajectory_overlay_enable:=false
```

Tune xyz overlay scale if the curve is too small or too large:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py trajectory_overlay_pixels_per_mm:=4.0
```

## Quick Start: Dual Cam

The key difference from `single_cam` is that `dual_cam` uses a two-pass data collection flow:

```text
1. stage1: record VR teaching trajectory only
2. pusher: send selected stage1 episode to the robot playback PC
3. dual-cam recording: record robot playback with cam0 + cam1
```

Stage1 teaching topics:

```text
position = /calibrated_pose
force    = /ftsensor/measured_Cvalue
command  = /vr_demo_recorder/command
```

Dual-cam HDF5 recording topics:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
cam1     = /realsense/global/color/image_raw
command  = /vr_demo_recorder/command
```

Expected inference topics:

```text
position = /ur10skku/currentP
force    = /ur10skku/currentF
cam0     = /realsense/robot/color/image_raw
cam1     = /realsense/global/color/image_raw
```

### 1. Record stage1 VR teaching episodes

Start the VR pose and force producers. If these aliases are configured on the PC, this is the usual setup:

```bash
vive
ft
```

Optional joystick command node:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

Run the stage1 recorder in a dedicated terminal:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_stage1_hdf5_recorder
```

Start one stage1 teaching episode:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: start_recording}"
```

End the episode:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: end_recording}"
```

Repeat start/end for more stage1 episodes. When finished, stop the stage1 recorder terminal with `Ctrl+C`.

Stage1 output:

```text
~/nrs_imitation/datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes/episode_*.hdf5
```

### 2. Start robot playback inputs and dual cameras

Start the robot-side force/pose stream and the two cameras. If these aliases are configured on the PC, this is the usual setup:

```bash
ft
rsr
```

Also make sure this topic is being published by the global camera:

```text
/realsense/global/color/image_raw
```

### 3. Start stage1 episode pusher

This auto-selects the latest `datasets/stage1/*/stage1_vr_episodes` directory:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_stage1_episode_pusher
```

In the pusher terminal:

```text
Enter : push current stage1 episode to the robot playback PC
d     : next episode
a     : previous episode
q     : quit
```

### 4. HDF5 recording during robot playback

Run this in a dedicated terminal and keep it running while replaying pushed stage1 episodes:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation hdf5_recorder_dual_cam
```

For each pushed/playback episode, start dual-cam recording:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: start_recording}"
```

After robot playback finishes, end dual-cam recording:

```bash
ros2 topic pub --once /vr_demo_recorder/command std_msgs/msg/String "{data: end_recording}"
```

Repeat this for each stage1 episode that you replay. When finished, stop the dual-cam recorder terminal with `Ctrl+C`.

Output:

```text
~/nrs_imitation/datasets/multi_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
```

### 5. Convert to imitation_form

This auto-selects the latest dual-cam merged HDF5:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_dual_cam.py --overwrite --write_summary
```

Output:

```text
~/nrs_imitation/datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
```

### 6. Train policy

This auto-selects the latest dual-cam `imitation_form`:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_dual_cam.py
```

By default, dual-cam Flow training uses `cam0`, `stain_mask`, and `cam1` in that order. The `stain_mask` is applied only to `cam0`; `cam1` remains a normal RGB/global camera feature. To train the RGB-only baseline:

```bash
python3 scripts/flow/train_flow_dual_cam.py --no_stain_mask
```

ACT uses the same dual-cam dataset root and common training arguments. It also uses `stain_mask` by default; add `--no_stain_mask` for RGB-only ACT:

```bash
cd ~/nrs_imitation
python3 scripts/act/train_act_dual_cam.py
```

Checkpoint output:

```text
~/nrs_imitation/checkpoints/flow/polishing/dual_cam/<YYYYMMDD_HHMM>/
~/nrs_imitation/checkpoints/act/polishing/dual_cam/<YYYYMMDD_HHMM>/
```

### 7. Inference

This auto-selects the latest dual-cam `policy_best.ckpt` and opens the Grad-CAM viewers:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
```

Default inference uses Flow checkpoints under `checkpoints/flow/polishing/dual_cam`. For ACT checkpoints, add:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py policy_class:=ACT
```

Checkpoints trained with the default stain-mask pooling require inference to receive a live `stain_mask` for `cam0`:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py \
  use_stain_mask:=true \
  stain_mask_topic:=<live_cam0_stain_mask_topic>
```

If you trained with `--no_stain_mask`, keep inference RGB-only with `use_stain_mask:=false`.

Visualization topics:

```text
/inference_dual_cam/gradcam_overlay         # local Grad-CAM + xyz trajectory overlay
/inference_dual_cam/gradcam_overlay_global  # global Grad-CAM only
```

Disable either visualization layer when needed:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py gradcam_enable:=false
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py trajectory_overlay_enable:=false
```

Tune xyz overlay scale if the curve is too small or too large:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py trajectory_overlay_pixels_per_mm:=4.0
```

## Data Format

The compact `imitation_form` HDF5 contains only the fields used by training.

Single cam:

```text
observations/position
observations/force
observations/images/cam0
action/position
action/force
```

Dual cam:

```text
observations/position
observations/force
observations/images/cam0
observations/images/cam1
action/position
action/force
```

Removed from the normal polishing imitation form:

```text
marker
qpos
action_flat
meta
is_pad
```

## Manual Paths

Most commands above auto-select the latest file. Use explicit paths when needed.

Single-cam conversion:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_single_cam.py \
  --input_h5 datasets/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

Dual-cam conversion:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_dual_cam.py \
  --input_h5 datasets/multi_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

Single-cam training with explicit dataset:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form

python3 scripts/act/train_act_single_cam.py \
  --dataset_dir datasets/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

Both commands use `observations/images/stain_mask` by default. Add `--no_stain_mask` to either command to train the RGB-only baseline.

Dual-cam training with explicit dataset:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_dual_cam.py \
  --dataset_dir datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form

python3 scripts/act/train_act_dual_cam.py \
  --dataset_dir datasets/multi_cam/<YYYYMMDD_HHMM>/imitation_form
```

Dual-cam stain pooling is cam0-only: policy observation order is `cam0`, `stain_mask`, `cam1`, and `cam1` is not masked.

Single-cam inference with explicit checkpoint:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py \
  ckpt_dir:=/home/nrs_display/nrs_imitation/checkpoints/flow/polishing/single_cam/<YYYYMMDD_HHMM>
```

Use `policy_class:=ACT` and an ACT checkpoint path for ACT inference.

Dual-cam inference with explicit checkpoint:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py \
  ckpt_dir:=/home/nrs_display/nrs_imitation/checkpoints/flow/polishing/dual_cam/<YYYYMMDD_HHMM>
```

Use `policy_class:=ACT` and an ACT checkpoint path for ACT inference.

## Stage-1 VR Workflow Details

Stage1 is the first pass of the `dual_cam` data collection pipeline. It records VR teaching trajectories into `datasets/stage1`, and `vr_stage1_episode_pusher` sends selected episodes to the robot playback PC. The dual-cam HDF5 recorder then records the robot playback with `cam0` and `cam1`.

Default output:

```text
~/nrs_imitation/datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes/episode_*.hdf5
```

Record stage-1 VR episodes:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_stage1_hdf5_recorder
```

Push the latest stage-1 episode directory:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_stage1_episode_pusher
```

Push a specific stage-1 episode directory:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_stage1_episode_pusher --ros-args \
  -p episode_dir:=/home/nrs_display/nrs_imitation/datasets/stage1/<YYYYMMDD_HHMM>/stage1_vr_episodes
```

## Installed ROS2 Commands and Launch Files

After building and sourcing `behavior_ws/install/setup.bash`, the main commands are:

```bash
ros2 run nrs_imitation hdf5_recorder_single_cam
ros2 run nrs_imitation hdf5_recorder_dual_cam
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py
ros2 run nrs_imitation vr_stage1_hdf5_recorder
ros2 run nrs_imitation vr_stage1_episode_pusher
```
