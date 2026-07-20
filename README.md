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

- `single_cam` polishing data is saved under `datasets/polishing/single_cam`
- `dual_cam` polishing recorder data is saved under `datasets/polishing/dual_cam`
- gripper data is saved under `datasets/gripper/<single_cam|dual_cam>`

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

## Quick Start: UMI Gripper + F710 Joystick

Use this when controlling the Dynamixel-based UMI gripper with a Logitech F710.

The default `umi_ros2` gripper config uses the stable FTDI by-id port and slower
serial command timing:

```text
dxl.port: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7D6-if00-port0
dxl.cmd_rate_hz: 10.0
dxl.pos_slew_per_sec: 1000.0
```

If the USB adapter was reconnected, confirm that the by-id path exists:

```bash
ls -l /dev/serial/by-id/* /dev/ttyUSB*
```

Terminal 1, start the UMI gripper node:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch umi_ros2 umi_grp.launch.py
```

Terminal 2, start the F710 joystick controller:

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch dynamixel_joy_controller f710_gripper_joy.launch.py
```

Default F710 mapping:

```text
LB           -> open, 590 tick
RB           -> close, 2500 tick
A            -> home, 1545 tick
D-pad left/right -> step by 50 tick
```

To verify joystick commands without looking at the gripper log:

```bash
ros2 topic echo /gripper/command
```

The joystick controller publishes `std_msgs/Int32` targets on
`/gripper/command`, and `umi_gripper` consumes that topic.

## Quick Start: Gripper Single-Cam Inference

This uses the latest Flow checkpoint under `checkpoints/flow/gripper/single_cam`
and runs the same robot motion inference/control path as polishing inference.
It publishes action `[0:9]` to `/ur10skku/cmdMotion` through the polishing safety
loop, and additionally publishes action `[9]` as the learned gripper target tick
to `/gripper/command`. Grad-CAM heatmap visualization is on by default, and
stain-mask inference is not used.

```bash
cd ~/nrs_imitation/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py
```

Default topics:

```text
position         = /ur10skku/currentP
force            = /ur10skku/currentF
cam0             = /realsense/vr/color/image_raw
gripper position = /gripper/present_position
gripper current  = /gripper/present_current_mA
robot command    = /ur10skku/cmdMotion
gripper command  = /gripper/command
heatmap overlay  = /inference_gripper_single_cam/gradcam_overlay
```

If you are running with tracker pose/force topics:

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py \
  pose_topic:=/calibrated_pose \
  force_topic:=/ftsensor/measured_Cvalue \
  force_msg_type:=wrench
```

Explicit checkpoint:

```bash
ros2 launch nrs_imitation inference_gradcam_gripper_single_cam.launch.py \
  ckpt_dir:=/home/eunseop/nrs_imitation/checkpoints/flow/gripper/single_cam/<YYYYMMDD_HHMM>
```

Safety defaults:

```text
tau_sec:=0.8
startup_ramp_sec:=3.0
step_cap_pos_mm:=0.05
step_cap_ang_rad:=0.0001
step_cap_fz:=0.05
gripper_command_step_cap_tick:=200.0
gripper_command_slew_per_sec:=1000.0
gripper_cmd_safety_max_tick_from_present:=700.0
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
cam0     = /realsense/vr/color/image_raw
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

Joystick gripper controls:

```text
A           -> fully close (-653 tick by default)
B           -> fully open (733 tick by default)
D-pad left  -> close by 50 tick
D-pad right -> open by 50 tick
```

The fine-adjustment size can be changed at launch, for example:

```bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py gripper_step_tick:=20
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
~/nrs_imitation/datasets/polishing/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
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
~/nrs_imitation/datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
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
  auto_stain_mask:=true
```

If you trained with `--no_stain_mask`, keep inference RGB-only with `use_stain_mask:=false`.

Visualization topics:

```text
/inference_single_cam/gradcam_overlay      # local Grad-CAM overlay
/inference_single_cam/stain_mask_overlay   # generated stain-mask overlay when auto_stain_mask=true
```

Disable Grad-CAM visualization when needed:

```bash
ros2 launch nrs_imitation inference_gradcam_single_cam.launch.py gradcam_enable:=false
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
~/nrs_imitation/datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/merged_hdf5/*.hdf5
```

### 5. Convert to imitation_form

This auto-selects the latest dual-cam merged HDF5:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_dual_cam.py --overwrite --write_summary
```

Output:

```text
~/nrs_imitation/datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form/episode_*.hdf5
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
  auto_stain_mask:=true
```

If you trained with `--no_stain_mask`, keep inference RGB-only with `use_stain_mask:=false`.

Visualization topics:

```text
/inference_dual_cam/gradcam_overlay         # local Grad-CAM overlay
/inference_dual_cam/gradcam_overlay_global  # global Grad-CAM only
/inference_dual_cam/stain_mask_overlay      # generated stain-mask overlay when auto_stain_mask=true
```

Disable Grad-CAM visualization when needed:

```bash
ros2 launch nrs_imitation inference_gradcam_dual_cam.launch.py gradcam_enable:=false
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
  --input_h5 datasets/polishing/single_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_single_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

Dual-cam conversion:

```bash
cd ~/nrs_imitation
python3 source/custom/demo_data_imitation_form_dual_cam.py \
  --input_h5 datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/merged_hdf5/hdf5_recorder_dual_cam_<YYYYMMDD_HHMM>.hdf5 \
  --output_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form \
  --overwrite \
  --write_summary
```

Single-cam training with explicit dataset:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form

python3 scripts/act/train_act_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/<YYYYMMDD_HHMM>/imitation_form
```

Both commands use `observations/images/stain_mask` by default. Add `--no_stain_mask` to either command to train the RGB-only baseline.

Dual-cam training with explicit dataset:

```bash
cd ~/nrs_imitation
python3 scripts/flow/train_flow_dual_cam.py \
  --dataset_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form

python3 scripts/act/train_act_dual_cam.py \
  --dataset_dir datasets/polishing/dual_cam/<YYYYMMDD_HHMM>/imitation_form
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
