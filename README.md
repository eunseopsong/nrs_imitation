# nrs_act

Refactored ACT-based imitation learning codebase for robotic polishing / manipulation experiments.  
This repository is organized around a modular `source/` layout, includes a ROS 2 behavior workspace `behavior_ws/`, and supports a **single-camera ACT recording pipeline** with optional **force history** input.

---

# Quick Start: Current Recording → ACT Dataset → Training Pipeline

This section is the most important part for the current workflow.

The current recommended pipeline is now:

```text
1. Record human demonstration directly with:
   vr_demo_hdf5_recorder.py
   - position
   - force
   - camera image
   - joystick-based start/end/erase/terminate

2. Convert merged HDF5 into ACT episode files with:
   demo_data_act_form_single_cam.py

3. Train ACT with:
   scripts/act/train_act.py
```

The previous 4-step pipeline was:

```text
1. vr_demo_hdf5_recorder.py
2. vr_demo_hdf5_episode_pusher.py
3. robot_playback_act_hdf5_recorder.py
4. demo_data_act_form.py
```

The current pipeline skips robot playback during dataset generation:

```text
1. vr_demo_hdf5_recorder.py
2. demo_data_act_form_single_cam.py
3. train_act.py
```

---

## A. Run the joystick controller

The joystick controller converts Logitech F710 / Xbox-style joystick input into discrete recorder commands.

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

This launch file runs both:

```text
joy_node
vr_demo_joy_controller
```

The command topic published by the joystick controller is:

```text
/vr_demo_recorder/command
```

Default joystick mapping:

```text
A             -> start_recording
B             -> end_recording
X             -> erase_current_episode
Y             -> terminate_node
D-pad left    -> prev_episode
D-pad right   -> next_episode
```

If D-pad left/right is reversed:

```bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py dpad_left_positive:=false
```

To verify joystick commands:

```bash
ros2 topic echo /vr_demo_recorder/command
```

---

## B. Run the recording node

The recorder directly records:

```text
position + force + camera
```

Input topics:

```text
/calibrated_pose                 Float64MultiArray [x, y, z, wx, wy, wz]
/ftsensor/measured_Cvalue         geometry_msgs/Wrench
/realsense/vr/color/image_raw     sensor_msgs/Image
/vr_demo_recorder/command         std_msgs/String
```

Run:

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation vr_demo_hdf5_recorder
```

The recorder saves a merged HDF5 file under:

```text
/home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/
└── vr_demo_merged_YYYYMMDD_HHMM.hdf5
```

Merged HDF5 structure:

```text
episodes/
├── ep_0000/
│   ├── position              # (T, 6)
│   ├── ft                    # (T, 3)
│   └── images/
│       └── cam0              # (T, H, W, 3)
├── ep_0001/
│   ├── position
│   ├── ft
│   └── images/
│       └── cam0
...
```

Dataset meanings:

```text
position = [x, y, z, wx, wy, wz]
ft       = [fx, fy, fz]
cam0     = RGB image from /realsense/vr/color/image_raw
```

Notes:
- Episode start/end is controlled only by joystick commands.
- The previous force-threshold start/end trigger was removed from this workflow.
- The recorder prints discrete recording state changes such as `IDLE`, `RECORDING`, `SAVED`, `ERASED`, and `SHUTDOWN`.

---

## C. Convert merged HDF5 into ACT episode files

The converter is a normal Python script, not a ROS 2 node.

Run:

```bash
cd ~/nrs_act/source/custom

python3 demo_data_act_form_single_cam.py
```

By default, it automatically finds the latest merged HDF5 file under:

```text
/home/eunseop/nrs_act/datasets/ACT/*/merged_hdf5/*.hdf5
```

It creates:

```text
/home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
...
└── manifest.json
```

Final ACT episode structure:

```text
episode_0.hdf5
├── action/
│   ├── position              # (T_pad, 6)
│   └── force                 # (T_pad, 3)
├── observations/
│   ├── position              # (T_pad, 6)
│   ├── force                 # (T_pad, 3)
│   ├── images/
│   │   └── cam0              # (T_pad, H, W, 3)
│   └── is_pad                # (T_pad,)
└── meta/
    ├── orig_len
    ├── T_pad
    ├── pad_starts_at
    ├── truncated
    └── camera_name
```

After successful conversion, the merged HDF5 file is deleted by default to save disk space.

To keep the merged HDF5 file for debugging:

```bash
python3 demo_data_act_form_single_cam.py --keep-merged
```

To convert a specific merged file:

```bash
python3 demo_data_act_form_single_cam.py \
  -i /home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/vr_demo_merged_YYYYMMDD_HHMM.hdf5
```

---

## D. Train ACT

The current `scripts/act/train_act.py` is configured so that running it without arguments starts training with the default current settings.

Run:

```bash
cd ~/nrs_act/scripts/act

python3 train_act.py
```

Default training settings:

```text
ckpt_dir        = /home/eunseop/nrs_act/checkpoints/ur10e_swing
policy_class    = ACT
task_name       = ur10e_swing
batch_size      = 6
seed            = 0
num_epochs      = 500
lr              = 1e-4
kl_weight       = 10
chunk_size      = 200
train_seq_len   = 200
val_seq_len     = 200
hidden_dim      = 512
dim_feedforward = 3200
camera_names    = ["cam0"]
```

If `--dataset_dir` is not provided, `train_act.py` automatically finds the latest `episodes_ft` directory under:

```text
/home/eunseop/nrs_act/datasets/ACT/*/episodes_ft
```

Expected startup log:

```text
[INFO] dataset_dir       = /home/eunseop/nrs_act/datasets/ACT/<LATEST_RUN_ID>/episodes_ft
[INFO] num_episodes      = <auto-counted episode count>
[INFO] camera_names      = ['cam0']
[INFO] chunk_size        = 200
[INFO] train_seq_len     = 200
[INFO] val_seq_len       = 200
```

To train on a specific dataset:

```bash
cd ~/nrs_act/scripts/act

python3 train_act.py \
  --dataset_dir /home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft
```

To enable force history:

```bash
python3 train_act.py \
  --use_force_history \
  --force_history_len 10
```

To print normalization debug output:

```bash
python3 train_act.py --debug_norm
```

---

## E. Minimal full workflow

```bash
# Terminal 1: joystick command node
cd ~/nrs_act/behavior_ws
source install/setup.bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

```bash
# Terminal 2: recorder
cd ~/nrs_act/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_demo_hdf5_recorder
```

```bash
# After recording: convert merged HDF5 into ACT episode files
cd ~/nrs_act/source/custom
python3 demo_data_act_form_single_cam.py
```

```bash
# Train
cd ~/nrs_act/scripts/act
python3 train_act.py
```

---

# Repository Documentation

## 0. What each top-level folder does

Before looking into the ACT model internals, the easiest way to understand this repository is to separate it into **training-side folders** and **robot/behavior-side folders**.

```text
nrs_act/
├── behavior_ws/
├── checkpoints/
├── datasets/
├── LICENSE
├── README.md
├── scripts/
└── source/
```

### `behavior_ws/`
ROS 2 workspace for the real experiment pipeline around ACT.

Main role:
- Vive tracker communication
- tracker interface definitions
- VR-to-robot frame calibration
- gravity-compensated force/torque sensor communication
- RealSense camera recording
- joystick-based recording control
- demonstration recording as merged `.hdf5`
- ACT inference node execution

Contained packages:
- `nrs_ft_aq2`
- `nrs_imitation`
- `vive_tracker_interfaces`
- `vive_tracker_ros2`
- `vr_calibration`

In short, `behavior_ws` is the **data acquisition / deployment side** of the project.

---

### `datasets/`
Stores ACT training datasets.

Typical role:
- episode-based imitation-learning datasets
- merged HDF5 files from demonstration recording
- final `episode_*.hdf5` files used by the dataloader
- task-specific dataset directories

Current generated layout:

```text
datasets/ACT/
└── YYYYMMDD_HHMM/
    ├── merged_hdf5/
    │   └── vr_demo_merged_YYYYMMDD_HHMM.hdf5
    └── episodes_ft/
        ├── episode_0.hdf5
        ├── episode_1.hdf5
        └── manifest.json
```

This is the main location for recorded episodes that will be consumed by `source/data/`.

---

### `checkpoints/`
Stores training outputs and saved models.

Typical contents:
- timestamped training result directories
- `policy_best.ckpt`
- `policy_last.ckpt`
- `dataset_stats.pkl`
- plots or debug outputs generated during training/evaluation

This is the main location for experiment results and reusable trained weights.

---

### `scripts/`
Thin entrypoint scripts for training or evaluation.

Most important file:
- `scripts/act/train_act.py`

Main role:
- CLI parsing
- default training argument setup
- automatic latest dataset discovery
- experiment setup
- calling the actual loader / model / training code in `source/`

Design philosophy:
- keep scripts light
- move most algorithmic logic into `source/`

---

### `source/`
Core ACT codebase after refactoring.

Main role:
- reusable project modules
- dataset loading and normalization
- ACT model and encoder definitions
- training / validation loop
- debug and plotting utilities
- custom dataset conversion utilities

This is the main folder to patch when modifying model behavior, data handling, losses, or training flow.

---

## 1. Repository role split

At a high level, the repository is divided like this:

### Training / research side
- `datasets/`
- `checkpoints/`
- `scripts/`
- `source/`

### Real-world robot / behavior side
- `behavior_ws/`

So the current overall pipeline is:
1. use `behavior_ws` to capture demonstrations directly as position / force / camera merged HDF5
2. convert merged HDF5 into ACT episode files under `datasets/`
3. train and evaluate with `scripts/act/train_act.py`
4. save outputs under `checkpoints/`
5. patch reusable logic mainly in `source/`

---

## 2. Overview

`nrs_act` is an imitation learning project built on a customized ACT codebase and later refactored for maintainability and future research patches.

Current baseline characteristics:
- ACT-based behavior cloning / imitation learning
- Observation = **position/orientation + force + single-camera RGB**
- Action = **position/orientation + force**
- Camera name = `cam0`
- Modular structure: `common / data / models / training`
- Main entrypoint kept at `scripts/act/train_act.py`
- `train_act.py` can start training with default arguments and automatic latest dataset discovery
- Force-history-aware encoder support added without changing raw `.hdf5` demo files
- ROS 2 real-world behavior workspace included under `behavior_ws/`
- Joystick-based demonstration recording is supported through `joy_node` and `vr_demo_joy_controller`

This project is designed so that future patches can be added mainly under `source/` while keeping `scripts/act/train_act.py` as an orchestration entrypoint. The original README described the modular ACT structure and force-history-aware encoder design, which are preserved here.

---

## 3. Credits / Origin / Upstream

This repository is **not a from-scratch implementation**. It is a refactored research codebase derived from a customized ACT implementation and upstream ACT/DETR components.

### Original customized ACT codebase
- **Chemin Ahn**
- Homepage: `https://chemx3937.github.io/`
- GitHub: `https://github.com/Chemx3937`

### Upstream references
1. **ACT: Action Chunking with Transformers**
   - Tony Z. Zhao
   - Project page: `https://tonyzhaozh.github.io/aloha/`

2. **DETR**
   - Facebook Research
   - GitHub: `https://github.com/facebookresearch/detr`

### Attribution rule
When sharing, patching, or redistributing this repository:
- keep credit to **Chemin Ahn**
- keep attribution to **ACT**
- keep attribution to **DETR**
- do not remove license / attribution files

---

## 4. License

The root `LICENSE` keeps integrated upstream notices.

Included upstream licenses:
- **ACT**: MIT License  
  Copyright (c) 2023 Tony Z. Zhao
- **DETR**: Apache License 2.0  
  Copyright 2020-present, Facebook, Inc.

Notes:
- root `LICENSE` must be preserved
- `README.md` and attribution notes should continue to mention the original customized code origin
- if new external code is added later, its license notice must also be preserved

---

## 5. Current Project Structure

```text
nrs_act/
├── LICENSE
├── README.md
├── behavior_ws/
│   └── src/
│       ├── nrs_ft_aq2
│       ├── nrs_imitation
│       ├── vive_tracker_interfaces
│       ├── vive_tracker_ros2
│       └── vr_calibration
├── checkpoints/
├── datasets/
├── scripts/
│   └── act/
│       └── train_act.py
└── source/
    ├── common/
    │   ├── fs.py
    │   └── utils.py
    ├── custom/
    │   ├── check_cam_serial.py
    │   ├── custom_constants.py
    │   ├── custom_real_env.py
    │   ├── custom_robot_utils.py
    │   ├── demo_data_act_form.py
    │   └── demo_data_act_form_single_cam.py
    ├── data/
    │   ├── dataset.py
    │   ├── loader.py
    │   └── normalization.py
    ├── models/
    │   ├── __init__.py
    │   ├── act_core.py
    │   ├── backbone.py
    │   ├── encoder.py
    │   ├── policy.py
    │   └── transformer.py
    └── training/
        ├── debug.py
        ├── engine.py
        └── plotting.py
```

### Key idea of the refactor
Previous monolithic logic was split by responsibility:
- `data/` → dataset / normalization / dataloader
- `models/` → ACT core / policy / encoder / backbone / transformer
- `training/` → train loop / debug / plotting
- `common/` → general shared utilities
- `behavior_ws/` → ROS 2 runtime pipeline for demonstration capture, calibration, recording, and inference

---

## 6. What Each Folder Does

### `scripts/act/`
Training / evaluation entrypoint.

Main file:
- `train_act.py`

Responsibilities:
- parse CLI arguments
- provide default training arguments
- automatically find the latest `episodes_ft` dataset directory if `--dataset_dir` is not provided
- auto-count `episode_*.hdf5` files if `--num_episodes` is not provided
- set `camera_names = ["cam0"]` by default
- assemble policy config
- call `load_data(...)`
- call `train_bc(...)`
- handle evaluation mode
- save checkpoint directory and dataset stats

Design rule:
- keep this file as thin as possible
- future algorithmic patches should mostly go into `source/`

---

### `source/common/`
General utilities.

Files:
- `fs.py` → checkpoint folder lookup helpers such as latest timestamped subdir search
- `utils.py` → common helpers like seeding and dictionary utilities

---

### `source/data/`
Dataset and dataloader logic.

Files:
- `dataset.py`
- `loader.py`
- `normalization.py`

Responsibilities:
- read `episode_*.hdf5`
- read single-camera image stream `observations/images/cam0`
- sample episode start timesteps
- build current observation and action chunk
- normalize qpos/action with per-dimension min-max
- optionally build **force history** on-the-fly from raw episode force trajectory
- create train / val `DataLoader`

This folder is the main patch point for:
- contact labels
- phase labels
- onset weighting
- previous-action history
- force-history generation
- normalization changes

---

### `source/models/`
Model definition and wrappers.

Files:
- `encoder.py` → split observation encoders
- `act_core.py` → ACT / CNNMLP core model builders
- `backbone.py` → CNN image backbone + positional encoding
- `transformer.py` → transformer encoder/decoder
- `policy.py` → training-facing policy wrapper and losses
- `__init__.py` → package import convenience

This folder is the main patch point for:
- encoder changes
- auxiliary heads
- force/contact prediction heads
- loss weighting
- fusion changes
- model architecture extensions

---

### `source/training/`
Training loop and debugging.

Files:
- `engine.py` → training / validation loop
- `debug.py` → normalization debug and AMP helpers
- `plotting.py` → training history plotting

Responsibilities:
- batch forward pass
- support 4-item and 5-item batch formats
- validation / checkpoint save
- normalization debug print
- optional AMP handling

---

### `source/custom/`
Custom environment / task-specific helpers kept from the original research workflow.

Important current files:
- `demo_data_act_form_single_cam.py`

Main role:
- convert merged HDF5 files from the recording node into ACT `episode_*.hdf5` files
- write `observations/images/cam0`
- write action as next-step hold
- write padding metadata
- delete merged HDF5 after successful conversion unless `--keep-merged` is used

---

### `behavior_ws/`
ROS 2 workspace for behavior capture and deployment.

Responsibilities:
- bring up Vive tracker communication
- provide tracker-related interfaces
- calibrate VR-frame pose into UR10e base frame
- read gravity-compensated force/torque data
- read RealSense camera images
- run joystick command mapping
- record human demonstrations directly as ACT merged HDF5
- run online ACT inference nodes

Current important nodes in `nrs_imitation`:
- `vr_demo_hdf5_recorder`
- `vr_demo_joy_controller`
- `vr_demo_txt_recorder`
- `vr_demo_hdf5_episode_pusher`
- `robot_playback_act_hdf5_recorder`
- ACT inference nodes

---

## 7. Current Dataset Recording Pipeline

### Previous dataset pipeline
The older dataset generation flow used robot playback:

```text
1. vr_demo_hdf5_recorder.py
   -> save position/force trajectory only

2. vr_demo_hdf5_episode_pusher.py
   -> send one episode trajectory to robot

3. robot_playback_act_hdf5_recorder.py
   -> record robot playback with position + force + two cameras

4. demo_data_act_form.py
   -> split merged HDF5 into episode_*.hdf5
```

### Current dataset pipeline
The current pipeline records everything at the human demonstration stage:

```text
1. vr_demo_hdf5_recorder.py
   -> record position + force + cam0 directly

2. demo_data_act_form_single_cam.py
   -> split merged HDF5 into episode_*.hdf5

3. train_act.py
   -> train ACT using latest dataset by default
```

This reduces the data generation loop and avoids the intermediate robot playback recording stage.

---

## 8. Observation / Action Definition

### Current action definition
The model still predicts the same 9D action as before:

\[
a_t = [x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

So **the action space has not changed**.

### Current observation definition
The current observation is based on:
- pose/orientation: `x y z wx wy wz`
- force: `fx fy fz`
- single-camera RGB image: `cam0`

The observation state vector is:

\[
q_t = [x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

---

## 9. Old Encoder Structure vs New Encoder Structure

## Before
A single shared state encoder processed the current 9D state directly:

\[
q_t = [x,y,z,w_x,w_y,w_z,f_x,f_y,f_z]
\]

\[
e_t = \phi_{shared}(q_t)
\]

Characteristics:
- position/orientation and force were mixed immediately
- force used only the current timestep
- no temporal force context

---

## After
The observation is now encoded in a split manner.

### 1) Position encoder
Current pose/orientation is encoded separately:

\[
p_t = [x,y,z,w_x,w_y,w_z]
\]

\[
e_t^{pos} = \phi_{pos}(p_t)
\]

### 2) Force encoder (GRU)
A short force history window is encoded when `--use_force_history` is enabled:

\[
H_t = [f_{t-L+1}, \dots, f_t], \quad f_t = [f_x,f_y,f_z]
\]

\[
e_t^{force} = \phi_{force}(H_t)
\]

Here `phi_force` is a **GRU-based force-history encoder**.

### 3) Fusion encoder
Position and force embeddings are fused into one observation embedding:

\[
e_t = \phi_{fuse}([e_t^{pos}; e_t^{force}])
\]

### 4) Image encoder
RGB observations from `cam0` are encoded by the image backbone:

\[
e_t^{img} = \phi_{img}(I_t^{cam0})
\]

These are then used by ACT / CNNMLP policy logic.

---

## 10. Why the New Structure Matters

The new structure improves the state representation in two ways.

### A. Position / force disentangling
Before, pose and force were forced into the same encoder.  
Now they are represented separately first, which reduces early entanglement.

### B. Temporal force modeling
Before, the model only saw current force:

\[
f_t
\]

Now it can see force trend:

\[
f_{t-L+1}, \dots, f_t
\]

This is especially important for:
- non-contact → contact transition
- pressing phase detection
- force continuity
- contact-aware action generation

---

## 11. Force History: How It Is Added Without Changing Raw `.hdf5`

Raw demo files are **not rewritten**.

Instead, `source/data/dataset.py` builds force history on-the-fly from the episode’s full force trajectory.

If the current sampled timestep is `t`, dataset constructs:

\[
H_t = [f_{t-L+1}, \dots, f_t]
\]

using `/observations/force` inside the same episode file.

### Episode start padding
If `t < L-1`, the left side is padded by repeating the first available force value.

Example:

\[
[f_0, f_0, \dots, f_0, f_1, \dots, f_t]
\]

### Normalization of force history
`force_history` uses the same min-max statistics as the force part of `qpos`.

So the raw `.hdf5` stays the same, while the dataset becomes history-aware.

---

## 12. Files Changed for the Current Pipeline

### Newly added / important
- `source/models/encoder.py`
- `source/custom/demo_data_act_form_single_cam.py`
- `behavior_ws/src/nrs_imitation/nrs_imitation/vr_demo_joy_controller.py`
- `behavior_ws/src/nrs_imitation/launch/vr_demo_joy_controller.launch.py`

### Main modified files
- `scripts/act/train_act.py`
- `source/models/act_core.py`
- `source/models/policy.py`
- `source/data/dataset.py`
- `source/data/loader.py`
- `source/training/engine.py`
- `source/training/debug.py`
- `behavior_ws/src/nrs_imitation/nrs_imitation/vr_demo_hdf5_recorder.py`

### Roles of these changes
- `vr_demo_hdf5_recorder.py` → joystick-controlled position / force / cam0 merged HDF5 recorder
- `vr_demo_joy_controller.py` → converts F710 joystick input into recorder command strings
- `demo_data_act_form_single_cam.py` → converts merged HDF5 into single-camera ACT episode files
- `train_act.py` → default cam0 training, auto latest dataset discovery, default chunk-200 training config
- `encoder.py` → position encoder / force GRU encoder / image encoder definitions
- `dataset.py` → reads ACT episode files and builds optional `force_history`
- `loader.py` → enables force-history dataset mode
- `engine.py` → supports both 4-item and 5-item batches
- `debug.py` → prints `force_history` stats when enabled

---

## 13. Encoder Components in `source/models/encoder.py`

### `PositionStateEncoder`
Encodes:

\[
[x,y,z,w_x,w_y,w_z]
\]

into a learned embedding.

### `ForceHistoryGRUEncoder`
Encodes:

\[
[f_{t-L+1}, \dots, f_t]
\]

with a GRU and uses the last hidden state as force embedding.

### `PositionForceFusionEncoder`
Takes concatenated position and force embeddings and maps them to one fused embedding.

Currently implemented as shallow fusion:

\[
\text{Linear} + \text{Activation}
\]

### `ImageObservationEncoder`
Used for ACT image features.

### `CNNMLPImageEncoder`
Used for the CNNMLP baseline path.

---

## 14. Current Data Format Assumption

Each dataset directory should contain:

```text
episode_0.hdf5
episode_1.hdf5
...
manifest.json
```

Expected keys:
- `/observations/position`
- `/observations/force`
- `/observations/images/cam0`
- `/observations/is_pad`
- `/action/position`
- `/action/force`
- `/meta/orig_len`
- `/meta/T_pad`

Current dimensionality:
- observation qpos: `position(6) + force(3) = 9D`
- action: `position(6) + force(3) = 9D`
- image: `cam0`
- force history: `(L, 3)` when enabled

Camera names used by default:
- `cam0`

Expected image tensor shape during training:

```text
(B, 1, 3, H, W)
```

---

## 15. Normalization

### qpos / action
Per-dimension min-max normalization to `[0, 1]`.

For each dimension independently:
- x
- y
- z
- wx
- wy
- wz
- fx
- fy
- fz

### images
- uint8 → float `[0,1]`
- ImageNet normalization is still applied inside `source/models/policy.py`

### force history
When enabled, `force_history` is normalized using the same min/max used for the force portion of qpos.

---

## 16. Training Flow

Training entrypoint:

```bash
python3 scripts/act/train_act.py
```

Flow:
1. parse CLI args
2. automatically resolve latest dataset dir if `--dataset_dir` is not provided
3. auto-count `episode_*.hdf5` files if `--num_episodes` is not provided
4. build `policy_config`
5. call `load_data(...)`
6. create train / val loaders
7. optionally print normalization debug
8. build policy
9. train / validate / save checkpoints

---

## 17. Main Training Command

### Default training

```bash
cd /home/eunseop/nrs_act/scripts/act
python3 train_act.py
```

This is equivalent to the current default setup:

```bash
cd /home/eunseop/nrs_act && python3 scripts/act/train_act.py \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 500 \
  --lr 1e-4 \
  --kl_weight 10 \
  --chunk_size 200 \
  --train_seq_len 200 \
  --val_seq_len 200 \
  --hidden_dim 512 \
  --dim_feedforward 3200
```

### Default training with force history

```bash
cd /home/eunseop/nrs_act/scripts/act
python3 train_act.py \
  --use_force_history \
  --force_history_len 10
```

### Debug normalization

```bash
cd /home/eunseop/nrs_act/scripts/act
python3 train_act.py --debug_norm
```

### Specific dataset

```bash
cd /home/eunseop/nrs_act/scripts/act
python3 train_act.py \
  --dataset_dir /home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft
```

### Specific checkpoint root

```bash
cd /home/eunseop/nrs_act/scripts/act
python3 train_act.py \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/my_experiment
```

---

## 18. Important CLI Flags for the Current Structure

### Dataset flags
- `--dataset_dir`  
  manually sets the ACT episode directory. If omitted, the latest `episodes_ft` directory is selected automatically.

- `--num_episodes`  
  manually sets the number of episodes. If omitted, the number of `episode_*.hdf5` files is counted automatically.

- `--camera_names cam0`  
  camera names used by the dataloader and model. Default is `cam0`.

### Sequence length flags
- `--chunk_size`  
  number of ACT action queries.

- `--train_seq_len` / `--val_seq_len`  
  action sequence lengths used by the dataset. If omitted, they follow `chunk_size`.

### Force history flags
- `--use_force_history`  
  enables dataset-side force history generation and passes it to the model.

- `--force_history_len 10`  
  sets the GRU history window length.

### Split encoder hyperparameters
- `--position_dim`
- `--force_dim`
- `--position_encoder_hidden_dim`
- `--force_encoder_hidden_dim`
- `--force_encoder_num_layers`
- `--force_encoder_dropout`
- `--observation_encoder_activation`
- `--cnnmlp_observation_embed_dim`

---

## 19. Inference / Evaluation Notes

Evaluation command:

```bash
cd /home/eunseop/nrs_act/scripts/act
python3 train_act.py \
  --eval \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing
```

If `--eval` is used and `policy_best.ckpt` is not directly inside `--ckpt_dir`, the script searches the latest timestamped checkpoint subdirectory.

### Important
If the model was trained with force history, online inference should also maintain a recent force buffer:

\[
[f_{t-L+1}, \dots, f_t]
\]

Otherwise train-time and inference-time input structures do not match.

---

## 20. Checkpoints and Saved Files

Training creates a timestamped directory under `checkpoints/<task_name>/...` or the user-specified `--ckpt_dir`.

Typical contents:
- `policy_best.ckpt`
- `policy_last.ckpt`
- `dataset_stats.pkl`
- optional plot outputs

`dataset_stats.pkl` stores normalization statistics for later denormalization / deployment.

---

## 21. Debug Output

When `--debug_norm` is enabled, training prints normalized statistics before training begins.

Current debug output includes:
- image shape
- qpos shape
- action shape
- is_pad shape
- force_history shape if enabled
- qpos per-dimension mean/std
- action per-dimension mean/std
- image RGB mean/std
- force_history mean/std if enabled
- range checks for normalized values

For single-camera training, expected image shape is:

```text
(B, 1, 3, H, W)
```

This is useful to verify:
- normalization correctness
- force-history value scale
- dataset pipeline integrity
- train/val split sanity
- single-camera loading through `cam0`

---

## 22. What Did *Not* Change

Even after the current recorder / converter / training updates:
- final action dimension is still **9D**
- output action remains:

\[
[x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

So the current pipeline changes the **data acquisition and image stream structure**, not the final action definition.

---

## 23. Expected Effect of the New Structure

### Before
- robot playback was required to generate camera-based ACT data
- two-camera dataset assumption: `cam_top`, `cam_ee`
- current force only unless force history was enabled
- pose and force immediately entangled in the basic encoder

### After
- direct human demonstration recording with position + force + camera
- joystick-controlled episode recording
- single-camera dataset assumption: `cam0`
- automatic latest dataset selection in `train_act.py`
- separate position encoder
- optional GRU-based force-history encoder
- fused observation representation
- better opportunity to model:
  - force transition
  - contact onset
  - contact maintenance
  - force-aware action chunk prediction

Expected inference-side benefits:
- simpler dataset creation loop
- less dependency on intermediate playback recording
- cleaner camera input shape
- more context-aware force prediction when force history is enabled
- better non-contact → contact transition handling
- more consistent force-conditioned action chunks
- cleaner separation between geometry and force representation

---

## 24. Current Limitations

This patch improves the dataset-generation loop and representation, but does **not** automatically solve all force prediction issues.

Still likely future patch targets:
- force dimension loss weighting
- contact auxiliary loss
- phase label / phase loss
- onset weighting
- auxiliary heads from latent / hidden state
- inference-side force history buffer verification
- camera mounting / view consistency checks
- force edge-zero duration tuning during recording

Most likely future files to patch:
- `source/data/dataset.py`
- `source/models/encoder.py`
- `source/models/act_core.py`
- `source/models/policy.py`
- `source/training/engine.py`
- `behavior_ws/src/nrs_imitation/nrs_imitation/vr_demo_hdf5_recorder.py`

---

## 25. Recommended Patch Rules Going Forward

1. keep `scripts/act/train_act.py` thin
2. keep folder responsibilities separated
3. do not mix dataset logic into model files unnecessarily
4. preserve attribution and license files
5. preserve import stability with `__init__.py` where needed
6. prefer modifying `source/` rather than rewriting the training script
7. keep recording / conversion / training responsibilities separated:
   - recorder: merged HDF5
   - converter: ACT episode files
   - trainer: model training
8. keep camera names explicit and consistent:
   - current default: `cam0`

---

## 26. Summary

`nrs_act` is now in a stronger baseline state for force-aware, camera-conditioned imitation learning.

Current baseline status:
- refactored modular structure completed
- ACT training runs successfully
- position / force encoder separation added
- optional force-history GRU path added
- single-camera input path standardized as `cam0`
- joystick-controlled recording added
- direct VR demonstration recording with position + force + camera added
- merged HDF5 to ACT episode conversion added
- automatic latest dataset discovery added to `train_act.py`
- debug pipeline updated to show normalization and force-history statistics
- ROS 2 `behavior_ws` maintained for real-world behavior capture and deployment

In short, the project has evolved from:

\[
\text{shared current-state encoder}
\]

into:

\[
\text{position encoder} + \text{optional force-history GRU encoder} + \text{fusion encoder}
\]

and the data-generation pipeline has evolved from:

\[
\text{human demo} \rightarrow \text{robot playback} \rightarrow \text{ACT dataset}
\]

into:

\[
\text{human demo with position/force/cam0} \rightarrow \text{ACT dataset}
\]
