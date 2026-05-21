# behavior_ws

ROS 2 workspace for **VR-based demonstration capture**, **Vive tracker communication**, **force/torque sensing**, **frame calibration**, and **ACT inference / data collection** used together with the `nrs_act` repository.

This workspace provides the real-world behavior-side pipeline around the ACT training code:
- read Vive tracker pose
- read gravity-compensated force/torque sensor data
- convert tracker pose from VR main station frame to robot base frame
- record demonstrations in `.txt` or `.hdf5`
- push recorded episodes back to the robot for playback
- record robot playback data into ACT-style `.hdf5`
- run learned model inference online

---

## Example execution commands

### 0) Calibration node
```bash
ros2 run vr_calibration vr_calibration
```

### 1) Vive tracker bridge node
```bash
ros2 launch vive_tracker_ros2 vive_bringup.launch.py
```

### 2) Force/torque sensor bridge node
```bash
ros2 launch nrs_ft_aq2 nrsvr_ft_aq.launch.py
```

### 3) Demonstration recording node for training data
```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder
```

### 4) Node that sends one recorded episode from the `.hdf5` file to the robot
```bash
ros2 run nrs_imitation vr_demo_hdf5_episode_pusher
```

### 5) Node that records robot position, force, and camera images during playback of the pushed episode
```bash
ros2 run nrs_imitation robot_playback_act_hdf5_recorder
```

### 6) ACT training with the dataset generated through steps 3), 4), and 5)
```bash
cd ~/nrs_imitation && python3 scripts/act/train_act.py \
  --ckpt_dir ~/nrs_imitation/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 500 \
  --lr 1e-4 \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --debug_norm
  --use_force_history
  --force_history_len 10
```

### 7) Evaluation after training
```bash
cd ~/nrs_imitation && python3 scripts/act/train_act.py \
  --ckpt_dir ~/nrs_imitation/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 500 \
  --lr 1e-4 \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --debug_norm
  --use_force_history
  --force_history_len 10
  --eval
```

### 8) Load the trained checkpoint and run online inference
```bash
ros2 run nrs_imitation node_act_cmdmotion_infer --ros-args \
  -p act_root:=~/nrs_imitation \
  -p ckpt_dir:=~/nrs_imitation/checkpoints/ur10e_swing/20260317_0043 \
  -p use_force_history:=true \
  -p force_history_len:=10
```

---

## Workspace role inside `nrs_act`

`behavior_ws` is the **real-world behavior pipeline workspace** stored inside the `nrs_act` repository.

While `nrs_act/source`, `scripts`, `datasets`, and `checkpoints` mainly cover **training / evaluation / model code / saved data**, `behavior_ws` covers the **robot-side ROS 2 execution pipeline** needed to:
- acquire human demonstration data
- transform sensor data into the robot frame
- build training-ready episodes
- replay demonstrations on the robot
- record robot execution data for ACT
- deploy trained ACT checkpoints online

In other words:
- `nrs_act` root = training / evaluation / model research code
- `behavior_ws` = communication, calibration, recording, playback, and inference-side ROS 2 packages

---

## Package overview

```text
behavior_ws/src/
├── nrs_ft_aq2
├── nrs_imitation
├── vive_tracker_interfaces
├── vive_tracker_ros2
└── vr_calibration
```

### 1. `vive_tracker_ros2`
ROS 2 package for **receiving Vive tracker data** and publishing it into the ROS 2 ecosystem.

Main role:
- communicate with the Vive tracking system
- publish tracker pose data into ROS 2
- provide calibration-matrix-related resources used together with downstream conversion

What it contains:
- tracker communication node(s)
- launch files for bring-up
- calibration-matrix-related data/config used in the tracker pipeline

Why it matters:
- this is the starting point of the VR teaching pipeline
- downstream packages depend on its tracker pose output

---

### 2. `vive_tracker_interfaces`
Interface package used by `vive_tracker_ros2` and other dependent ROS 2 packages.

Main role:
- define shared ROS 2 interfaces needed by the Vive tracker pipeline
- keep message/service definitions separated from implementation packages

Why it matters:
- prevents interface definitions from being mixed into runtime packages
- allows clean dependency structure for tracker-related nodes

---

### 3. `vr_calibration`
Package for converting Vive tracker measurements from the **VR main station frame** into the **UR10e robot base frame**.

Input conceptually:
- Vive tracker pose measured in the VR station frame
- position: `x, y, z`
- orientation: quaternion

Output conceptually:
- robot-base-frame pose expressed as:
  - `x, y, z, wx, wy, wz`

Main role:
- apply calibration transform between the VR frame and robot frame
- produce robot-usable pose data for teaching / playback / imitation data generation

Why it matters:
- raw VR tracker data is not directly usable by the robot
- this package aligns human teaching motion to the robot base coordinate system

---

### 4. `nrs_ft_aq2`
ROS 2 package for reading the **force/torque sensor mounted on the VR tracker** through **CAN communication**.

Main role:
- communicate with the FT sensor
- read force/torque values from hardware
- publish sensor values for the behavior pipeline
- provide **gravity-compensated** force/torque output

Why it matters:
- force is part of both demonstration understanding and ACT data collection
- gravity compensation makes the measured signal more useful for learning and control

---

### 5. `nrs_imitation`
ROS 2 package that ties the above packages into the **demonstration / dataset / inference workflow**.

Main roles:
- record demonstrations
- save data as `.txt` or `.hdf5`
- push recorded demonstration episodes to the robot
- record robot playback data into ACT-style training episodes
- run trained ACT checkpoint inference online

Typical responsibilities inside this package:
- demonstration recording node(s)
- episode playback / pusher node(s)
- ACT dataset recording node(s)
- trained policy inference node(s)

Why it matters:
- this is the bridge between raw sensor streams and ACT training/deployment
- it is the package that turns tracker/force/camera/robot data into usable imitation-learning episodes

---

## End-to-end workflow

### A. Sensor and frame pipeline
1. `vive_tracker_ros2` reads Vive tracker pose.
2. `vr_calibration` converts that pose from VR frame to robot base frame.
3. `nrs_ft_aq2` reads gravity-compensated force/torque sensor data.
4. `nrs_imitation` consumes these streams for recording or inference.

### B. Demonstration-to-training pipeline
1. Run the calibration node.
2. Run the Vive tracker node.
3. Run the force/torque sensor node.
4. Record a human-guided demonstration using `vr_demo_hdf5_recorder`.
5. Push one recorded episode to the robot with `vr_demo_hdf5_episode_pusher`.
6. During robot playback, record robot position, force, and images with `robot_playback_act_hdf5_recorder`.
7. Use the generated dataset to train ACT in the root `nrs_act` repository.
8. Evaluate the trained model.
9. Deploy the checkpoint with `node_act_cmdmotion_infer`.

---

## Relationship to ACT dataset generation

This workspace supports two different but connected data layers.

### Human-side demonstration data
Produced from:
- Vive tracker pose
- calibrated robot-frame pose
- force/torque sensor data

Saved by:
- demonstration recorder nodes in `nrs_imitation`

Possible formats:
- `.txt`
- `.hdf5`

### Robot-side ACT training data
Produced during playback of recorded demonstrations on the robot.

Saved by:
- `robot_playback_act_hdf5_recorder`

Typical contents:
- robot position / pose-related signals
- force signals
- camera images
- episode-structured ACT training data

---

## Practical separation of responsibilities

### Packages for communication / sensing
- `vive_tracker_ros2`
- `vive_tracker_interfaces`
- `nrs_ft_aq2`

### Package for frame conversion
- `vr_calibration`

### Package for behavior recording / playback / inference
- `nrs_imitation`

This separation keeps the workspace organized as:
- hardware communication
- geometric calibration
- data recording and learning deployment

---

## Recommended usage order

For real experiments, the most common order is:

1. start `vr_calibration`
2. start `vive_tracker_ros2`
3. start `nrs_ft_aq2`
4. start the relevant `nrs_imitation` recorder / pusher / inference node

For ACT training/deployment, the common sequence is:

1. capture demonstrations
2. create playback-based ACT dataset
3. train with `scripts/act/train_act.py`
4. evaluate checkpoint
5. deploy with `node_act_cmdmotion_infer`

---

## Summary

`behavior_ws` is the ROS 2 behavior-side workspace embedded inside `nrs_act`.

It provides the full runtime pipeline needed for:
- Vive tracker communication
- tracker interface definitions
- VR-to-robot frame calibration
- gravity-compensated force/torque sensing
- demonstration recording
- episode playback
- ACT dataset generation
- trained ACT policy inference

Together with the root `nrs_act` training code, it forms one end-to-end system from **human VR teaching** to **robot ACT learning and deployment**.
