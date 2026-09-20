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
