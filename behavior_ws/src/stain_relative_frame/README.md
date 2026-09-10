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
