# E2 C vibration investigation — 2026-09-20

Read-only diagnosis of recorded trials; no robot commands or hardware tests were issued. Controller code/configuration was not changed during this investigation.

## Runs

| Trial | Executor run ID | Recorded termination |
|---|---|---|
| C first | E2_line_C_r01_executor_20260920T201310_1789902790957592437_gwzlij3x | manual_abort; controller_hold_verified=true |
| C second | E2_line_C_r01_executor_20260920T201420_1789902860394556982_3l8f6539 | safety_stop at execution +18.320 s: controller Force mode lost or stale; STOP acknowledgement and physical hold not verified |

Provider IDs: E2_line_C_r01_20260920T201313_1789902793252210481_tp7jw3r4 and E2_line_C_r01_20260920T201422_1789902862705332180_75h5dukq. Sources are under logs/inference_metrics/. E1 reference: C_force_obs_ON_E1_20260916_20260920T171726_1789892246083511592_ua8reqys.

## Verified implementation differences

Checkpoint, normalizer and resolved policy settings match E1. E1 used service_stream (ptp9d_use_stream=true); E2 uses the independent timed_topic executor (ptp9d_use_stream=false).

E1 _ptp9d_stream_topup applies a 35-point centered moving average to pose columns, selects stride-2 points from an anchor near the queued tail, and sends them to the C++ stream. That stream chooses segment duration from distance/velocity, and receives force separately via PTP9D_STREAM_SET_FORCE using a policy sample nearest measured position.

E2 TimedExecution samples the raw postprocessed policy sequence at elapsed time. It retains a gain/rate limiter but bypasses the E1 35-point pose smoothing and distance-based segment timing. New plans replace the active plan at their generation timestamp. Thus identical policy weights do not imply equivalent robot execution.

For the second C's first 128-point postprocessed plan, adjacent requested XYZ displacement had median 5.288 mm, p95 10.473 mm and maximum 15.556 mm. Applying the existing E1 35-point moving average offline to exactly those pose samples reduced these to 0.576, 1.155 and 1.276 mm. This is a mathematical comparison of requested points, not a hardware validation of a fix.

## Measured evidence

| Quantity | C first | C second |
|---|---:|---:|
| Logged Fz minimum / maximum after execution start (N) | -10.337 / 85.840 | -10.790 / 86.944 |
| Executor contact gate transitions in first second | 12 | 14 |
| Executor contact gate transitions in first four seconds | 31 | 33 |
| Minimum node-sent Fz in first second (N) | -4.555 | -4.554 |
| Requested XYZ discontinuity when plan 3 replaced plan 2 (mm) | 62.729 | 56.699 |
| Maximum interval between node-sent command log rows (ms) | 8.713 | 10.563 |

At the second C's plan 2→3 boundary, the requested XYZ differed from interpolated measured XYZ by 42.825 mm just before replacement, and by 16.354 mm after replacement. The discontinuity is in the requested trajectory; rate limiting prevents an instantaneous robot displacement of this size.

The early gate transitions occur while approaching well above the working plane. This demonstrates repeated switching of the force command gate, not proof of repeated physical contact. Signed negative policy force was passed when the gate was on. E1 also contains brief negative force requests and a larger approach Fz peak (101.626 N), so negative requests or the E2 86–87 N peak alone do not establish a newly introduced or sole cause.

Executor CSV command intervals were close to 8 ms through tracking, with no recorded control timer deadline-missed fault. Summary files report drained loggers, zero write errors and zero queue drops. This does not establish remote delivery timing, exclude remote controller overload, or prove why the second trial lost Force mode/feedback.

## Assessment and limits

The removed E1 smoothing, altered time consumption and plan replacement discontinuities are concrete execution regressions relative to the previously used C setup, and strong candidate contributors to vibration. Early force gate chatter can additionally change lower controller behavior. The available data cannot isolate the unique mechanical cause or distinguish a user/robot stop, driver failure and communication failure behind the second trial's mode loss.

The second trial recorded stop_not_verified approximately 15 seconds after safety_stop: queue_cancel_ack=false and feedback_age_s=13.734. It ended with shutdown_without_verified_stop. Do not interpret process exit as verified physical hold.

These runs should not be described as validated successful C trials. Resolve the execution differences and controller-stop status before another hardware run. Any proposed change needs offline continuity/rate/gating checks; offline tests cannot certify absence of physical vibration. Preserve T's original time/pose/force semantics when revising shared execution.

Sources: inference_core.py (_ptp9d_stream_topup, _ptp9d_stream_update_force, _on_control_timer); e2_timed_execution.py (TimedExecution.accept/tick); inspected local dev_ws robot_command.cpp (streamLoop); each run's metadata.json, commands.csv, tcp_pose.csv, wrench.csv, events.jsonl and summary.json. The remote controller binary/configuration was not independently read back.
