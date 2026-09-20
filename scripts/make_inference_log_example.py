#!/usr/bin/env python3
"""Write a uniquely named, explicitly SYNTHETIC example. No ROS initialization."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "behavior_ws/src/nrs_imitation"))
from nrs_imitation.execution_metrics import ExecutionRecorder, stamp, pose_fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=ROOT / "experiments/e1_inference_logging_20260917/examples")
    p.add_argument("--force-observation", choices=["ON", "OFF"], default="OFF")
    args = p.parse_args()
    meta = dict(synthetic_only=True, hardware_verified=False, task="schema_example_not_robot_run",
        force_observation=args.force_observation, tool_motion_type="rotary", rpm_setpoint=None,
        rpm_setpoint_source=None, measured_rpm=None, rpm_measurement_available=False,
        rpm_assumed_constant=False, surface_normal=None, compression_sign=None,
        contact_point_offset=None, effective_contact_area_mm2=None, preston_coefficient=None,
        clocks=dict(use_sim_time=False, source="synthetic timestamps", receipt="synthetic example grid"))
    rec = ExecutionRecorder(args.output_root, "SYNTHETIC_"+args.force_observation, meta)
    assert rec.ready.wait(timeout=3), "writer startup timed out"
    base = time.monotonic_ns()
    def at(i, source):
        row = stamp(1_000_000_000+i*5_000_000, source=source, frame_id="synthetic_base")
        row.update(receipt_monotonic_ns=base+i*5_000_000,
                   source_stamp_ns=1_000_000_000+i*5_000_000,
                   stamp_basis="synthetic_grid", source_clock="synthetic_ros")
        return row
    rec.event("run_start", at(0, "synthetic_node"), physical_task_started=False)
    rec.emit("roi", dict(at(0, "synthetic_reference"), center_xy_mm=[400,500], direction_rad=None,
                         reference_transform="synthetic translation only; not calibration"))
    rec.event("inference_start", at(0, "synthetic_node"), inference_id=1,
              force_observation_method="measured" if args.force_observation == "ON" else "constant_zero")
    for i in range(10):
        rec.emit("wrench", dict(at(i, "synthetic_force"), representation="synthetic_only",
            fx=1., fy=-2., fz=5.+i, tx=.1, ty=.2, tz=-.3, force_unit="N", torque_unit="N*m",
            semantic_frame="synthetic_base", raw_values=[1.,-2.,5.+i,.1,.2,-.3],
            transform_validity="unknown", correction_status="synthetic_no_correction"))
        if i % 2 == 0:
            pose = pose_fields([100.+i,200.,300.,0.,0.,0.], verified=True)
            pose.update(semantic_frame="synthetic_base", pose_kind="synthetic_TCP")
            rec.emit("tcp_pose", dict(at(i, "synthetic_feedback"),
                **pose))
    rec.emit("commands", dict(at(1, "synthetic_node"), command_stage="node_sent",
        command_mode="cmdMotion", inference_id=1, plan_id=1, command_id=1, action_index=0,
        x=999., y=200., z=300., rx=0., ry=0., rz=0., fx=0., fy=0., fz=12.,
        execution_status="SYNTHETIC_NOT_SENT_TO_HARDWARE", semantic_frame="synthetic_base",
        orientation_unit="rotvec_rad", position_unit="mm", force_unit="N"))
    rec.event("run_end", at(10, "synthetic_node"), physical_task_completion="not_a_robot_run")
    assert rec.close() and not rec.write_errors, rec.write_errors
    print(rec.path)


if __name__ == "__main__":
    main()
