"""ROS-free, bounded, best-effort execution recorder. No control/RNG dependencies.

Blank CSV cells are unknown, never zero measurements. JSON payloads preserve
raw schemas. Only the writer thread opens/writes/flushes/encodes files.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time


COMMON = ["run_id", "logger_seq", "source_seq", "source_stamp_ns", "receipt_ros_ns",
          "receipt_monotonic_ns", "stamp_basis", "source_clock", "frame_id",
          "source", "validity"]
FIELDS = {
    "wrench": COMMON + ["representation", "fx", "fy", "fz", "tx", "ty", "tz",
        "force_unit", "torque_unit", "semantic_frame", "correction_status",
        "normal_force_signed", "surface_normal", "compression_sign",
        "transform_stamp_ns", "transform_validity", "raw_values", "details"],
    "tcp_pose": COMMON + ["x", "y", "z", "rx", "ry", "rz", "qx", "qy", "qz", "qw",
        "position_unit", "orientation_representation", "quaternion_order",
        "pose_kind", "semantic_frame", "velocity", "work_pose", "contact_offset",
        "raw_values", "details"],
    "commands": COMMON + ["inference_id", "plan_id", "command_id", "action_index",
        "command_stage", "command_mode", "semantic_frame", "orientation_unit",
        "position_unit", "force_unit", "x", "y", "z", "rx", "ry", "rz",
        "fx", "fy", "fz", "limited", "limiting_reason", "execution_status",
        "raw_values", "details"],
    "legacy": COMMON + ["t_wall", "t_elapsed_sec", "stage", "policy_class", "ckpt_dir",
        "meas_x_mm", "meas_y_mm", "meas_z_mm", "meas_rx", "meas_ry", "meas_rz",
        "meas_fx_N", "meas_fy_N", "meas_fz_N", "cmd_x_mm", "cmd_y_mm", "cmd_z_mm",
        "cmd_rx", "cmd_ry", "cmd_rz", "cmd_fx_N", "cmd_fy_N", "cmd_fz_N",
        "contact", "cmd_safety_blocked", "pose_receipt_monotonic_ns",
        "force_receipt_monotonic_ns", "pose_age_ns", "force_age_ns", "stale",
        "measurement_semantics"],
}


def clean(value):
    """Strict JSON: retain nonfinite evidence as strings, not invented null/zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    return str(value)


def cell(value):
    if value is None:
        return ""
    value = clean(value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value


def stamp(ros_ns, msg=None, source="", frame_id=None):
    header = getattr(msg, "header", None)
    source_stamp = None
    if header is not None:
        source_stamp = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
    return dict(source_seq=getattr(header, "seq", None), source_stamp_ns=source_stamp,
                receipt_ros_ns=int(ros_ns) if ros_ns is not None else None, receipt_monotonic_ns=time.monotonic_ns(),
                stamp_basis="source_header" if header is not None else "receipt_only_no_header",
                source_clock="publisher_ros_unverified_sync" if header is not None else None,
                frame_id=getattr(header, "frame_id", None) if frame_id is None else frame_id,
                source=source, validity="valid")


def numeric_valid(values, minimum):
    return len(values) >= minimum and all(math.isfinite(float(v)) for v in values)


def pose_fields(values, verified=False):
    row = dict(raw_values=values, validity="valid" if numeric_valid(values, 6) else "invalid",
               position_unit="mm" if verified else None, pose_kind="TCP_from_joint_feedback_FK" if verified else "unknown",
               semantic_frame="robot_base" if verified else None,
               orientation_representation="rotation_vector_rad" if verified else "unknown",
               quaternion_order="xyzw", velocity=None, work_pose=None, contact_offset=None)
    row.update(zip(["x", "y", "z", "rx", "ry", "rz"], values[:6]))
    if verified and numeric_valid(values, 6):
        a = math.sqrt(sum(float(v)**2 for v in values[3:6]))
        scale = math.sin(a / 2) / a if a > 1e-12 else 0.5
        row.update(zip(["qx", "qy", "qz"], [float(v) * scale for v in values[3:6]]))
        row["qw"] = math.cos(a / 2)
    return row


class ExecutionRecorder:
    def __init__(self, root, tag, metadata, queue_size=8192, warn=None):
        if queue_size < 1:
            raise ValueError("metrics_queue_size must be positive (bounded queue required)")
        self.root = Path(root)
        self.tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag or "run")[:100]
        self.metadata = metadata
        self.queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.warn = warn or (lambda text: None)
        self.seq = 0
        self.counts = {}
        self.source_counts = {}
        self.write_errors = []
        self.last_error = None
        self.path = None
        self.run_id = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="inference-metrics-writer", daemon=True)
        self.thread.start()

    def emit(self, stream, record):
        # Short memory-only critical section. Each producer supplies an owned snapshot.
        with self.lock:
            self.seq += 1
            row = dict(record, logger_seq=self.seq)
            c = self.counts.setdefault(stream, dict(received=0, enqueued=0, dropped=0, written=0))
            sc = self.source_counts.setdefault(stream+":"+str(record.get("source", "")),
                dict(received=0, enqueued=0, dropped=0, written=0))
            c["received"] += 1
            sc["received"] += 1
            if self.closed or self.last_error is not None:
                c["dropped"] += 1
                sc["dropped"] += 1
                return False
            try:
                self.queue.put_nowait((stream, row))
                c["enqueued"] += 1
                sc["enqueued"] += 1
                return True
            except queue.Full:
                c["dropped"] += 1
                sc["dropped"] += 1
                return False

    def event(self, name, timing, **details):
        return self.emit("events", dict(timing, event=name, details=details))

    def close(self, timeout=3.0):
        with self.lock:
            self.closed = True
        self.stop.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            self.warn("[METRICS] flush timeout; writer still draining; summary may be absent")
        return not self.thread.is_alive()

    def _json(self, name, value, exclusive=False):
        # Atomic replacement is only used for OUR newly created per-run state files.
        target = self.path / name
        if exclusive:
            with target.open("x") as f:
                json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        else:
            temp = self.path / (name + ".tmp")
            with temp.open("w") as f:
                json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temp, target)

    def _error(self, exc):
        self.last_error = repr(exc)
        self.write_errors.append(self.last_error)
        self.warn("[METRICS] WRITE ERROR (control unchanged): " + self.last_error)
        try:
            self._json("logger_error.json", dict(run_id=self.run_id, errors=self.write_errors))
        except Exception:
            pass

    def _provenance(self):
        """Background only: hashes and copies configuration, never unpickle checkpoints."""
        root = self.metadata.get("code_root")
        if root:
            try:
                def git(*args):
                    return subprocess.run(["git", "-C", root, *args], capture_output=True,
                                          text=True, timeout=10, check=True).stdout
                self.metadata["git_commit"] = git("rev-parse", "HEAD").strip()
                self.metadata["git_status"] = git("status", "--short", "--untracked-files=normal")
                with (self.path / "code_changes.patch").open("x") as f:
                    f.write(git("diff", "--", "behavior_ws/src/nrs_imitation", "source/data/dataset.py",
                                "scripts/flow/flow_train_core.py"))
            except Exception as exc:
                self.metadata["git_error"] = repr(exc)
        artifacts = []
        (self.path / "artifacts").mkdir()
        for n, filename in enumerate(self.metadata.pop("artifact_paths", [])):
            p = Path(filename)
            item = dict(path=str(p), exists=p.is_file())
            if p.is_file():
                digest = hashlib.sha256()
                with p.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                item.update(sha256=digest.hexdigest(), bytes=p.stat().st_size)
                # Small source/config artifacts only, never duplicate large checkpoints.
                if p.stat().st_size < 4 * 1024 * 1024:
                    dest = self.path / "artifacts" / f"{n}_{p.name}"
                    shutil.copyfile(p, dest)
                    item["copy"] = str(dest.relative_to(self.path))
                if p.name == "homography.json":
                    try:
                        self.metadata["camera_calibration_candidate"] = dict(
                            path=str(p), data=json.loads(p.read_text()),
                            status="archived existing calibration, validity at runtime unverified; NOT applied by logger")
                    except (ValueError, OSError) as exc:
                        item["parse_error"] = repr(exc)
            artifacts.append(item)
        self.metadata["artifacts"] = artifacts

    def _diagnostic(self, handle, name, **details):
        with self.lock:
            self.seq += 1
            seq = self.seq
        # Receipt ROS is unavailable on the standalone writer; don't forge it.
        row = dict(run_id=self.run_id, logger_seq=seq, source="logger_writer", source_seq=None,
                   source_stamp_ns=None, receipt_ros_ns=None, receipt_monotonic_ns=time.monotonic_ns(),
                   stamp_basis="writer_monotonic_only", frame_id=None, validity="valid",
                   event=name, details=details)
        handle.write(json.dumps(clean(row), allow_nan=False) + "\n")

    def _run(self):
        handles = {}
        writers = {}
        stats = {}
        roi = dict(reference=None, mask=None, initial_image=None, final_image=None,
                   calibration_ref="metadata.json:camera_calibration_candidate",
                   evaluation_roi=None, reason="manual ROI required; no mask/canonicalization enabled by logger")
        last_flush = time.monotonic()
        reported_drop = 0
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            prefix = self.tag + "_" + time.strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns()}_"
            self.path = Path(tempfile.mkdtemp(prefix=prefix, dir=self.root))
            self.run_id = self.path.name
            self.metadata.update(run_id=self.run_id, schema_version=1,
                                 queue_size=self.queue.maxsize, middleware_loss_count=None)
            roi["run_id"] = self.run_id
            self._json("metadata.json", self.metadata, exclusive=True)
            self._json("roi.json", roi, exclusive=True)
            for stream, fields in FIELDS.items():
                f = (self.path / (stream + ".csv")).open("x", newline="")
                handles[stream] = f
                writers[stream] = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
                writers[stream].writeheader()
            handles["events"] = (self.path / "events.jsonl").open("x")
            self.ready.set()
            self.warn("[METRICS] run directory: " + str(self.path))
            # Hashing 1GB weights would block draining the queue for seconds. Checkpoint
            # identity is path/size/mtime + config/normalizer hashes, not a promised SHA.
            self._provenance()
            self._json("metadata.json", self.metadata)
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    stream, row = self.queue.get(timeout=0.1)
                except queue.Empty:
                    stream = None
                if stream is not None:
                    row["run_id"] = self.run_id
                    if stream == "roi":
                        if roi["reference"] is None:
                            roi["reference"] = row
                            self._json("roi.json", roi)
                    elif stream == "snapshot":
                        kind, rgb = row.pop("kind"), row.pop("rgb")
                        if roi.get(kind) is None:
                            import cv2
                            (self.path / "snapshots").mkdir(exist_ok=True)
                            dest = self.path / "snapshots" / (kind + ".png")
                            if not cv2.imwrite(str(dest), rgb if kind == "mask" else rgb[:, :, ::-1]):
                                raise OSError("image encoding failed: " + str(dest))
                            roi[kind] = dict(row, path=str(dest.relative_to(self.path)),
                                             shape=list(rgb.shape))
                            self._json("roi.json", roi)
                    elif stream == "events":
                        handles[stream].write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")
                    else:
                        writers[stream].writerow({k: cell(v) for k, v in row.items()})
                    key = stream + ":" + str(row.get("source", ""))
                    s = stats.setdefault(key, dict(rows=0, first_ns=None, last_ns=None, max_gap_ns=0, retrograde=0))
                    now = row.get("receipt_monotonic_ns")
                    if now is not None:
                        if s["last_ns"] is not None:
                            gap = now - s["last_ns"]
                            s["max_gap_ns"] = max(s["max_gap_ns"], gap)
                            s["retrograde"] += int(gap < 0)
                        if s["first_ns"] is None:
                            s["first_ns"] = now
                        s["last_ns"] = now
                    s["rows"] += 1
                    with self.lock:
                        self.counts[stream]["written"] += 1
                        self.source_counts[key]["written"] += 1
                    self.queue.task_done()
                if time.monotonic() - last_flush >= 0.5:
                    with self.lock:
                        dropped = sum(v["dropped"] for v in self.counts.values())
                        counts = {k: dict(v) for k, v in self.counts.items()}
                        source_counts = {k: dict(v) for k, v in self.source_counts.items()}
                    if dropped != reported_drop:
                        self._diagnostic(handles["events"], "logger_overflow_or_drop", dropped_total=dropped)
                        self.warn(f"[METRICS] dropped records={dropped}; see events/summary")
                        reported_drop = dropped
                    for f in handles.values():
                        f.flush()
                    self._json("logger_status.json", dict(run_id=self.run_id, counts=counts,
                              source_counts=source_counts,
                              queue_pending=self.queue.qsize(), write_errors=self.write_errors))
                    last_flush = time.monotonic()
        except Exception as exc:
            self._error(exc)
            try:
                self._diagnostic(handles["events"], "logger_write_error", error=repr(exc))
            except Exception:
                pass
        finally:
            self.ready.set()
            try:
                with self.lock:
                    dropped = sum(v["dropped"] for v in self.counts.values())
                if dropped != reported_drop:
                    self._diagnostic(handles["events"], "logger_overflow_or_drop", dropped_total=dropped)
            except Exception:
                pass
            for f in handles.values():
                try:
                    f.close()
                except Exception as exc:
                    self._error(exc)
            if self.path is not None:
                try:
                    with self.lock:
                        counts = {k: dict(v) for k, v in self.counts.items()}
                        source_counts = {k: dict(v) for k, v in self.source_counts.items()}
                    self._json("summary.json", dict(run_id=self.run_id, counts=counts, streams=stats,
                        source_counts=source_counts, write_error_count=len(self.write_errors),
                        writer_diagnostic_rows_note="logger_writer events are additional to enqueued stream counts",
                        write_errors=self.write_errors, queue_pending=self.queue.qsize(),
                        drained=self.queue.empty() and not self.write_errors,
                        middleware_loss_count=None, physical_task_completion="unknown"))
                except Exception as exc:
                    self._error(exc)
