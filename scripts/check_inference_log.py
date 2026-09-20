#!/usr/bin/env python3
"""Read-only raw E1 log quality audit, including incomplete/crash logs. No alignment/fitting."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def inspect(path, max_gap_ms=None, max_age_ms=None, source_max_gap_ms=None):
    path = Path(path)
    source_max_gap_ms = source_max_gap_ms or {}
    issues, groups = [], {}
    def read_json(name):
        try:
            return json.loads((path / name).read_text())
        except (OSError, ValueError) as exc:
            issues.append(f"{name}: {exc}")
            return {}
    meta, roi = read_json("metadata.json"), read_json("roi.json")
    summary = read_json("summary.json")
    run_id = meta.get("run_id")
    id_mismatch = 0
    for name in ("wrench", "tcp_pose", "commands", "legacy"):
        file = path / (name + ".csv")
        if not file.exists():
            issues.append(f"missing {file.name}")
            continue
        with file.open(newline="") as f:
            for row in csv.DictReader(f):
                if None in row or any(v is None for v in row.values()):
                    issues.append(f"incomplete/malformed CSV row in {file.name} (possibly unclean shutdown)")
                key = name + ":" + str(row.get("source"))
                s = groups.setdefault(key, dict(rows=0, missing_source_stamp=0, invalid=0,
                    missing_required_values=0, nonfinite=0, monotonic_retrograde=0, ros_retrograde=0,
                    source_stamp_retrograde=0, repeated_source_stamp=0, logger_seq_retrograde=0,
                    gaps_ms=[], stale=0 if name == "legacy" and max_age_ms is not None else None,
                    cached_age_unavailable=0 if name == "legacy" else None,
                    frames=set(), semantic_frames=set(), last={}, first_mono=None))
                id_mismatch += row.get("run_id") != run_id
                s["rows"] += 1
                s["missing_source_stamp"] += not bool(row.get("source_stamp_ns"))
                s["invalid"] += row.get("validity") != "valid"
                s["frames"].add(row.get("frame_id") or "unknown")
                s["semantic_frames"].add(row.get("semantic_frame") or "unknown")
                required = ["fx", "fy", "fz", "tx", "ty", "tz"] if name == "wrench" else ["x", "y", "z", "rx", "ry", "rz"] if name == "tcp_pose" else []
                s["missing_required_values"] += sum(row.get(k) in (None, "") for k in required)
                for value in row.values():
                    try:
                        s["nonfinite"] += not math.isfinite(float(value))
                    except (TypeError, ValueError):
                        pass
                for field, counter in [("receipt_monotonic_ns", "monotonic_retrograde"),
                    ("receipt_ros_ns", "ros_retrograde"), ("source_stamp_ns", "source_stamp_retrograde"),
                    ("logger_seq", "logger_seq_retrograde")]:
                    try:
                        v = int(row[field])
                    except (TypeError, ValueError, KeyError):
                        if field != "source_stamp_ns":
                            issues.append(f"malformed {field}: {file.name} row {s['rows']}")
                        continue
                    if field in s["last"]:
                        delta = v - s["last"][field]
                        s[counter] += delta < 0
                        if field == "receipt_monotonic_ns":
                            s["gaps_ms"].append(delta / 1e6)
                        if field == "source_stamp_ns":
                            s["repeated_source_stamp"] += delta == 0
                    s["last"][field] = v
                    if field == "receipt_monotonic_ns" and s["first_mono"] is None:
                        s["first_mono"] = v
                if name == "legacy":
                    s["cached_age_unavailable"] += sum(not row.get(k) for k in ("pose_age_ns", "force_age_ns"))
                if s["stale"] is not None:
                    s["stale"] += any(float(row[k])/1e6 > max_age_ms for k in ("pose_age_ns", "force_age_ns") if row.get(k))
    for key, s in groups.items():
        gaps = s.pop("gaps_ms")
        limit = source_max_gap_ms.get(key.split(":", 1)[1], max_gap_ms)
        last, first = s.pop("last"), s.pop("first_mono")
        duration = (last.get("receipt_monotonic_ns", first)-first)/1e9 if first is not None else 0
        s.update(receipt_rate_hz=(s["rows"]-1)/duration if duration > 0 else None,
                 median_gap_ms=statistics.median(gaps) if gaps else None,
                 max_gap_ms=max(gaps) if gaps else None,
                 explicit_gap_limit_ms=limit,
                 gaps_over_explicit_limit=sum(g > limit for g in gaps) if limit is not None else None,
                 frames=sorted(s["frames"]), semantic_frames=sorted(s["semantic_frames"]))
    events, bad_event_lines, event_counts = 0, 0, {}
    event_path = path / "events.jsonl"
    if event_path.exists():
        with event_path.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    events += 1
                    id_mismatch += e.get("run_id") != run_id
                    event_counts[e.get("event", "unknown")] = event_counts.get(e.get("event", "unknown"), 0)+1
                except ValueError:
                    bad_event_lines += 1
    else:
        issues.append("events.jsonl missing")
    for obj in (roi, summary):
        if obj:
            id_mismatch += obj.get("run_id") != run_id
    missing = []
    if not any(k.startswith("wrench:") for k in groups):
        missing.append("actual force stream")
    if not any(k.startswith("tcp_pose:") for k in groups):
        missing.append("actual feedback TCP pose stream")
    if not any("ftdata:" in k or k.endswith("/ftdata") for k in groups):
        missing.append("timestamped full-rate base force; currentF alone may repeat acquisitions")
    for k in ("surface_normal", "compression_sign", "contact_point_offset", "effective_contact_area_mm2", "preston_coefficient", "rpm_setpoint"):
        if meta.get(k) is None:
            missing.append(k)
    if not roi.get("reference"):
        missing.append("frozen task reference")
    if not roi.get("evaluation_roi"):
        missing.append("fixed evaluation ROI/polygon (manual selection still required)")
    if not roi.get("initial_image"):
        missing.append("initial RGB snapshot")
    missing.extend(["verified deployed calibration and torque reference", "clock synchronization/acquisition age verification",
                    "explicit machining/intentional lift intervals", "rotary slip/contact model (TCP speed is not rotary speed)"])
    report = dict(run_id=run_id, streams=groups, events=events, event_counts=event_counts,
        malformed_event_lines=bad_event_lines, run_id_mismatches=id_mismatch,
        logger_counts=summary.get("counts"), logger_source_counts=summary.get("source_counts"),
        logger_write_errors=summary.get("write_errors"),
        queue_pending=summary.get("queue_pending"), middleware_loss_count=None,
        thresholds=dict(max_gap_ms=max_gap_ms, max_age_ms=max_age_ms,
                        source_max_gap_ms=source_max_gap_ms,
                        note="explicit receipt-clock thresholds only; no cross-clock ages, no silent interpolation"),
        roi=roi, rpm={k: meta.get(k) for k in ("rpm_setpoint", "rpm_setpoint_source", "measured_rpm", "rpm_assumed_constant")},
        insufficient_for_spatial_exposure=missing, issues=issues,
        quantitative_removal_ready=False, hardware_verified=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--max-gap-ms", type=float, help="Explicit receipt-gap warning limit (all sources; otherwise report only)")
    p.add_argument("--max-age-ms", type=float, help="Explicit cached legacy-sample age limit; NOT source acquisition age")
    p.add_argument("--source-max-gap-ms", action="append", default=[], metavar="SOURCE=MS",
                   help="Per-source receipt-gap limit, overrides --max-gap-ms; repeat for different sources")
    args = p.parse_args()
    try:
        limits = {source: float(ms) for source, ms in (item.rsplit("=", 1) for item in args.source_max_gap_ms)}
        if any(not math.isfinite(x) or x <= 0 for x in [*limits.values(), *[x for x in (args.max_gap_ms,args.max_age_ms) if x is not None]]):
            raise ValueError("thresholds must be finite and positive")
    except ValueError as exc:
        p.error(str(exc))
    report = inspect(args.run_dir, args.max_gap_ms, args.max_age_ms, limits)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    bad = report["issues"] or report["run_id_mismatches"] or report["malformed_event_lines"] or report["logger_write_errors"]
    bad = bad or any(s["invalid"] or s["nonfinite"] or s["missing_required_values"] or s["monotonic_retrograde"] for s in report["streams"].values())
    bad = bad or any(s["dropped"] for s in (report["logger_counts"] or {}).values())
    bad = bad or report["queue_pending"] or not report["streams"]
    bad = bad or any(s["ros_retrograde"] or s["source_stamp_retrograde"] or s["logger_seq_retrograde"]
                    or s["gaps_over_explicit_limit"] or s["stale"] for s in report["streams"].values())
    raise SystemExit(2 if bad else 0)


if __name__ == "__main__":
    main()
