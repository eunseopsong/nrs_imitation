"""Offline, reproducible RA-L-style tables from the selected 2026-09-20 logs.

No ROS imports, services, robot commands, network access or source-log writes.
Tables are stored under the existing results/table directory.
"""
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nrs_ral_mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/table/20260920"
LOGS = ROOT / "logs/inference_metrics"
ARCHIVE = ROOT / "results/20260920"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = set()
KST = ZoneInfo("Asia/Seoul")
AUTHOR_URL = "https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/"


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_csv(path):
    SOURCES.add(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_events(path):
    SOURCES.add(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_csv(name, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return fields


def mono(event):
    return event["receipt_monotonic_ns"]


def force_metrics(rows, t0, stop):
    # Same convention as the previously archived E1 comparison: filter the
    # window first, then weight left-held samples between recorded receipts.
    # Do not extrapolate at boundaries or integrate gaps exceeding 0.2 s.
    t = np.array([(int(r["receipt_monotonic_ns"]) - t0) / 1e9 for r in rows])
    f = np.array([float(r["fz"]) for r in rows])
    keep = (t >= 0) & (t <= stop) & np.isfinite(f)
    t, f = t[keep], f[keep]
    assert len(t) >= 2 and np.all(np.diff(t) > 0)
    dt = np.diff(t)
    valid = (dt > 0) & (dt <= .2)
    w, v = dt[valid], f[:-1][valid]
    mean = float(np.average(v, weights=w))
    return dict(force_samples=len(t), covered_s=float(w.sum()),
                excluded_gap_s=float(dt[~valid].sum()),
                mean_fz_N=mean, temporal_sd_fz_N=float(np.sqrt(np.average((v-mean)**2, weights=w))),
                max_fz_N=float(f.max()), min_fz_N=float(f.min()), peak_t_s=float(t[f.argmax()]),
                time_fz_ge_3N_s=float(w[v >= 3].sum()),
                time_fz_ge_50N_s=float(w[v >= 50].sum()))


def pose_metrics(rows, t0, stop):
    keep = [r for r in rows if 0 <= (int(r["receipt_monotonic_ns"])-t0)/1e9 <= stop]
    p = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in keep])
    t = np.array([(int(r["receipt_monotonic_ns"])-t0)/1e9 for r in keep])
    assert len(p) >= 2
    valid = (np.diff(t) > 0) & (np.diff(t) <= .2)
    return dict(pose_samples=len(p), xy_path_mm=float(np.linalg.norm(np.diff(p[:, :2], axis=0)[valid], axis=1).sum()),
                min_z_mm=float(p[:, 2].min()), max_z_mm=float(p[:, 2].max()))


def latency(events):
    starts = {e["details"]["inference_id"]: mono(e) for e in events if e["event"] == "inference_start"}
    values = [(mono(e)-starts[e["details"]["inference_id"]])/1e6 for e in events
              if e["event"] == "inference_end" and e["details"]["inference_id"] in starts]
    return len(values), float(np.median(values[1:])) if len(values) > 1 else None


def quality(directory):
    s = read_json(directory / "summary.json")
    return dict(logger_drops=sum(v["dropped"] for v in s["counts"].values()),
                logger_write_errors=len(s.get("write_errors", [])), logger_drained=s.get("drained"))


selected = read_json(ARCHIVE / "selected_runs.json")
prior = read_json(ARCHIVE / "comparison.json")
e1, e2, audit, details = [], [], [], {}
for method in ("B", "C"):
    directory = ARCHIVE / selected[method]["logs"]
    events = read_events(directory / "events.jsonl")
    meta = read_json(directory / "metadata.json")
    t0 = mono(next(e for e in events if e["event"] == "inference_end"))
    end = mono(next(e for e in events if e["event"] == "run_interrupted"))
    count, inference_ms = latency(events)
    row = dict(experiment="E1", method=method, attempt=1, selected=True, n=1,
               force_observation="OFF" if method == "B" else "ON", learned_force_action="ON",
               window_s=37.5, tracking_duration_s=(end-t0)/1e9,
               **force_metrics(read_csv(directory / "wrench.csv"), t0, 37.5),
               **pose_metrics(read_csv(directory / "tcp_pose.csv"), t0, 37.5),
               inference_count=count, warm_inference_median_ms=inference_ms,
               transport="service_stream", outcome="Operator interrupted; task success unverified",
               removal_rate="Not evaluated", physical_success="Not evaluated",
               **quality(directory), run_id=directory.parent.name, log_directory=str(directory))
    ref = prior["runs"][method]
    mapping = {"mean_fz_N": "mean_N", "temporal_sd_fz_N": "std_N", "max_fz_N": "peak_N",
               "time_fz_ge_3N_s": "time_ge_3N_s", "time_fz_ge_50N_s": "time_ge_50N_s", "covered_s": "covered_s"}
    for key, refkey in mapping.items():
        assert np.isclose(row[key], ref["common_force"][refkey], atol=1e-8, rtol=0), (method, key)
    assert np.isclose(row["xy_path_mm"], ref["common_xy_path_mm"], atol=1e-8, rtol=0)
    assert np.isclose(inference_ms, ref["warm_inference_median_ms"], atol=1e-8, rtol=0)
    e1.append(row)
    details["E1_"+method] = dict(checkpoint=meta["checkpoint"], force_observation=meta["force_observation"],
                                 video_paths=[str(ARCHIVE/v["file"]) for v in selected[method]["videos"]],
                                 window_origin="first inference_end receipt", t0_monotonic_ns=t0,
                                 removal_note="Uncalibrated heatmaps are not material removal measurements.")

# Preserve all previously catalogued E1 attempts, including exclusions.
for r in read_csv(ARCHIVE / "runs_index.csv"):
    audit.append(dict(experiment="E1", method=r["condition"], start=r["start"],
                      included_in_main_table=r["selected"] == "True", execution_started=int(r["inference_count"]) > 0,
                      status=r["status"], note=r["checkpoint_family"], run_id=r["run_id"],
                      provider_run_id=r["run_id"], log_directory=str(ARCHIVE/r["directory"]/"logs")))

selected_e2 = {
    "E2_line_R_validation_02_executor_20260920T195830_1789901910681378337_w5_mo1k8": ("R", 1, "Release timeout"),
    "E2_line_T_r01_executor_20260920T201023_1789902623731930402_xpzzpqr3": ("T", 1, "Path ended*"),
    "E2_line_C_r01_executor_20260920T201310_1789902790957592437_gwzlij3x": ("C", 1, "Manual abort"),
    "E2_line_C_r01_executor_20260920T201420_1789902860394556982_3l8f6539": ("C", 2, "Mode lost*")}
providers = {}
for directory in sorted(LOGS.glob("E2*20260920T*")):
    if "_executor_" not in directory.name:
        m = read_json(directory / "metadata.json")
        providers[m["e2_session_id"]] = (directory, m)

all_started = []
for directory in sorted(LOGS.glob("E2*_executor_20260920T*")):
    events = read_events(directory / "events.jsonl")
    meta = read_json(directory / "metadata.json")
    start = next((e for e in events if e["event"] == "execution_start"), None)
    provider, pm = providers[meta["session_id"]]
    pe = read_events(provider / "events.jsonl")
    selected_main = directory.name in selected_e2
    started = start is not None
    status_events = [e for e in events if e["event"] in
                     ("trajectory_end", "normal_completion", "safety_stop", "manual_abort", "stop_not_verified", "shutdown_without_verified_stop")]
    last_status = status_events[-1]["event"] if status_events else "No tracking start"
    ast = datetime.fromtimestamp(events[0]["receipt_ros_ns"] / 1e9, KST).isoformat()
    note = ("Selected one-way R; release verification failed before demo-start return" if selected_main and meta["method"] == "R"
            else "Selected pilot attempt" if selected_main else "Superseded two-pass R protocol" if started else "Excluded: no execution_start")
    audit.append(dict(experiment="E2", method=meta["method"], start=ast,
                      included_in_main_table=selected_main, execution_started=started, status=last_status,
                      note=note, run_id=directory.name, provider_run_id=provider.name, log_directory=str(directory)))
    if not started:
        continue
    t0 = mono(start)
    first_stop = next(e for e in events if e["event"] == "stop_requested" and mono(e) > t0)
    duration = (mono(first_stop)-t0)/1e9
    method, attempt, outcome = selected_e2.get(directory.name, (meta["method"], 0, "Superseded two-pass R"))
    fm = force_metrics(read_csv(directory / "wrench.csv"), t0, duration)
    ps = pose_metrics(read_csv(directory / "tcp_pose.csv"), t0, duration)
    commands = read_csv(directory / "commands.csv")
    sent = [r for r in commands if r["command_stage"] == "node_sent" and r["force_unit"] == "N"
            and 0 <= (int(r["receipt_monotonic_ns"])-t0)/1e9 <= duration]
    gates = [json.loads(r["details"])["contact"] for r in sent]
    command_times = np.array([int(r["receipt_monotonic_ns"]) for r in sent], dtype=np.int64)
    count, inf_ms = latency(pe)
    holds = [e for e in events if mono(e) > t0 and e.get("details", {}).get("controller_hold_verified") is True]
    terminal = status_events[-1] if status_events else first_stop
    row = dict(experiment="E2", method=method, attempt=attempt, selected=selected_main, n=1,
               force_observation="ON" if method == "C" else "N/A", learned_force_action="ON" if method == "C" else "N/A",
               window_s=duration, tracking_duration_s=duration, **fm, **ps,
               inference_count=count, warm_inference_median_ms=inf_ms, transport="timed_topic",
               outcome=outcome, stop_reason=terminal["event"],
               hold_verified=bool(holds) and terminal["event"] != "shutdown_without_verified_stop",
               home_pose_verified=bool(terminal.get("details", {}).get("home_pose_reached", False)),
               contact_release="Not verified", removal_rate="Not evaluated", physical_success="Not evaluated",
               gate_transitions=int(np.count_nonzero(np.diff(np.array(gates, int)))),
               node_sent_fz_min_N=min(float(r["fz"]) for r in sent),
               node_sent_fz_max_N=max(float(r["fz"]) for r in sent),
               command_gap_max_ms=float(np.diff(command_times).max()/1e6),
               terminal_event_elapsed_s=(mono(terminal)-t0)/1e9,
               **quality(directory), provider_logger_drops=quality(provider)["logger_drops"],
               provider_logger_write_errors=quality(provider)["logger_write_errors"],
               run_id=directory.name, provider_run_id=provider.name, log_directory=str(directory))
    all_started.append(row)
    if selected_main:
        e2.append(row)
    phases = [e for e in events if e["event"] == "provider_phase"]
    extra = {}
    proc = next((e for e in phases if e["details"]["phase"] == "processing"), None)
    rampout = next((e for e in phases if e["details"]["phase"] == "force_ramp_out"), None)
    if proc and rampout:
        extra["provider_processing_phase"] = dict(duration_s=(mono(rampout)-mono(proc))/1e9,
            **force_metrics(read_csv(directory / "wrench.csv"), mono(proc), (mono(rampout)-mono(proc))/1e9),
            note="Recipe phase; not independently confirmed physical machining interval")
    stamp = ast[0:10].replace("-", "") + "_" + ast[11:19].replace(":", "")
    videos = sorted((Path.home()/"Videos/Screencasts").glob("*"+stamp+"*"+pm["run_tag"]+"*"))
    details[directory.name] = dict(provider_run_id=provider.name, t0_monotonic_ns=t0,
        config_sha256=pm.get("e2_config_sha256"), checkpoint=pm.get("checkpoint"),
        window_origin="execution_start receipt", window_end="first stop_requested receipt; excludes R return",
        video_paths=[str(v) for v in videos], outcome_events=[e for e in status_events if mono(e) > t0],
        first_stop=first_stop, configuration=meta["config"], **extra)

e2.sort(key=lambda r: (dict(R=0, T=1, C=2)[r["method"]], r["attempt"]))
assert [(r["method"], r["attempt"]) for r in e2] == [("R",1),("T",1),("C",1),("C",2)]
assert all(r["logger_drops"] == r["logger_write_errors"] == 0 for r in [*e1, *e2])
assert e2[0]["outcome"] == "Release timeout" and not e2[-1]["hold_verified"]

protocol = [
    dict(experiment="E1", method="A", label="Motion-only IL", measured_force_input="OFF", learned_force_action="OFF",
         force_reference="Externally tuned F0", controller="Maintained", status="Deferred; no A trial or A-specific training result"),
    dict(experiment="E1", method="B", label="Force-obs OFF", measured_force_input="OFF", learned_force_action="ON",
         force_reference="Policy prediction", controller="Maintained", status="Selected 2026-09-16 OFF checkpoint; one trial"),
    dict(experiment="E1", method="C", label="Force-obs ON", measured_force_input="ON", learned_force_action="ON",
         force_reference="Policy prediction", controller="Maintained", status="Selected latest E1 ON run at 17:17; one trial"),
    dict(experiment="E2", method="R", label="Rule", measured_force_input="N/A (no IL policy)", learned_force_action="N/A (no IL policy)",
         force_reference="+18 N during one-way processing", controller="Maintained", status="19:58 single-pass validation; return incomplete"),
    dict(experiment="E2", method="T", label="Teacher replay", measured_force_input="N/A (no IL policy)", learned_force_action="N/A (no IL policy)",
         force_reference="episode_29 recorded pose/force/time", controller="Maintained", status="20:10 selected replay; 14.9104 s source duration"),
    dict(experiment="E2", method="C", label="Force-obs ON IL", measured_force_input="ON", learned_force_action="ON",
         force_reference="2026-09-16 ON policy prediction", controller="Maintained", status="20:13 and 20:14 pilot attempts; reported vibration")]

notes = [
    ("Scope", "E1 B/C and E2 R/T/C on 2026-09-20; A is deferred. These are descriptive pilot results, not a completed controlled evaluation."),
    ("E1 selection", "The explicitly selected archived B 16:23 and C 17:17 runs are retained. The earlier 17:15 C and 09/10 checkpoint baseline are not pooled."),
    ("E2 selection", "R=19:58 one-way validation; T=20:10 episode_29; C1=20:13; C2=20:14. Both C attempts are shown independently. Prior round-trip R is in the audit and all-started CSV."),
    ("A", "Motion-only A requires separate training without force observation or learned force action. Deferred; no fabricated A numbers."),
    ("E1 window", "0–37.5 s after first inference_end. Startup PTP excluded. Every published E1 statistic was recomputed and matched against the archived comparison."),
    ("E2 window", "execution_start through the first stop_requested. R return/lift is excluded from force/path statistics but its failed release verification is included in outcome. Durations differ and are not comparable completed cycle times."),
    ("Mean and SD", "Receipt-time weighted mean and population SD within each individual trace. The ± term is temporal fluctuation, not variation across independent trials or a confidence interval."),
    ("Coverage", "Left-held successive samples within each window; only 0 < dt <= 0.2 s. No extrapolation at boundaries. Coverage and excluded gaps are in numeric CSVs."),
    ("Fz", "Filtered, republished robot-base Z force in N. No verified surface-normal calibration, pressure, raw sensor trace, or full-bandwidth force peak measurement. Recording is at most 20 Hz."),
    ("Threshold durations", "tau3 and tau50 integrate time with measured Fz >= 3 and >= 50 N. These are descriptive thresholds, not contact success labels or certified safety limits."),
    ("Path length", "Sum of successive measured TCP XY displacements, excluding gaps > 0.2 s. Includes approach and all motion within the stated window."),
    ("Inference latency", "Median inference_start-to-inference_end event latency excluding the first call; not a pure GPU benchmark. N/A for R/T."),
    ("R outcome", "Single-pass trajectory ended; vertical lift reached clearance height. Contact-release verification timed out; demo-start XY return was not begun. Controlled hold subsequently verified."),
    ("T outcome", "Requested trajectory time elapsed and controller hold verified. final_target_reached=false; physical contact release and completed polishing not verified."),
    ("C1 outcome", "User interrupted after 9.592 s tracking; controlled hold verified. Vibration reported. Not a successful completed trial."),
    ("C2 outcome", "Force mode lost or stale at 18.320 s. STOP acknowledgement and physical hold were not verified; later shutdown_without_verified_stop. Do not infer a confirmed robot crash or physical hold."),
    ("Removal / success", "Actual material removal, stain removal ratio, task success rate, tracking RMSE and pressure are not evaluated. k=1 Preston-like heatmaps are uncalibrated proxies and were not substituted for measurements."),
    ("Comparability", "E1 used service_stream with 35-point pose smoothing; E2 used timed_topic without that smoothing and with different plan timing. Same C checkpoint does not make E1 and E2 executions equivalent."),
    ("Protocol differences", "E2 R includes a separate post-path lift/return procedure; T/C terminate in hold. R was a validation recipe, and no common verified processing interval exists for all methods."),
    ("Uncontrolled factors", "RPM unknown; specimen_id=default does not confirm equivalent stain preparation/reset. Do not claim causal superiority or significance from these runs."),
    ("Statistics", "E1 n=1 per condition. E2 R n=1, T n=1 and C has two interrupted attempts shown separately; no pooled mean, p-value, CI or success-rate denominator is invented."),
    ("Audit", "The existing E2 validation manifest still marks R02 prepared_not_started. This export uses the actual execution_start and event records; historical source files are preserved."),
    ("Format", "English captions above tables, Roman numbering, serif type and horizontal rules, designed at 7.16-inch full two-column width. IEEEtran table* LaTeX source and vector PDF/SVG are supplied."),
    ("RA-L author guidance", AUTHOR_URL),
    ("Reproduction", "python3 scripts/export_ral_e1_e2_20260920.py (offline only). Source hashes are in source_manifest.json."),
]
save_csv("E1_results.csv", e1)
save_csv("E2_results.csv", e2)
save_csv("E1_E2_results.csv", [*e1, *e2])
save_csv("E2_all_started_runs.csv", all_started)
save_csv("run_audit.csv", audit)
save_csv("conditions.csv", protocol)
save_csv("metric_definitions.csv", [dict(item=a, description=b) for a,b in notes])
save_json("metrics_and_provenance.json", dict(e1=e1, e2=e2, e2_all_started=all_started, details=details, notes=notes))

def f(value):
    return f"{value:.2f}"


e1_display = [["B: Force-obs OFF" if r["method"] == "B" else "C: Force-obs ON", "1",
               f(r["mean_fz_N"])+" ± "+f(r["temporal_sd_fz_N"]), f(r["max_fz_N"]),
               f(r["time_fz_ge_3N_s"]), f(r["time_fz_ge_50N_s"]), f(r["xy_path_mm"]),
               f(r["warm_inference_median_ms"])] for r in e1]
e2_display = [["R (one-way)" if r["method"] == "R" else "T (ep. 29)" if r["method"] == "T" else "C, attempt "+str(r["attempt"]),
               f(r["tracking_duration_s"]), f(r["mean_fz_N"])+" ± "+f(r["temporal_sd_fz_N"]),
               f(r["max_fz_N"]), f(r["time_fz_ge_50N_s"]), f(r["xy_path_mm"]), r["outcome"]] for r in e2]
TABLES = [dict(stem="E1_table", number="I", title="E1: FORCE-OBSERVATION ABLATION AT 90°",
    headers=["Method", "n", "Fz: mean ± SD\n[N]", "Peak Fz\n[N]", "τ3\n[s]", "τ50\n[s]", "XY path\n[mm]", "Inference\n[ms]"],
    rows=e1_display, widths=[1.45,.30,1.22,.69,.62,.62,.77,.77],
    footnotes=["One selected run per condition; A (Motion-only IL) is deferred. Both B and C learn force actions and retain the lower force controller.",
      "Window: 0–37.5 s after the first policy plan, including approach. Mean ± SD denotes time-weighted within-run force fluctuation, not across-trial uncertainty.",
      "Fz is measured base-axis feedback. τ3/τ50: time with Fz ≥ 3/50 N. Inference: warm-call median. Removal and physical task success were not evaluated."]),
    dict(stem="E2_table", number="II", title="E2: RULE, TEACHER REPLAY AND IL — PILOT RUNS",
    headers=["Method / attempt", "Tracking\n[s]", "Fz: mean ± SD\n[N]", "Peak Fz\n[N]", "τ50\n[s]", "XY path\n[mm]", "Recorded outcome"],
    rows=e2_display, widths=[1.03,.60,1.25,.66,.60,.72,1.58],
    footnotes=["One row per attempt. Tracking spans execution_start to the first stop request; force/path metrics use that interval. These are unequal, partly interrupted observation windows, not completed cycle times.",
      "R: one-way path ended, then release verification timed out before demo-start return (later hold verified). T*: replay time elapsed and hold verified; final target/contact release not verified.",
      "C1: manual abort with hold verified. C2*: Force mode lost/stale; STOP acknowledgement and physical hold unverified. Both C attempts had reported vibration. Prior two-pass R is in the run audit.",
      "Mean ± SD is temporal, not across-trial variation. τ50: measured Fz ≥ 50 N duration. Removal/success are unassessed; no performance ranking is supported."])]

plt.rcParams.update({"font.family":"Liberation Serif", "font.size":8, "pdf.fonttype":42, "ps.fonttype":42})


def draw_table(ax, spec):
    ax.axis("off")
    ax.text(.5,.97,"TABLE "+spec["number"],ha="center",va="top",fontsize=9)
    ax.text(.5,.90,spec["title"],ha="center",va="top",fontsize=9)
    n = len(spec["rows"])
    top=.78
    height=.31 if n == 2 else .39
    widths=np.array(spec["widths"])/sum(spec["widths"])
    table=ax.table(cellText=spec["rows"],colLabels=spec["headers"],colWidths=widths,
                   cellLoc="center",loc="center",bbox=[.012,top-height,.976,height])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r,c),cell in table.get_celld().items():
        cell.set_facecolor("white");cell.visible_edges="";cell.PAD=.025
        if r==0: cell.get_text().set_weight("bold");cell.visible_edges="TB";cell.set_linewidth(.7)
        if r==n: cell.visible_edges="B";cell.set_linewidth(.7)
        if c==0: cell.get_text().set_ha("left")
    foot_y=top-height-.038
    axes_height_in=ax.figure.get_size_inches()[1]*ax.get_position().height
    line_height=(7.1*1.18/72)/axes_height_in
    for note in spec["footnotes"]:
        lines=wrap(note,width=147)
        ax.text(.012,foot_y,"\n".join(lines),ha="left",va="top",fontsize=7.1,linespacing=1.18)
        foot_y-=line_height*len(lines)+.04/axes_height_in
    return foot_y


for spec in TABLES:
    fig,ax=plt.subplots(figsize=(7.16,3.0 if spec["number"]=="I" else 4.0))
    fig.subplots_adjust(left=.025,right=.975,top=.995,bottom=.015)
    bottom=draw_table(ax,spec)
    assert bottom>0, (spec["stem"],bottom)
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/(spec["stem"]+"."+ext),dpi=600,facecolor="white")
    plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(7.16,7.2),gridspec_kw={"height_ratios":[3,4]})
fig.subplots_adjust(left=.025,right=.975,top=.995,bottom=.015,hspace=.02)
for ax,spec in zip(axes,TABLES): draw_table(ax,spec)
fig.savefig(OUT/"E1_E2_tables.png",dpi=600,facecolor="white")
fig.savefig(OUT/"E1_E2_tables.pdf",facecolor="white")
plt.close(fig)


def latex_escape(s):
    s=str(s).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_")
    return s.replace("±",r"$\pm$").replace("≥",r"$\geq$").replace("τ",r"$\tau$").replace("°",r"$^\circ$").replace("—","--").replace("–","--")


latex=["% IEEEtran two-column manuscript; requires \\usepackage{booktabs,graphicx}",
       "% ± values are within-run temporal SD, not across-trial SD. A is deferred."]
for spec in TABLES:
    latex += [r"\begin{table*}[t]", r"\centering",r"\caption{"+latex_escape(spec["title"])+"}",
              r"\label{tab:"+spec["stem"]+"}",r"\footnotesize",r"\setlength{\tabcolsep}{5pt}",
              r"\begin{tabular}{l"+"c"*(len(spec["headers"])-1)+"}",r"\toprule",
              " & ".join(latex_escape(s.replace("\n"," ")) for s in spec["headers"])+r" \\",r"\midrule"]
    latex += [" & ".join(latex_escape(v) for v in row)+r" \\" for row in spec["rows"]]
    latex += [r"\bottomrule",r"\end{tabular}",r"\par\smallskip",r"\begin{minipage}{0.98\textwidth}\scriptsize"]
    latex += [latex_escape(note)+r"\par" for note in spec["footnotes"]]
    latex += [r"\end{minipage}",r"\end{table*}",""]
(OUT/"tables_ieee_ral.tex").write_text("\n".join(latex)+"\n")

# SpreadsheetML workbook: real numeric cells, no dependency installation.
NS="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG="http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("",NS)
ET.register_namespace("r",REL)


def sub(parent,tag,attrs=None,text=None):
    node=ET.SubElement(parent,"{"+NS+"}"+tag,attrs or {})
    if text is not None: node.text=str(text)
    return node


def col(n):
    result=""
    while n:
        n,r=divmod(n-1,26);result=chr(65+r)+result
    return result


def xml(node):
    return ET.tostring(node,encoding="utf-8",xml_declaration=True)


def sheet(title,subtitle,headers,rows,widths,landscape=True):
    ws=ET.Element("{"+NS+"}worksheet")
    sub(ws,"sheetPr");sub(ws[0],"pageSetUpPr",{"fitToPage":"1"})
    sub(ws,"dimension",{"ref":f"A1:{col(len(headers))}{len(rows)+4}"})
    views=sub(ws,"sheetViews");view=sub(views,"sheetView",{"workbookViewId":"0","showGridLines":"0"})
    sub(view,"pane",{"ySplit":"4","topLeftCell":"A5","activePane":"bottomLeft","state":"frozen"})
    sub(ws,"sheetFormatPr",{"defaultRowHeight":"26"})
    cs=sub(ws,"cols")
    for i,width in enumerate(widths,1):sub(cs,"col",{"min":str(i),"max":str(i),"width":str(width),"customWidth":"1"})
    data=sub(ws,"sheetData")
    for i,values in enumerate([[title],[subtitle],[],headers,*rows],1):
        if i==1:
            h=28
        elif i==2:
            h=max(65,14*math.ceil(len(subtitle)/(sum(widths)*.90))+12)
        elif i==3:
            h=10
        elif i==4:
            h=48
        else:
            lines=max(sum(max(1,math.ceil(len(part)/(width*.90))) for part in str(value).split("\n"))
                      for value,width in zip(values,widths))
            h=max(32,14*lines+10)
        rr=sub(data,"row",{"r":str(i),"ht":str(h),"customHeight":"1"})
        for j,value in enumerate(values,1):
            if value is None: value="N/A"
            numeric=isinstance(value,(float,int)) and not isinstance(value,bool)
            style=1 if i==1 else 2 if i==2 else 3 if i==4 else 5 if numeric else 4
            cc=sub(rr,"c",{"r":f"{col(j)}{i}","s":str(style)})
            if numeric:
                assert math.isfinite(value)
                sub(cc,"v",text=value)
            else:
                cc.set("t","inlineStr");sub(sub(cc,"is"),"t",text=str(value))
    sub(ws,"autoFilter",{"ref":f"A4:{col(len(headers))}{len(rows)+4}"})
    merges=sub(ws,"mergeCells",{"count":"2"})
    for r in (1,2):sub(merges,"mergeCell",{"ref":f"A{r}:{col(len(headers))}{r}"})
    sub(ws,"pageMargins",{"left":"0.25","right":"0.25","top":"0.4","bottom":"0.4","header":"0.2","footer":"0.2"})
    sub(ws,"pageSetup",{"orientation":"landscape" if landscape else "portrait","paperSize":"9","fitToWidth":"1","fitToHeight":"0"})
    return xml(ws)


styles=f'''<styleSheet xmlns="{NS}"><fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="16"/><name val="Times New Roman"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFEDEFF2"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left/><right/><top style="thin"><color rgb="FF000000"/></top><bottom style="thin"><color rgb="FF000000"/></bottom><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="6"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="0" fontId="2" fillId="2" borderId="1" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center" horizontal="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment vertical="center" horizontal="right"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''

def records_sheet(name,records,subtitle):
    headers=list(dict.fromkeys(k for r in records for k in r))
    return name,subtitle,headers,[[r.get(k) for k in headers] for r in records],[24 if k not in ("run_id","provider_run_id","log_directory","description","note") else 75 for k in headers]


sheets=[]
for spec in TABLES:
    # Presentation sheets have numeric cells wherever the displayed value is a single number.
    rows=[[float(v) if c>0 and str(v).replace(".","",1).isdigit() else v for c,v in enumerate(row)] for row in spec["rows"]]
    sheets.append(("E1" if spec["number"]=="I" else "E2", " ".join(spec["footnotes"]),
                   spec["headers"],rows,[26]+[17]*(len(spec["headers"])-2)+[26]))
sheets += [records_sheet("Numeric_results",[*e1,*e2],"Full precision metrics; SD is within-run temporal SD. N/A is unmeasured, not zero."),
           records_sheet("E2_all_started",all_started,"Includes superseded two-pass R for traceability; do not pool it with the selected one-way R."),
           records_sheet("Run_audit",audit,"Actual logged attempts, including exclusions. This report does not rewrite historical manifests."),
           records_sheet("Conditions",protocol,"A deferred. Lower force control maintained for all executed methods."),
           ("Definitions","Read these limitations before using the numbers in a manuscript.",["Item","Definition / limitation"],notes,[30,140])]
wb=ET.Element("{"+NS+"}workbook");wsheets=sub(wb,"sheets")
for i,(name,*_) in enumerate(sheets,1):sub(wsheets,"sheet",{"name":name,"sheetId":str(i),"{"+REL+"}id":f"rId{i}"})
names=[name for name,*_ in sheets]
defined=sub(wb,"definedNames")
for i,name in enumerate(names):sub(defined,"definedName",{"name":"_xlnm.Print_Titles","localSheetId":str(i)},f"'{name}'!$1:$4")
types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))+'</Types>'
rels=f'<Relationships xmlns="{PKG}">'+''.join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+f'<Relationship Id="rId{len(sheets)+1}" Type="{REL}/styles" Target="styles.xml"/></Relationships>'
xlsx=OUT/"E1_E2_results.xlsx"
with ZipFile(xlsx,"w",ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml",types)
    z.writestr("_rels/.rels",f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    z.writestr("xl/workbook.xml",xml(wb));z.writestr("xl/_rels/workbook.xml.rels",rels);z.writestr("xl/styles.xml",styles)
    for i,(name,subtitle,headers,rows,widths) in enumerate(sheets,1):
        title=TABLES[i-1]["title"] if i<=2 else name
        z.writestr(f"xl/worksheets/sheet{i}.xml",sheet(title,subtitle,headers,rows,widths))

# Artifact checks: CSV row counts and values, workbook XML/ZIP/row counts,
# numeric cells, and image integrity. No synthetic robot tests are involved.
with ZipFile(xlsx) as z:
    assert z.testzip() is None
    for name in z.namelist():ET.fromstring(z.read(name))
    for i,(_,_,headers,rows,_) in enumerate(sheets,1):
        root=ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml"))
        assert len(root.findall(f"{{{NS}}}sheetData/{{{NS}}}row"))==len(rows)+4
    root=ET.fromstring(z.read("xl/worksheets/sheet3.xml"))
    numeric_cell_count=len(root.findall(f".//{{{NS}}}c/{{{NS}}}v"))
    assert numeric_cell_count>100
from PIL import Image
image_info={}
for path in OUT.glob("*.png"):
    with Image.open(path) as im:
        image_info[path.name]=dict(size=im.size,dpi=im.info.get("dpi"));im.verify()
for name,expected in [("E1_results.csv",2),("E2_results.csv",4),("E1_E2_results.csv",6)]:
    with (OUT/name).open(encoding="utf-8-sig") as fobj: assert len(list(csv.DictReader(fobj)))==expected
save_json("validation.json",dict(e1_recomputed_matches_archived=True,e1_rows=2,e2_rows=4,
    e2_methods=[r["method"] for r in e2],a_status="Deferred",xlsx_sheets=names,
    numeric_cells=numeric_cell_count,images=image_info,hardware_commands_issued=False))

SOURCES.add(Path(__file__).resolve())
SOURCES.add(ROOT/"reports/20260920_e2_c_vibration/diagnosis.md")
manifest=[]
for path in sorted(SOURCES):
    manifest.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
save_json("source_manifest.json",dict(created_at=datetime.now(KST).isoformat(),sources=manifest))

readme="""# 2026-09-20 E1 / E2 — RA-L 형식 결과표

저장 경로: `results/table/20260920/` (실험 날짜 기준).

- **A: 보류.** A의 수치나 성공률은 만들지 않았습니다.
- **E1:** 기존에 선택한 16:23 B(9/16 OFF), 17:17 C(9/16 ON), 각 1회.
- **E2:** 19:58 편도 R, 20:10 episode_29 T, 20:13 C 1차, 20:14 C 2차를 모두 포함했습니다.
- 이전 19:38 왕복 R도 `E2_all_started_runs.csv` / Excel 같은 이름의 시트에 수치를 보존했습니다. 편도 R과는 합치지 않았습니다.

## 파일

- `E1_E2_tables.png`: 두 표를 합친 600 dpi 이미지.
- `E1_table.png`, `E2_table.png`: 각각의 논문용 영문 표. PDF/SVG 벡터 파일도 포함.
- `E1_E2_results.xlsx`: E1, E2, 원정밀도 수치, 모든 E2 실행, 실행 이력, 조건, 지표 정의의 7개 시트.
- `E1_results.csv`, `E2_results.csv`, `E1_E2_results.csv`: Excel 호환 UTF-8 BOM CSV, 숫자 원정밀도 유지.
- `run_audit.csv`: E1의 9개 기존 기록과 E2의 8개 시도를 포함한 이력/선택 근거.
- `tables_ieee_ral.tex`: IEEEtran의 `table*`용 표 소스; `booktabs,graphicx` 사용.
- `metrics_and_provenance.json`, `source_manifest.json`, `validation.json`: 상세 수치·종료 이벤트·원본 SHA-256·검증 결과.

## 해석

E1은 첫 정책 plan 이후 공통 0–37.5초를 재계산했으며, 기존 선택 B/C 결과와 수치가 일치합니다.
E2 수치는 실제 execution_start부터 첫 stop_requested까지입니다. R의 복귀 구간은 힘/경로 통계에 포함하지 않되, 복귀 실패는 결과 상태에 반영했습니다.

평균±표준편차는 **한 실행 내 시간 가중 힘 변동**입니다. 반복 실험 간 표준편차, 신뢰구간, 유의성 검정이 아닙니다. E2의 서로 다른/중단된 관측 시간을 완료 작업시간으로 비교할 수 없습니다.

R은 편도 경로 후 수직 상승을 수행했지만 접촉 해제 확인 시간 초과로 교시 시작 XY 복귀를 시작하지 못했습니다. T는 재생 시간 종료와 정지 유지가 확인됐으나 마지막 목표 도달과 접촉 해제는 미확인입니다. C 1차는 수동 중단 후 정지 유지 확인, C 2차는 Force 모드 상실/갱신 중단 후 STOP 응답과 정지 유지 미확인입니다. 두 C 시도는 진동이 보고된 실행입니다.

Fz는 로봇 base Z축의 필터된 측정 힘입니다. 검증된 표면 법선력이나 압력이 아닙니다. 기록은 최대 20 Hz이므로 모든 고주파 피크를 측정했다고 볼 수 없습니다. τ3/τ50는 3/50 N 이상 시간이며 성공/접촉/안전 판정이 아닙니다.

실측 제거량, 얼룩 제거율, 물리 작업 성공률 및 검증된 힘 추종 RMSE는 **미평가**입니다. k=1 removal heatmap을 실제 제거량으로 대체하지 않았습니다. 회전수는 미확인이고 시편/얼룩 초기화 동일성도 검증되지 않았습니다.

E1의 service_stream(35점 평활화)과 E2의 timed_topic 실행부가 다릅니다. 같은 C 체크포인트라도 E1/E2를 동등한 실행 조건으로 합산하거나 성능 우열을 주장하면 안 됩니다. R의 후처리 복귀 절차도 T/C와 다릅니다. 표는 현재 파일럿 실행의 기술적 기록입니다.

## 형식과 재현

RA-L의 IEEE 2단 형식 안내를 참고해 전폭 7.16 inch 영문 표, 상단 캡션, 로마 숫자 표 번호, serif 글꼴과 수평선으로 구성했습니다. 최종 원고에서는 실제 문서의 표 번호/캡션과 함께 검토해야 합니다.
공식 안내: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/

재생성: `python3 scripts/export_ral_e1_e2_20260920.py`
로봇 실행/제어기 변경 없이 저장된 로그만 읽습니다. 기존 로그·CSV·영상·실험 manifest는 수정하지 않았습니다.
"""
(OUT/"README.md").write_text(readme)
print(json.dumps(dict(output=str(OUT),e1=e1_display,e2=e2_display,
                     source_files=len(manifest),audit_rows=len(audit),files=sorted(p.name for p in OUT.iterdir())),ensure_ascii=False,indent=2))
