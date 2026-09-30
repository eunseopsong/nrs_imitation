"""Offline, read-only comparison of two recorded runs. Outputs stay beside this script."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from check_inference_log import inspect

RUNS = {
    "ON_baseline": "rel_single_90deg_best_20260920T160636_1789887996648993205_r3lvdifh",
    "OFF_E1": "IL_minus_F_E1_off_20260920T162341_1789889021414718505_kkmoieb_",
}
WINDOW = 37.5  # common interval after the first policy plan; excludes startup PTP


def window(df, lo=0., hi=WINDOW):
    return df[(df.t >= lo) & (df.t <= hi)].copy()


def force_summary(df):
    """Left-held receipt samples, weighted by observed intervals. No extrapolation."""
    t, f = df.t.to_numpy(), df.fz.to_numpy()
    dt, v = np.diff(t), f[:-1]
    valid = (dt > 0) & (dt <= .2) & np.isfinite(v)
    dt, v = dt[valid], v[valid]
    mean = float(np.average(v, weights=dt))
    result = dict(covered_s=float(dt.sum()), rows=len(df), mean_N=mean,
                  std_N=float(np.sqrt(np.average((v - mean) ** 2, weights=dt))),
                  peak_N=float(f.max()), peak_t_s=float(t[f.argmax()]),
                  excluded_gap_s=float(np.diff(t)[~valid].sum()))
    for threshold in (3, 5, 10, 50, 80):
        mask = v >= threshold
        result[f"time_ge_{threshold}N_s"] = float(dt[mask].sum())
        result[f"fraction_ge_{threshold}N"] = float(dt[mask].sum() / dt.sum())
        result[f"mean_when_ge_{threshold}N"] = float(np.average(v[mask], weights=dt[mask])) if mask.any() else None
    return result


data, summaries = {}, {}
for label, run_id in RUNS.items():
    path = ROOT / "logs/inference_metrics" / run_id
    meta = json.loads((path / "metadata.json").read_text())
    roi = json.loads((path / "roi.json").read_text())
    status = json.loads((path / "summary.json").read_text())
    events = [json.loads(l) for l in (path / "events.jsonl").read_text().splitlines()]
    event = lambda name: [e for e in events if e["event"] == name]
    t0 = event("inference_end")[0]["receipt_monotonic_ns"]
    end = event("run_interrupted")[0]["receipt_monotonic_ns"]
    frames = {}
    for name in ("wrench", "tcp_pose", "commands", "legacy"):
        df = pd.read_csv(path / (name + ".csv"))
        df["t"] = (df.receipt_monotonic_ns - t0) / 1e9
        frames[name] = df
    commands = frames["commands"]
    sent = commands[(commands.command_stage == "node_sent") &
                    (commands.command_mode == "PTP9D_STREAM_SET_FORCE")].copy()
    frames["sent_force"] = sent
    origin = np.asarray(roi["reference"]["center_xy_mm"])
    pose = frames["tcp_pose"]
    pose[["rel_x", "rel_y"]] = pose[["x", "y"]].to_numpy() - origin
    f = force_summary(window(frames["wrench"]))
    peak_command = sent[sent.t <= f["peak_t_s"]].iloc[-1]
    f["last_sent_at_peak_N"] = float(peak_command.fz)
    f["last_sent_at_peak_age_s"] = float(f["peak_t_s"] - peak_command.t)
    p = window(pose)
    burst = window(sent, 9., 10.5)
    starts = {e["details"]["inference_id"]: e["receipt_monotonic_ns"] for e in event("inference_start")}
    latency = [(e["receipt_monotonic_ns"] - starts[e["details"]["inference_id"]]) / 1e6
               for e in event("inference_end")]
    reasons = [e["details"]["reason"] for e in event("safety_limit")]
    assert all(r == "existing_precontact_fz_zero_gate" for r in reasons)
    summary = dict(run_id=run_id, path=str(path), checkpoint=meta["checkpoint"]["path"],
                   force_observation=meta["force_observation"], origin_xy_mm=origin.tolist(),
                   run_duration_s=(end - event("run_start")[0]["receipt_monotonic_ns"]) / 1e9,
                   policy_duration_s=(end - t0) / 1e9, first_plan_receipt_ns=t0,
                   inference_count=len(latency), inference_latency_ms=latency,
                   warm_inference_median_ms=float(np.median(latency[1:])),
                   service_responses=len(event("service_response")),
                   service_failures=sum(not e["details"]["success"] for e in event("service_response")),
                   precontact_force_zero_events=len(reasons),
                   errors=len(event("run_error")) + len(event("inference_error")) + len(event("service_error")),
                   logger_drained=status["drained"], logger_write_errors=status["write_errors"],
                   logger_drops=sum(v["dropped"] for v in status["counts"].values()),
                   counts=status["counts"], common_force=f,
                   middle_10_25s_force=force_summary(window(frames["wrench"], 10., 25.)),
                   plateau_15_25s_force=force_summary(window(frames["wrench"], 15., 25.)),
                   common_xy_path_mm=float(np.linalg.norm(np.diff(p[["x", "y"]], axis=0), axis=1).sum()),
                   common_xyz_span_mm=(p[["x", "y", "z"]].max() - p[["x", "y", "z"]].min()).to_dict(),
                   common_z_min_mm=float(p.z.min()), final_z_mm=float(window(pose, 0., (end-t0)/1e9).z.iloc[-1]),
                   force_command_burst_9_10_5s=dict(requests=len(burst),
                       zero_requests=int((burst.fz == 0).sum()),
                       zero_nonzero_switches=int(np.sum(np.diff((burst.fz == 0).astype(int)) != 0))),
                   negative_sent_force_requests=int((sent.fz < 0).sum()),
                   commanded_fz_range_N=[float(sent.fz.min()), float(sent.fz.max())],
                   sampling=meta["sampling"], extra_telemetry_enabled=meta["extra_telemetry_enabled"])
    audit = inspect(path)
    (OUT / f"{label}_log_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    frames["meta"], frames["summary"] = meta, summary
    data[label], summaries[label] = frames, summary

a, b = [data[k]["meta"]["runtime_parameters"] for k in RUNS]
parameter_diff = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
result = dict(common_window_s=WINDOW, runs=summaries, runtime_parameter_diff=parameter_diff,
              limitations=["Different training runs/checkpoints; not a matched E1 ON/OFF comparison",
                           "OFF disables policy force observations only; force actions/controller remain active",
                           "No confirmed specimen reset/equal stain preparation; specimen_id=default in both",
                           "20Hz maximum receipt sampling cannot resolve all force peaks or contact-gate transitions",
                           "Fz is base-axis feedback, not a verified surface-normal force or pressure",
                           "Sent force is a service request, not verified controller-applied force; no tracking-error claim",
                           "No verified RPM, removal calibration, fixed image evaluation ROI or completed-task signal"])
(OUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = {"ON_baseline": "#2563eb", "OFF_E1": "#dc2626"}
labels = {"ON_baseline": "ON: Sep 10 baseline", "OFF_E1": "OFF: Sep 16 E1"}
for index, label in enumerate(RUNS):
    d, s = data[label], summaries[label]
    ax = axes[0, index]
    w = window(d["wrench"], 0., s["policy_duration_s"])
    c = window(d["sent_force"], 0., s["policy_duration_s"])
    ax.plot(w.t, w.fz, color=colors[label], lw=1.25, label="Measured base Fz")
    # Extend only for display of last sent request, not controller-applied force.
    ax.step([*c.t, s["policy_duration_s"]], [*c.fz, c.fz.iloc[-1]], where="post", color="#333333", lw=.9,
            alpha=.8, label="Last sent force request")
    peak = s["common_force"]
    ax.annotate(f"{peak['peak_N']:.1f} N", xy=(peak["peak_t_s"], peak["peak_N"]),
                xytext=(12., 98.), arrowprops={"arrowstyle": "->", "color": colors[label]})
    ax.axvspan(9., 10.5, alpha=.1, color="#f59e0b", label="Command-switch interval")
    ax.set(title=labels[label], xlabel="Seconds after first policy plan", ylabel="Fz [N]", ylim=(-12, 120), xlim=(0, 39))
    ax.legend(fontsize=8, loc="upper right")
    p = window(d["tcp_pose"])
    axes[1, 0].plot(p.rel_x, p.rel_y, color=colors[label], label=labels[label], lw=1.3)
    axes[1, 0].scatter(p.rel_x.iloc[0], p.rel_y.iloc[0], color=colors[label], marker="o", s=30)
    axes[1, 0].scatter(p.rel_x.iloc[-1], p.rel_y.iloc[-1], color=colors[label], marker="x", s=45)
    axes[1, 1].plot(p.t, p.z, color=colors[label], label=labels[label], lw=1.3)
axes[1, 0].scatter([0], [0], marker="+", color="black", s=75, label="Latched stain center")
axes[1, 0].set(title="TCP path, common 0–37.5 s (circle=start, x=end)", xlabel="X - own stain center [mm]", ylabel="Y - own stain center [mm]")
axes[1, 0].set_aspect("equal", adjustable="datalim")
axes[1, 1].set(title="Measured TCP Z, common 0–37.5 s", xlabel="Seconds after first policy plan", ylabel="Base Z [mm]")
for ax in axes.ravel():
    ax.grid(alpha=.2)
for ax in axes[1]:
    ax.legend(fontsize=8)
fig.suptitle("Force-observation OFF vs earlier ON baseline — exploratory single-run comparison", fontsize=13)
fig.text(.5, .015, "Startup PTP excluded. Receipt-time samples, max 20 Hz. Different checkpoints/stain setup; no causal or removal-rate conclusion.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .04, 1, .96))
fig.savefig(OUT / "comparison.png", dpi=160)
plt.close(fig)

on, off = summaries.values()
metric_rows = [
    ("초기 이동 제외 정책 실행 시간 [s]", on["policy_duration_s"], off["policy_duration_s"]),
    ("추론 횟수", on["inference_count"], off["inference_count"]),
    ("공통 구간 평균 Fz [N]", on["common_force"]["mean_N"], off["common_force"]["mean_N"]),
    ("공통 구간 Fz 표준편차 [N]", on["common_force"]["std_N"], off["common_force"]["std_N"]),
    ("관측 최대 Fz [N]", on["common_force"]["peak_N"], off["common_force"]["peak_N"]),
    ("Fz >= 3 N 시간 [s] (물리 접촉 판정 아님)", on["common_force"]["time_ge_3N_s"], off["common_force"]["time_ge_3N_s"]),
    ("Fz >= 3 N 구간 평균 [N]", on["common_force"]["mean_when_ge_3N"], off["common_force"]["mean_when_ge_3N"]),
    ("Fz >= 50 N 시간 [s]", on["common_force"]["time_ge_50N_s"], off["common_force"]["time_ge_50N_s"]),
    ("공통 구간 XY 이동 거리 [mm]", on["common_xy_path_mm"], off["common_xy_path_mm"]),
    ("9–10.5 s 힘 명령 0/비0 전환 횟수", on["force_command_burst_9_10_5s"]["zero_nonzero_switches"], off["force_command_burst_9_10_5s"]["zero_nonzero_switches"]),
]
pd.DataFrame(metric_rows, columns=["metric", "ON_baseline", "OFF_E1"]).to_csv(OUT / "comparison.csv", index=False)
table = "\n".join(f"| {name} | {a:.2f} | {b:.2f} |" for name, a, b in metric_rows)
report = f"""# 2026-09-20 OFF 실행 분석

OFF 실행(16:23:41–16:24:24)은 시작 위치 확인 후 추론 10회와 서비스 스트리밍을 진행했다.
서비스 응답 479건은 전부 성공이며, 정상적인 키보드 중단 후 로그가 모두 저장됐다.
이는 명령 전달/로그 저장 확인이며 물리 작업 완료 판정은 아니다.

이번 OFF는 힘 관측을 제거한 모델이다. 9D 힘 action과 힘 제어는 유지된다.
비교 ON은 9/10 기존 성공 ckpt이고, OFF는 9/16 E1 재학습 ckpt다.
실행 파라미터 차이는 ckpt_dir/use_force_observation/metrics_run_tag뿐이지만,
동일한 학습 실험의 두 조건이 아니므로 힘 관측 효과의 인과 추정은 할 수 없다.

## 동일 시간 구간의 정량 비교

각 실행의 첫 정책 plan 생성 완료를 0초로 하고 공통 0–37.5초를 사용한다.
아래 평균·표준편차·임계값 이상 시간은 측정 receipt 간격으로 가중했다.

| 지표 | ON (9/10 baseline) | OFF (9/16 E1) |
|---|---:|---:|
{table}

![비교 그래프](comparison.png)

## 해석

- OFF는 ON보다 공통 구간 평균 Fz가 높고, Fz >= 3 N인 시간은 짧았다.
  임계값은 분석용이며, 의도적인 이탈/가공 구간이 구분되지 않아 성공률이나 접촉 실패율로 해석하지 않는다.
- 15–25초의 평균/표준편차는 ON {on['plateau_15_25s_force']['mean_N']:.2f} ± {on['plateau_15_25s_force']['std_N']:.2f} N,
  OFF {off['plateau_15_25s_force']['mean_N']:.2f} ± {off['plateau_15_25s_force']['std_N']:.2f} N이었다.
  이 구간은 OFF의 변동이 더 작아, OFF가 전 구간에서 더 불안정했다고 결론낼 수 없다.
- 두 조건 모두 첫 plan 후 약 7.5초에 100 N 이상의 큰 Fz를 기록했다.
  해당 시점의 마지막 전송 Fz 요청은 ON {on['common_force']['last_sent_at_peak_N']:.2f} N,
  OFF {off['common_force']['last_sent_at_peak_N']:.2f} N이었다. 따라서 정책이 100 N 목표를 보냈다는 뜻은 아니다.
  초기 진입 궤적/힘 좌표·부호/실제 제어기 적용값을 확인할 대상이다. 원인은 이 로그만으로 확정하지 않는다.
- OFF의 9–10.5초에는 힘 요청 31건에서 0과 약 23–24 N 사이 전환이 30회 있었다.
  0 N 요청의 이유는 기존 접촉 판정에 따른 force-zero gate였다. ON의 같은 구간 전환은 5회다.
  20Hz 저장은 이보다 빠른 센서 변화를 모두 보존하지 않아 접촉 판정 변화의 원인을 재구성할 수 없다.
- OFF의 safety_limit 이벤트 21건은 전부 이 gate 기록이다. 확인된 hard safety stop으로 세면 안 된다.
- 초기 위치의 이동은 분석에서 제외했다. XY는 각 실행의 얼룩 중심을 뺀 상대 좌표로 표시했으며,
  원점 차이 약 {np.linalg.norm(np.array(off['origin_xy_mm']) - on['origin_xy_mm']):.2f} mm를 결과 편차로 오인하지 않는다.

## 기록 및 해석 한계

- OFF 위치 759행, 힘 759행, commands 1,961행 저장. writer 오류/큐 드롭/종료 대기 모두 0.
  의도한 20Hz 샘플링 생략은 큐 손실과 구별하며 DDS 손실은 측정하지 않았다.
- force command는 commands.csv의 PTP9D_STREAM_SET_FORCE만 사용했다.
  PTP9D_STREAM_APPEND의 force 열 0은 사용되지 않는 자리이며, legacy.csv에는 이 0이 섞여 있어
  그 열 전체로 힘 추종 오차를 계산하면 잘못된 결과가 된다. 실제 제어기 적용 힘은 별도 미검증이다.
- Fz는 base 좌표 축 성분이다. 표면 법선/부호/접촉 면적이 미검증이라 압력이나 실제 법선력으로 바꾸지 않는다.
- 최종 이미지에 검은 자국이 남아 있다. 시작/종료 카메라 자세가 달라 정렬된 제거 면적률은 계산하지 않았다.
  시편 ID는 양쪽 모두 default이며 시편 재준비 여부는 로그에 없다.
- 자동 removal heatmap은 미보정 k=1과 TCP 속도를 사용한 상대 지표다.
  회전 공구 RPM/실제 재료 제거량을 측정하지 않았고 각 그림의 격자 영역도 달라 실제 제거율 비교에 쓰지 않았다.
- ON 1회와 OFF 1회이므로 재현성/통계적 우열을 주장할 수 없다.

## 원본

- ON: `{on['path']}`
- OFF: `{off['path']}`
- OFF 최종 이미지: `{off['path']}/snapshots/final_image.png`
- OFF 영상: `/home/eunseop/Videos/Screencasts/polishing_inference_20260920_162346_FLOW_IL_minus_F_E1_off.webm`
  (별도 ffprobe 검사: 10fps, 385 frames, 38.5 s, 오류 없이 파일 마무리됨)

다음 비교는 대응하는 9/16 E1 ON ckpt, 동일 시편 준비/시작 상태/평가 구간으로 반복해야 한다.
분석 재실행: `python3 reports/20260920_force_observation_comparison/analyze.py`
"""
(OUT / "report.md").write_text(report)
print(json.dumps({k: {"policy_duration_s": v["policy_duration_s"], "common_force": v["common_force"],
                         "burst": v["force_command_burst_9_10_5s"]} for k, v in summaries.items()}, indent=2))
print("Saved:", OUT / "report.md", OUT / "comparison.png")
