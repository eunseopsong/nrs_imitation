"""Archive today's B/C logs and videos, and select the latest completed E1 ON run as C.

Offline only: copies existing files and computes descriptive tables; no ROS nodes.
"""
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/20260920"
OLD_REPORT = ROOT / "reports/20260920_force_observation_comparison"
TZ = ZoneInfo("Asia/Seoul")
WINDOW = 37.5
OUT.mkdir(parents=True, exist_ok=True)
copied = []


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    sha = digest(src)
    if dst.exists():
        if digest(dst) != sha:
            raise RuntimeError(f"Refusing to overwrite different archived file: {dst}")
    else:
        shutil.copy2(src, dst)
    assert digest(dst) == sha
    copied.append(dict(source=str(src), destination=str(dst.relative_to(OUT)),
                       bytes=dst.stat().st_size, sha256=sha))


def copy_dir(src, dst):
    for f in sorted(src.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            copy_file(f, dst / f.relative_to(src))


def when(ns):
    return datetime.fromtimestamp(ns / 1e9, TZ)


runs = []
for p in sorted((ROOT / "logs/inference_metrics").glob("*20260920T*")):
    meta = json.loads((p / "metadata.json").read_text())
    ckpt = meta["checkpoint"]["path"]
    if "e1_force_observation_20260916" not in ckpt and "20260910_90deg_rel_single" not in ckpt:
        continue
    events = [json.loads(l) for l in (p / "events.jsonl").read_text().splitlines()]
    summary = json.loads((p / "summary.json").read_text())
    start = next(e for e in events if e["event"] == "run_start")
    end = next(e for e in reversed(events) if e["event"] == "run_end")
    plans = [e for e in events if e["event"] == "inference_end"]
    condition = "B" if meta["force_observation"] == "OFF" else "C"
    runs.append(dict(run_id=p.name, source=p, meta=meta, events=events, summary=summary,
                     start=when(start["receipt_ros_ns"]), end=when(end["receipt_ros_ns"]),
                     plans=plans, condition=condition, e1="e1_force_observation_20260916" in ckpt))

selected = {condition: max((r for r in runs if r["condition"] == condition and r["e1"] and r["plans"]),
                           key=lambda r: r["start"]) for condition in ("B", "C")}
assert selected["B"]["run_id"] == "IL_minus_F_E1_off_20260920T162341_1789889021414718505_kkmoieb_"
assert selected["C"]["run_id"] == "C_force_obs_ON_E1_20260916_20260920T171726_1789892246083511592_ua8reqys"
videos = list((Path.home() / "Videos/Screencasts").glob("polishing_inference_20260920_*.webm"))
video_owners = set()
index = []
for r in runs:
    if not r["plans"]:
        category = "archive/incomplete_runs"
    elif not r["e1"]:
        category = "archive/C_baseline_20260910_ckpt"
    else:
        category = f"{r['condition']}_force_obs_{'OFF' if r['condition']=='B' else 'ON'}"
    folder = OUT / category / r["run_id"]
    r["folder"] = folder
    copy_dir(r["source"], folder / "logs")
    matched = []
    for video in videos:
        match = re.match(r"polishing_inference_(\d{8}_\d{6})_FLOW_(.*)\.webm$", video.name)
        if not match or match[2] != r["meta"]["runtime_parameters"]["metrics_run_tag"]:
            continue
        created = datetime.strptime(match[1], "%Y%m%d_%H%M%S").replace(tzinfo=TZ)
        if r["start"].timestamp() - 1 <= created.timestamp() <= r["end"].timestamp() + 1:
            assert video not in video_owners
            video_owners.add(video)
            dest = folder / "video" / video.name
            copy_file(video, dest)
            probe = subprocess.run(["ffprobe", "-v", "warning", "-count_frames", "-show_entries",
                "stream=nb_read_frames,r_frame_rate:format=duration,size", "-of", "json", str(dest)],
                capture_output=True, text=True)
            result = dict(file=str(dest.relative_to(OUT)), returncode=probe.returncode,
                          warnings=probe.stderr.strip(), probe=json.loads(probe.stdout))
            matched.append(result)
    r["videos"] = matched
    for removal in (ROOT / "logs/polishing_removal").glob(r["meta"]["runtime_parameters"]["metrics_run_tag"] + "_20260920_*"):
        stamp = datetime.strptime(removal.name[-15:], "%Y%m%d_%H%M%S").replace(tzinfo=TZ)
        if abs((stamp - r["end"]).total_seconds()) <= 10:
            copy_dir(removal, folder / "removal")
    identity = dict(condition=r["condition"], selected_for_comparison=r is selected.get(r["condition"]),
                    checkpoint=r["meta"]["checkpoint"]["path"], source_run_id=r["run_id"],
                    start=r["start"].isoformat(), end=r["end"].isoformat(),
                    original_log_directory=str(r["source"]), video_checks=matched)
    (folder / "archive_identity.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False))
    index.append(dict(condition=r["condition"], selected=identity["selected_for_comparison"],
                      checkpoint_family="E1_20260916" if r["e1"] else "baseline_20260910",
                      start=identity["start"], end=identity["end"], inference_count=len(r["plans"]),
                      status="inference_executed" if r["plans"] else "no_inference_alignment_failed",
                      video_count=len(matched), video_status="; ".join("complete" if not v["warnings"] and v["returncode"]==0 else "incomplete_or_warning" for v in matched) or "not_created",
                      run_id=r["run_id"], directory=str(folder.relative_to(OUT))))
assert video_owners == set(videos), "Unmatched video; preserve and identify before declaring archive complete"
copy_dir(OLD_REPORT, OUT / "archive/previous_BC_comparison")
with (OUT / "runs_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)


def window(d, lo=0., hi=WINDOW):
    return d[(d.t >= lo) & (d.t <= hi)].copy()


def force_summary(d):
    t, f = d.t.to_numpy(), d.fz.to_numpy()
    dt, v = np.diff(t), f[:-1]
    valid = (dt > 0) & (dt <= .2) & np.isfinite(v)
    weights, vals = dt[valid], v[valid]
    mean = float(np.average(vals, weights=weights))
    result = dict(rows=len(d), covered_s=float(weights.sum()), excluded_gap_s=float(dt[~valid].sum()),
                  mean_N=mean, std_N=float(np.sqrt(np.average((vals-mean)**2, weights=weights))),
                  peak_N=float(f.max()), peak_t_s=float(t[f.argmax()]))
    for threshold in (3, 50):
        mask = vals >= threshold
        result[f"time_ge_{threshold}N_s"] = float(weights[mask].sum())
        result[f"mean_when_ge_{threshold}N"] = float(np.average(vals[mask], weights=weights[mask])) if mask.any() else None
    return result


summaries, frames = {}, {}
for label, r in selected.items():
    event = lambda name: [e for e in r["events"] if e["event"] == name]
    t0 = r["plans"][0]["receipt_monotonic_ns"]
    end = event("run_interrupted")[0]["receipt_monotonic_ns"]
    frame = {}
    for name in ("wrench", "tcp_pose", "commands"):
        d = pd.read_csv(r["folder"] / "logs" / (name + ".csv"))
        d["t"] = (d.receipt_monotonic_ns - t0) / 1e9
        frame[name] = d
    commands = frame["commands"]
    sent = commands[(commands.command_stage=="node_sent") & (commands.command_mode=="PTP9D_STREAM_SET_FORCE")]
    frame["sent_force"] = sent
    f = force_summary(window(frame["wrench"]))
    peak_command = sent[sent.t <= f["peak_t_s"]].iloc[-1]
    f["last_sent_at_peak_N"] = float(peak_command.fz)
    burst = window(sent, 9., 10.5)
    roi = json.loads((r["folder"] / "logs/roi.json").read_text())
    origin = np.array(roi["reference"]["center_xy_mm"])
    pose = frame["tcp_pose"]
    pose[["rel_x", "rel_y"]] = pose[["x", "y"]].to_numpy() - origin
    starts = {e["details"]["inference_id"]: e["receipt_monotonic_ns"] for e in event("inference_start")}
    latency = [(e["receipt_monotonic_ns"]-starts[e["details"]["inference_id"]])/1e6 for e in r["plans"]]
    status = r["summary"]
    s = dict(run_id=r["run_id"], path=str(r["folder"] / "logs"), original_path=str(r["source"]),
             condition=label, checkpoint=r["meta"]["checkpoint"]["path"],
             time_range=f"{r['start']:%H:%M:%S}–{r['end']:%H:%M:%S}",
             policy_duration_s=(end-t0)/1e9, inference_count=len(r["plans"]),
             service_responses=len(event("service_response")), service_failures=sum(not e["details"]["success"] for e in event("service_response")),
             warm_inference_median_ms=float(np.median(latency[1:])),
             common_force=f, plateau_15_25s_force=force_summary(window(frame["wrench"],15.,25.)),
             common_xy_path_mm=float(np.linalg.norm(np.diff(window(pose)[["x","y"]],axis=0),axis=1).sum()),
             force_command_burst_9_10_5s=dict(requests=len(burst), zero_nonzero_switches=int(np.sum(np.diff((burst.fz==0).astype(int))!=0))),
             counts=status["counts"], logger_write_errors=status["write_errors"], logger_drained=status["drained"],
             logger_drops=sum(v["dropped"] for v in status["counts"].values()), videos=r["videos"])
    assert s["policy_duration_s"] > WINDOW and s["logger_drained"]
    summaries[label], frames[label] = s, frame
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_inference_log import inspect
    audit = inspect(r["folder"] / "logs")
    assert not (audit["issues"] or audit["run_id_mismatches"] or audit["malformed_event_lines"])
    (r["folder"] / "log_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

B, C = summaries["B"], summaries["C"]
params = [selected[k]["meta"]["runtime_parameters"] for k in ("B", "C")]
diff = {k:[params[0].get(k),params[1].get(k)] for k in params[0].keys() | params[1].keys() if params[0].get(k)!=params[1].get(k)}
assert set(diff) == {"ckpt_dir", "use_force_observation", "metrics_run_tag"}
(OUT / "comparison.json").write_text(json.dumps(dict(common_window_s=WINDOW, runs=summaries, runtime_parameter_diff=diff), indent=2, ensure_ascii=False))
(OUT / "selected_runs.json").write_text(json.dumps({k:dict(run_id=r["run_id"], logs=str((r["folder"] / "logs").relative_to(OUT)), videos=r["videos"], checkpoint=r["meta"]["checkpoint"]["path"]) for k,r in selected.items()}, indent=2, ensure_ascii=False))

columns = ["분류", "지표", "단위", "B — Force-obs OFF (9/16 E1)", "C — Force-obs ON (9/16 E1, 최신)", "집계 범위 / 설명"]
rows = []
def add(group, name, unit, getter, note):
    rows.append([group,name,unit,getter(B),getter(C),note])
add("실행","실행 시각 (2026-09-20 KST)","시각",lambda s:s["time_range"],"현재 선택된 B와 가장 최근 C")
for name,unit,key in [("정책 실행 시간","s","policy_duration_s"),("추론 횟수","회","inference_count"),("서비스 응답 수","건","service_responses"),("서비스 실패 응답","건","service_failures"),("첫 회 제외 추론 시간 중앙값","ms","warm_inference_median_ms")]:
    add("실행",name,unit,lambda s,k=key:s[k],"전체 실행; 서비스 성공은 물리 작업 완료 판정이 아님")
for name,unit,key in [("평균 측정 Fz","N","mean_N"),("측정 Fz 표준편차","N","std_N"),("관측 최대 Fz","N","peak_N"),("최대 Fz 발생 시각","s","peak_t_s"),("Fz ≥ 3 N인 시간","s","time_ge_3N_s"),("Fz ≥ 3 N 구간 평균 Fz","N","mean_when_ge_3N"),("Fz ≥ 50 N인 시간","s","time_ge_50N_s")]:
    add("힘",name,unit,lambda s,k=key:s["common_force"][k],"공통 0–37.5 s; receipt 간격 가중. 임계값은 접촉 성공/실패 판정 아님")
for name,key in [("15–25 s 평균 Fz","mean_N"),("15–25 s Fz 표준편차","std_N")]:
    add("힘",name,"N",lambda s,k=key:s["plateau_15_25s_force"][k],"동일한 중간 구간; receipt 간격 가중")
add("동작","XY 이동 거리","mm",lambda s:s["common_xy_path_mm"],"공통 0–37.5 s; 측정 TCP 샘플 사이 XY 거리 합")
add("힘 명령","9–10.5 s 힘 요청 수","건",lambda s:s["force_command_burst_9_10_5s"]["requests"],"PTP9D_STREAM_SET_FORCE 요청만 집계")
add("힘 명령","9–10.5 s 0/비0 전환","회",lambda s:s["force_command_burst_9_10_5s"]["zero_nonzero_switches"],"기존 접촉 판정의 force-zero gate; hard safety stop 건수가 아님")
add("로그","위치 / 힘 저장 행 수","행",lambda s:f"{s['counts']['tcp_pose']['written']} / {s['counts']['wrench']['written']}","전체 실행; 최대 20Hz 저장")
add("로그","commands 저장 행 수","행",lambda s:s["counts"]["commands"]["written"],"예측·전송 명령 포함; 서비스 요청 건수와 다름")
add("로그","로그 쓰기 오류","건",lambda s:len(s["logger_write_errors"]),"전체 실행")
add("로그","로거 큐 드롭","건",lambda s:s["logger_drops"],"의도한 샘플링 생략과 구분; DDS 손실은 미측정")
add("결과 판정","물리 작업 성공률 / 제거율","",lambda s:"미평가","완료 신호·정렬된 평가 ROI 없음")
conditions = [["측정 힘 observation","X","O","정책 입력 기준"],
              ["힘 history","정규화 후 0","측정 힘 사용","B의 현재 구현은 상수 0 입력을 유지"],
              ["목표 힘 action 학습·출력","O","O","교시 목표 힘을 학습; 위치와 함께 9D 출력"],
              ["로봇 목표 힘","정책 예측값","정책 예측값","공통 접근·이탈 실행 규칙 유지"],
              ["하위 힘 제어 / 안전 감시","유지","유지","제어기 ON/OFF 비교가 아님"],
              ["사용 ckpt","9/16 E1 OFF","9/16 E1 ON","대응하는 학습 조건의 ON/OFF ckpt"],
              ["실행 모드","service_stream","service_stream","동일"],
              ["Grad-CAM / 실시간 창","OFF / OFF","OFF / OFF","자동 녹화 유지"],
              ["측정 로그 / 추가 고주파 구독","최대 20Hz / OFF","최대 20Hz / OFF","정책 입력/제어 주기 유지"],
              ["시편 ID","default","default","시편·얼룩 재준비 여부는 기록에서 확인 불가"],
              ["A — Motion-only","실행하지 않음","실행하지 않음","별도 학습 + 외부 F0; 하위 힘 제어 유지"]]
notes = [["C 갱신",f"가장 최근 {C['time_range']} 실행을 C로 선택. 17:15의 C와 16:06의 9/10 baseline은 보존하되 이번 비교표에는 미사용."],
         ["비교 범위","9/16 E1 OFF(B)와 ON(C) 각각 1회. 실행 파라미터 차이는 ckpt/힘 observation/run tag뿐. 시편 준비·반복 실험은 미확인으로 통계적 우열을 단정하지 않음."],
         ["시간 정렬","각 실행 첫 inference_end를 0초로 설정. 비교 구간 0–37.5초; 초기 PTP 제외. 마지막 샘플 이후 외삽 없이 receipt 간격을 가중하며 0.2초 초과 간격 제외."],
         ["포함 시간",f"B {B['common_force']['covered_s']:.4f}s / C {C['common_force']['covered_s']:.4f}s. 제외 간격 B {B['common_force']['excluded_gap_s']:.4f}s / C {C['common_force']['excluded_gap_s']:.4f}s."],
         ["힘 의미","Fz는 base 좌표 축 성분. 표면 법선/부호/면적 미검증이므로 압력이나 검증된 법선력으로 해석하지 않음. 최대 20Hz 기록은 모든 고주파 피크를 포착하지 못함."],
         ["힘 명령 집계","STREAM_SET_FORCE 전송 요청만 사용. STREAM_APPEND의 force 열 0은 미사용 자리이므로 legacy CSV 전체 cmd_fz 열로 추종 오차를 계산하지 않음. 실제 제어기 적용 힘은 미검증."],
         ["제거율/성공률","전후 동일 자세 영상 ROI, 실측 제거량, 완료 신호가 없어 미평가. removal heatmap은 미보정 k=1과 TCP 속도로 계산한 상대 지표이며 실제 제거량이 아님."],
         ["원본 보존","원본 로그·영상은 수정/이동하지 않고 복사. 모든 복사 파일은 SHA-256 대조. 낮은 주파수로 의도적으로 샘플링한 생략과 로거 큐 손실을 구별."],
         ["오늘의 다른 실행","runs_index.csv에 9개 실행을 분리. 초기 정렬 실패 5회, 이전 C 1회, 기존 baseline 1회는 이번 선택 B/C와 섞지 않음."]]
for k,s in summaries.items():
    notes.extend([[f"{k} 보관 로그",s["path"]],[f"{k} 원본 로그",s["original_path"]],[f"{k} ckpt",s["checkpoint"]]])
    notes.extend([[f"{k} 보관 영상",str(OUT / v["file"])] for v in s["videos"]])
spec = dict(title="B·C 실험 결과 — 2026-09-20 (최신 C 반영)",
            subtitle="B: 9/16 E1 OFF (16:23) / C: 9/16 E1 ON (17:17, 최신). 첫 plan 이후 공통 0–37.5초 비교. 각 1회이며 통계적 우열/제거율은 미평가.",
            columns=columns,rows=rows,conditions=conditions,notes=notes)
(OUT / "table_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8))
for i,k in enumerate(("B","C")):
    frame,s=frames[k],summaries[k];color={"B":"#dc2626","C":"#2563eb"}[k]
    w=window(frame["wrench"],0.,s["policy_duration_s"]);c=window(frame["sent_force"],0.,s["policy_duration_s"])
    axes[0,i].plot(w.t,w.fz,color=color,label="Measured base Fz")
    axes[0,i].step([*c.t,s["policy_duration_s"]],[*c.fz,c.fz.iloc[-1]],where="post",color="#444444",lw=.8,label="Last sent force request")
    axes[0,i].set(title=f"{k}: Force-obs {'OFF' if k=='B' else 'ON'} (Sep 16 E1)",xlabel="Seconds after first plan",ylabel="Fz [N]",ylim=(-12,125))
    p=window(frame["tcp_pose"])
    axes[1,0].plot(p.rel_x,p.rel_y,color=color,label=k)
    axes[1,1].plot(p.t,p.z,color=color,label=k)
axes[1,0].set(title="TCP path, common 0–37.5 s",xlabel="X - stain center [mm]",ylabel="Y - stain center [mm]")
axes[1,0].set_aspect("equal",adjustable="datalim")
axes[1,1].set(title="TCP Z, common 0–37.5 s",xlabel="Seconds after first plan",ylabel="Base Z [mm]")
for ax in axes.ravel():ax.legend();ax.grid(alpha=.2)
fig.suptitle("B vs updated C — latest C run at 17:17:26")
fig.tight_layout();fig.savefig(OUT / "comparison.png",dpi=150);plt.close(fig)
table="\n".join("| "+" | ".join(f"{x:.2f}" if isinstance(x,float) else str(x) for x in r)+" |" for r in rows)
(OUT / "report.md").write_text("# B·C 결과 갱신\n\n"+spec["subtitle"]+"\n\n| "+" | ".join(columns)+" |\n|"+"---|"*6+"\n"+table+"\n\n![비교 그래프](comparison.png)\n\n"+"\n\n".join(f"**{k}**: {v}" for k,v in notes))
(OUT / "README.md").write_text(f"""# 2026-09-20 B·C 결과 모음

현재 결과표의 B: **{B['time_range']} — 9/16 E1 OFF**

현재 결과표의 C: **{C['time_range']} — 9/16 E1 ON, 가장 최근 실행**

- `BC_results_20260920.csv` / `.xlsx`: 갱신된 비교표.
- `B_force_obs_OFF/`: B 원본 CSV·메타데이터·영상·removal 그림.
- `C_force_obs_ON/`: 오늘 9/16 ON C 실행 2회. 비교 대상으로 선택된 것은 `selected_runs.json` 참조.
- `archive/C_baseline_20260910_ckpt/`: 기존 9/10 baseline 기록(현재 C 비교표에서 제외).
- `archive/incomplete_runs/`: 오늘 초기 정렬에 실패해 정책 추론을 시작하지 못한 5회 기록.
- `archive/previous_BC_comparison/`: 갱신 전 B vs 9/10 baseline 표와 분석 보존.
- `runs_index.csv`: 오늘 9개 실행과 선택 여부, 영상 상태.
- `selected_runs.json`: 현재 B/C 원본 경로와 ckpt 및 영상 검사 결과.
- `manifest.json`: 복사 원본·대상 경로, 크기, SHA-256.
- `report.md`, `comparison.json`, `comparison.png`: 최신 비교의 수치·방법·그래프.

영상은 총 4개를 보존했다. B, 최신 C, 17:15 C 영상은 정상 마무리됐다.
기존 16:06 baseline 영상은 원본의 파일 끝부분 불완전 상태를 그대로 보존하고 표시했다.
원본 로그·영상은 그대로 두고 복사했으며, A 실행 결과는 없다.

공통 비교 구간은 첫 plan 이후 0–37.5초, 로그는 최대 20Hz다.
표는 기술적 단일 실행 비교이며 실제 제거율·작업 성공률·통계적 우열을 주장하지 않는다.
""")
(OUT / "manifest.json").write_text(json.dumps(dict(date="20260920",selected={k:r["run_id"] for k,r in selected.items()},copied_file_count=len(copied),copied_bytes=sum(f["bytes"] for f in copied),files=copied),indent=2,ensure_ascii=False))
print(json.dumps(dict(selected={k:s["run_id"] for k,s in summaries.items()},runs=len(runs),videos=len(video_owners),files=len(copied),summaries={k:{"mean_Fz":s["common_force"]["mean_N"],"peak_Fz":s["common_force"]["peak_N"],"plans":s["inference_count"]} for k,s in summaries.items()}),indent=2))
