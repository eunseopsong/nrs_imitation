"""Build compact, typeset manuscript tables and two 600-dpi PNGs from CSV.

The table bodies can be inserted into an IEEEtran two-column manuscript.
Standalone previews use newtx serif fonts; no ROS or trial data is modified.
"""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/table/20260920"
OUT = DATA / "ral"
BUILD = OUT / "build"
BUILD.mkdir(parents=True, exist_ok=True)


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(r, k):
    return f"{float(r[k]):.2f}"


e1, e2 = read("E1_results.csv"), read("E2_results.csv")
assert [r["method"] for r in e1] == ["B", "C"]
assert [(r["method"], r["attempt"]) for r in e2] == [("R", "1"), ("T", "1"), ("C", "1"), ("C", "2")]

rows1 = [
    [r["method"], "Off" if r["method"] == "B" else "On",
     number(r,"mean_fz_N")+r" $\pm$ "+number(r,"temporal_sd_fz_N"),
     *[number(r,k) for k in ("max_fz_N","time_fz_ge_3N_s","time_fz_ge_50N_s","xy_path_mm","warm_inference_median_ms")]]
    for r in e1]
rows2 = []
for r, outcome in zip(e2,[r"Release timeout",r"Trajectory end$^{a}$",r"Manual abort",r"Stop unverified$^{b}$"]):
    label = r["method"] if r["method"] != "C" else "C"+r["attempt"]
    rows2.append([label,number(r,"tracking_duration_s"),
        number(r,"mean_fz_N")+r" $\pm$ "+number(r,"temporal_sd_fz_N"),
        *[number(r,k) for k in ("max_fz_N","time_fz_ge_50N_s","xy_path_mm")],outcome])

tables = [dict(
    name="E1_RAL", roman="I", label="tab:e1_force_observation",
    caption=r"E1: Force-observation ablation on the $90^\circ$ polishing task",
    columns="lccccccc",
    headers=[r"Method", r"Force obs.", r"\shortstack{$\overline{F}_z \pm \sigma_t$\\(N)}",
             r"\shortstack{$F_{z,\max}$\\(N)}", r"\shortstack{Time with\\$F_z\geq3$ N\\$\tau_3$ (s)}",
             r"\shortstack{Time with\\$F_z\geq50$ N\\$\tau_{50}$ (s)}", r"\shortstack{$L_{XY}$\\(mm)}",
             r"\shortstack{$t_{\mathrm{inf}}$\\(ms)}"],
    rows=rows1,
    notes=[
        r"\textit{Threshold durations:} $\tau_3$ is the cumulative time with measured $F_z\geq3$ N; $\tau_{50}$ is the cumulative time with measured $F_z\geq50$ N, within the 37.5-s analysis window. \textbf{Tau denotes time in seconds, not torque.}",
        r"\textit{Methods:} B/C denote imitation learning (IL) without/with measured-force observations. Both learn motion and target-force actions and retain low-level force control. A (motion-only IL) is deferred.",
        r"\textit{Metrics:} $F_z$ is filtered robot-base Z-axis force; $\overline{F}_z$ and $\sigma_t$ are its time-weighted mean and within-run temporal standard deviation. $F_{z,\max}$ is the largest logged value. $L_{XY}$ is measured TCP planar path length. $t_{\mathrm{inf}}$ is median inference latency excluding the first call.",
        r"\textit{Scope:} $n=1$ selected run per condition, using the first 37.5 s after the initial policy plan, including approach. Force/pose logging is at most 20 Hz. $\sigma_t$ is not across-trial uncertainty. Material removal and physical task success are unassessed; no statistical superiority is claimed."
    ]),dict(
    name="E2_RAL", roman="II", label="tab:e2_rule_replay_il",
    caption=r"E2: Rule-based control, teacher replay, and IL---pilot execution results",
    columns="lcccccl",
    headers=[r"Method",r"\shortstack{$t_{\mathrm{tr}}$\\(s)}",
             r"\shortstack{$\overline{F}_z \pm \sigma_t$\\(N)}",r"\shortstack{$F_{z,\max}$\\(N)}",
             r"\shortstack{Time with\\$F_z\geq50$ N\\$\tau_{50}$ (s)}",r"\shortstack{$L_{XY}$\\(mm)}",r"Recorded outcome"],
    rows=rows2,
    notes=[
        r"\textit{Threshold duration:} $\tau_{50}$ is the cumulative time with measured $F_z\geq50$ N within each recorded tracking interval. \textbf{Tau denotes time in seconds, not torque.}",
        r"\textit{Methods:} R is a one-way rule-based path with an 18 N processing-force reference. T replays the original pose, force, and timing of episode 29 (14.91 s). C1/C2 are two attempts of IL with measured-force observations and learned target-force actions (September 16 checkpoint). All retain low-level force control.",
        r"\textit{Metrics:} $t_{\mathrm{tr}}$ spans tracking start to the first stop request, including approach and excluding R's subsequent return. $F_z$ is filtered robot-base Z-axis force; $\overline{F}_z\pm\sigma_t$ is its time-weighted mean $\pm$ within-run temporal SD, and $F_{z,\max}$ is its logged maximum. $L_{XY}$ is measured TCP planar path length. Logging is at most 20 Hz.",
        r"\textit{Outcomes:} R's release verification timed out before return to the demonstration start; hold was subsequently verified. $^{a}$T's replay ended and hold was verified, but final-target arrival and contact release were not verified. C1 was manually aborted with hold verified. $^{b}$C2 lost Force mode or fresh mode feedback; STOP acknowledgement and physical hold were unverified. Vibration was reported in both C attempts.",
        r"\textit{Scope:} One pilot attempt per row; unequal, interrupted observation windows are not completed cycle times. E2 uses a different execution pipeline from E1. No performance ranking, material-removal result, or physical task-success rate is established."
    ])]


def body(t):
    return "\n".join([
        r"\begingroup",r"\fontsize{8.5}{10.5}\selectfont",
        r"\setlength{\tabcolsep}{3.5pt}",r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}"+t["columns"]+r"@{}}",
        r"\toprule", " & ".join(t["headers"])+r" \\",r"\midrule",
        *(" & ".join(row)+r" \\" for row in t["rows"]),
        r"\bottomrule",r"\end{tabular*}\par",r"\endgroup",
        r"\vspace{4pt}",r"\begingroup\fontsize{7.5}{9.3}\selectfont\raggedright",
        *(note+r"\par\vspace{2pt}" for note in t["notes"]),r"\endgroup"
    ])


preamble = r"""\documentclass[border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs,array}
\setlength{\parindent}{0pt}
\begin{document}
\begin{minipage}{7.16in}
"""
insert = [r"% Insert in an IEEEtran two-column manuscript; requires \usepackage{booktabs,array,amsmath}.",
          "% Source data: results/table/20260920/E1_results.csv and E2_results.csv.",
          "% Each standard deviation describes temporal variation within one run."]
validation = []
for t in tables:
    table_body=body(t)
    caption=(r"{\centering\fontsize{8}{10}\selectfont TABLE "+t["roman"]+"\n"+
             r"\par\vspace{2pt}{\scshape "+t["caption"]+r"}\par}"+"\n"+r"\vspace{5pt}")
    source = OUT/(t["name"]+".tex")
    source.write_text(preamble+caption+"\n"+table_body+"\n"+r"\end{minipage}"+"\n"+r"\end{document}"+"\n")
    result=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",
        "-output-directory="+str(BUILD),str(source)],capture_output=True,text=True,cwd=OUT)
    (BUILD/(t["name"]+"_compile.txt")).write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError("LaTeX failed:\n"+result.stdout[-5000:]+result.stderr)
    log=(BUILD/(t["name"]+".log")).read_text()
    assert "Overfull" not in log, "Table overflow: "+t["name"]
    assert "Missing character" not in log, "Missing glyph: "+t["name"]
    pdf=OUT/(t["name"]+".pdf")
    pdf.write_bytes((BUILD/pdf.name).read_bytes())
    subprocess.run(["pdftoppm","-r","600","-png","-singlefile",str(pdf),str(OUT/t["name"])],check=True,capture_output=True)
    extracted=subprocess.run(["pdftotext","-layout",str(pdf),"-"],check=True,capture_output=True,text=True).stdout
    assert "Methods:" in extracted and "Metrics:" in extracted and "Scope:" in extracted
    assert "Tau denotes time in seconds, not torque." in " ".join(extracted.split())
    for row in t["rows"]:
        for value in row:
            for numeric in re.findall(r"\d+\.\d{2}",value):
                assert numeric in extracted,(t["name"],numeric)
    from PIL import Image
    with Image.open(OUT/(t["name"]+".png")) as im:
        assert im.width>4200
        dimensions=list(im.size)
        im.verify()
    info=subprocess.run(["pdfinfo",str(pdf)],check=True,capture_output=True,text=True).stdout
    validation.append(dict(table=t["name"],png_dimensions=dimensions,dpi=600,
                           no_overfull_boxes=True,all_display_numbers_in_pdf=True,pdfinfo=info))
    insert += [r"\begin{table*}[t]",r"\centering",r"\caption{"+t["caption"]+"}",
               r"\label{"+t["label"]+"}",table_body,r"\end{table*}",""]
(OUT/"E1_E2_IEEE_tables.tex").write_text("\n".join(insert)+"\n")
(OUT/"validation.json").write_text(json.dumps(dict(tables=validation,
    sources={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ("E1_results.csv","E2_results.csv")},
    format_reference="https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/",
    renderer="Standalone LaTeX/newtx; IEEEtran-compatible table* insertion source supplied"),indent=2)+"\n")
(OUT/"README.md").write_text("""# RA-L 원고 삽입용 표 이미지

사용자의 형식 정정을 반영해 논문 표 형태로 다시 제작했습니다.

- **E1_RAL.png**: E1 B/C의 측정 힘 입력 비교. B/C의 뜻, 두 조건의 목표 힘 학습 및 하위 힘 제어 유지, 각 지표 정의를 각주에 포함했습니다. A는 보류입니다.
- **E2_RAL.png**: E2 R(편도 규칙 +18 N), T(episode_29 재생), C1/C2(힘 관측 ON IL 두 시도)의 실행 결과. 각 방법·지표와 종료 상태를 각주에서 설명합니다.

두 이미지는 600 dpi, 흰 배경, Times 계열 serif 서체, 상단 TABLE I/II 캡션 및 수평선으로 구성했습니다. 표는 2단 원고의 전폭 7.16 inch를 기준으로 LaTeX에서 조판했고, 표 주위로 잘랐습니다.

- 논문에는 PNG보다 `E1_RAL.pdf`, `E2_RAL.pdf` 또는 원고용 `E1_E2_IEEE_tables.tex`를 권장합니다.
- 각 `E1_RAL.tex`, `E2_RAL.tex`는 독립 렌더링용 소스입니다. 실제 원고의 표 번호는 원고용 table* 소스를 넣으면 자동으로 결정됩니다.
- 형식 안내: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
- 수치는 상위 폴더의 검증된 CSV를 그대로 사용했습니다. 표의 ±는 시간 변동이며 반복 간 오차가 아닙니다. 중단/미확인 결과를 성공으로 바꾸지 않았습니다.
- τ₃·τ₅₀ 열 머리에 힘 기준 이상의 시간임을 표시했고, 첫 각주에 각 기준값과 누적 시간의 뜻을 풀어 썼습니다. **τ는 초 단위 시간이며 토크가 아님**을 이미지 안에 명시했습니다.

재생성: `python3 scripts/render_ral_paper_tables.py`
표 숫자, PDF 문자, 넘침 여부 및 PNG 크기를 자동 확인했으며 결과는 validation.json에 저장합니다.
""")
print(json.dumps(dict(output=str(OUT),tables=validation),ensure_ascii=False,indent=2))
