"""Render two Korean 16:9 presentation figures from the verified result CSVs.

This is a data-table renderer; it never connects to ROS or changes trial data.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nrs_ppt_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/table/20260920"
OUT = DATA / "slides"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1920, 1080
REG = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
BOLD = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"
INK = "#17243A"
MUTED = "#536176"
BLUE = "#2558C6"
TEAL = "#007C83"
PURPLE = "#6B4CC7"
BG = "#F5F7FB"
LINE = "#DCE3ED"
AMBER = "#89520D"


def rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


e1 = rows("E1_results.csv")
e2 = rows("E2_results.csv")
assert [r["method"] for r in e1] == ["B", "C"]
assert [(r["method"], r["attempt"]) for r in e2] == [("R", "1"), ("T", "1"), ("C", "1"), ("C", "2")]


def num(row, key):
    return f"{float(row[key]):.2f}"


class Slide:
    def __init__(self):
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(H, 0))
        self.ax.axis("off")
        self.texts = []
        self.box(0, 0, W, 8, INK, radius=0)

    def box(self, x, y, w, h, fill="white", edge=None, radius=14):
        if radius:
            p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge or fill, linewidth=.8)
        else:
            p = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge or fill, linewidth=.8)
        self.ax.add_patch(p)

    def text(self, x, y, text, size=26, color=INK, bold=False, ha="left", maxw=None):
        artist = self.ax.text(x, y, text, va="top", ha=ha, color=color,
            fontproperties=FontProperties(fname=BOLD if bold else REG, size=size*.72), linespacing=1.25)
        self.texts.append((artist, maxw))
        return artist

    def line(self, x0, y0, x1, y1, color=LINE, width=.8):
        self.ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, solid_capstyle="butt")

    def header(self, index, title, subtitle):
        self.text(80, 43, "POLISHING  /  EXPERIMENT " + index, 22, BLUE, True, maxw=1000)
        self.text(1840, 43, "90° 데이터  ·  2026.09.20", 22, MUTED, ha="right", maxw=600)
        self.text(80, 96, title, 52, bold=True, maxw=1760)
        self.text(80, 166, subtitle, 27, MUTED, maxw=1760)

    def badge(self, x, y, letter, color, size=68):
        self.box(x, y, size, size, color, radius=14)
        self.text(x+size/2, y+12, letter, 40, "white", True, "center", size-8)

    def table(self, x, y, widths, headers, values, accents, header_h=74, row_h=67, font=30):
        width = sum(widths)
        self.box(x, y, width, header_h+row_h*len(values), "white", LINE)
        self.box(x, y, width, header_h, "#EAF0F8", radius=0)
        left=x
        for w, header in zip(widths, headers):
            self.text(left+w/2, y+14, header, 23, INK, True, "center", w-16)
            left+=w
        for i, row in enumerate(values):
            yy=y+header_h+i*row_h
            if i % 2:
                self.box(x+1, yy, width-2, row_h, "#F9FAFC", radius=0)
            self.line(x, yy, x+width, yy)
            left=x
            for j,(w,value) in enumerate(zip(widths,row)):
                color=accents[i] if j==0 else INK
                size=font if j!=len(row)-1 or width<1700 else font
                # Outcome phrases need slightly smaller type than numbers.
                if len(headers)==7 and j==6:
                    size=23
                    color=AMBER if i in (0,2,3) else MUTED
                self.text(left+w/2, yy+(row_h-size)/2-1, value, size, color, j==0, "center", w-16)
                left+=w

    def save(self, filename):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for artist,maxw in self.texts:
            bbox=artist.get_window_extent(renderer)
            if maxw is not None:
                assert bbox.width <= maxw*2+2, (filename, artist.get_text(), bbox.width/2, maxw)
            assert bbox.x0>=-1 and bbox.x1<=W*2+1 and bbox.y0>=-1 and bbox.y1<=H*2+1, artist.get_text()
        self.fig.savefig(OUT/filename, dpi=200, facecolor=BG)
        plt.close(self.fig)
        with Image.open(OUT/filename) as im:
            assert im.size == (3840,2160)
            im.verify()


s=Slide()
s.header("E1", "E1 | 측정 힘 입력의 효과", "연구 질문: 목표 힘을 생성하는 IL 정책에 현재 측정 힘을 입력하면 도움이 되는가?")
for x,letter,title,color,detail in [
    (80,"B","Force-obs OFF",BLUE,"측정 힘 입력 X   ·   목표 힘 학습·출력 O"),
    (980,"C","Force-obs ON",TEAL,"측정 힘 입력 O   ·   목표 힘 학습·출력 O")]:
    s.box(x,226,860,170,"white",LINE)
    s.badge(x+26,251,letter,color)
    s.text(x+117,250,title,34,color,True,maxw=705)
    s.text(x+117,302,detail,27,maxw=705)
    s.text(x+117,346,"9/16 학습 체크포인트  ·  선택된 실행 1회",23,MUTED,maxw=705)
s.text(92,419,"공통: 영상·위치 입력, 동작·목표 힘 출력, 하위 힘 제어 유지  ·  IL = 모방학습",26,MUTED,maxw=1290)
s.box(1410,414,430,50,"#E9EDF4",radius=10)
s.text(1625,426,"A: 동작만 학습 · 보류",23,MUTED,True,"center",405)
s.text(82,480,"결과  |  동일한 0–37.5 s 관측 구간",26,bold=True,maxw=1700)
v1=[]
for r in e1:
    v1.append([r["method"],"1",num(r,"mean_fz_N")+" ± "+num(r,"temporal_sd_fz_N"),
        num(r,"max_fz_N"),num(r,"time_fz_ge_3N_s"),num(r,"time_fz_ge_50N_s"),
        num(r,"xy_path_mm"),num(r,"warm_inference_median_ms")])
s.table(80,526,[150,100,350,230,210,210,255,255],
    ["조건","n","평균 Fz ± SD\n[N]","최대 Fz\n[N]","τ3\n[s]","τ50\n[s]","XY 거리\n[mm]","추론 지연\n[ms]"],
    v1,[BLUE,TEAL],header_h=76,row_h=78,font=32)
s.box(80,786,1760,170,"white",LINE)
s.text(104,805,"기호 읽기",24,bold=True,maxw=1600)
s.text(104,844,"Fz = 측정된 로봇 base Z축 힘  ·  SD = 한 실행 내 시간 가중 표준편차  ·  n = 실행 횟수",24,MUTED,maxw=1710)
s.text(104,882,"τ3 / τ50 = Fz가 각각 3 / 50 N 이상인 누적 시간  ·  XY 거리 = 측정 TCP의 평면 이동 거리",24,MUTED,maxw=1710)
s.text(104,920,"추론 지연 = 첫 회 제외 중앙값  ·  분석 구간 = 첫 정책 plan 이후 0–37.5 s, 접근 포함",24,MUTED,maxw=1710)
s.box(80,978,1760,61,"#EAF0FA",radius=10)
s.text(104,996,"해석: 조건별 1회이므로 통계적 우열을 판단하지 않음. 실측 제거율·물리 작업 성공률은 미평가.",25,INK,True,maxw=1710)
s.text(80,1053,"※ Fz는 검증된 표면 법선력이 아니며, 최대 20 Hz 기록으로 모든 고주파 피크를 포착하지 못함.",18,MUTED,maxw=1760)
s.save("E1_ppt_summary.png")

s=Slide()
s.header("E2", "E2 | 규칙·교시 재생·IL 실행 비교", "연구 질문: 동작과 목표 힘을 정하는 방식에 따라 실행 결과가 어떻게 달라지는가?")
cards=[
    (80,"R","Rule · 규칙 기반",BLUE,["편도 규칙 경로 + 가공 목표 힘 18 N","정책 추론 없이 사전 규칙으로 명령 생성"]),
    (675,"T","Teacher replay · 교시 재생",PURPLE,["episode_29의 위치·힘·시간을 재생","원본 교시 길이 14.91 s"]),
    (1270,"C","Force-obs ON · IL",TEAL,["영상·위치·측정 힘 → 동작·목표 힘","9/16 ON 모방학습 정책 · 2회 실행"])]
for x,letter,title,color,lines in cards:
    s.box(x,226,570,189,"white",LINE)
    s.badge(x+22,247,letter,color,60)
    s.text(x+98,249,title,29,color,True,maxw=452)
    s.text(x+24,326,lines[0],25,maxw=522)
    s.text(x+24,368,lines[1],23,MUTED,maxw=522)
s.text(84,442,"공통: 90° 데이터 · 하위 힘 제어 유지   |   편도 R·T 각 1회, C 두 번을 개별 표시",25,MUTED,maxw=1750)
outcomes=["편도 종료 · 해제 확인 시간 초과","재생 종료 · 목표 도달/해제 미확인","진동 보고 · 수동 중단, 정지 확인","진동 보고 · Force 모드 이상, 정지 미확인"]
v2=[]
for r,outcome in zip(e2,outcomes):
    label=r["method"] if r["method"]!="C" else "C "+r["attempt"]+"차"
    v2.append([label,num(r,"tracking_duration_s"),num(r,"mean_fz_N")+" ± "+num(r,"temporal_sd_fz_N"),
        num(r,"max_fz_N"),num(r,"time_fz_ge_50N_s"),num(r,"xy_path_mm"),outcome])
s.table(80,490,[140,160,300,170,150,180,660],
    ["방법 / 시도","추종 시간\n[s]","평균 Fz ± SD\n[N]","최대 Fz\n[N]","τ50\n[s]","XY 거리\n[mm]","기록된 종료 상태"],
    v2,[BLUE,PURPLE,TEAL,TEAL],header_h=70,row_h=62,font=29)
s.box(80,831,1760,133,"white",LINE)
s.text(104,848,"Fz = 로봇 base Z축의 측정 힘  ·  SD = 한 실행 내 시간 가중 표준편차 (반복 간 오차 아님)",23,MUTED,maxw=1710)
s.text(104,885,"τ50 = Fz ≥ 50 N 누적 시간  ·  XY 거리 = 측정 TCP의 평면 이동 거리  ·  힘 기록 최대 20 Hz",23,MUTED,maxw=1710)
s.text(104,922,"추종 시간 = 실행 시작~첫 정지 요청, 접근 포함·R 복귀 제외  ·  R은 교시 시작 자세 복귀를 완료하지 못함",23,MUTED,maxw=1710)
s.box(80,985,1760,56,"#FFF1DC",radius=10)
s.text(104,1001,"해석: 관측 시간·종료 상태가 달라 성능 순위는 보류. 실측 제거율·작업 성공률은 미평가.",25,AMBER,True,maxw=1710)
s.text(80,1053,"※ E1과 E2는 실행 방식이 다름. 두 C 시도는 중단된 파일럿 기록이며, 추종 시간은 완료 작업시간이 아님.",18,MUTED,maxw=1760)
s.save("E2_ppt_summary.png")

sources={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ("E1_results.csv","E2_results.csv")}
(OUT/"slides_manifest.json").write_text(json.dumps(dict(resolution=[3840,2160],aspect_ratio="16:9",
    source_sha256=sources,method_order=dict(E1=["B","C"],E2=["R","T","C1","C2"]),
    numeric_display=dict(E1=v1,E2=v2),script=str(Path(__file__).resolve())),ensure_ascii=False,indent=2)+"\n")
(OUT/"README.md").write_text("""# PPT 삽입용 이미지 2장

두 파일 모두 16:9, 3840×2160 PNG입니다. 기존 논문용 원본 표는 그대로 보존합니다.

1. **E1_ppt_summary.png — 힘 observation ablation 설명과 결과**
   - B: 측정 힘 입력 OFF, 학습된 목표 힘 출력 ON.
   - C: 측정 힘 입력 ON, 학습된 목표 힘 출력 ON.
   - A는 보류이며 양쪽 모두 하위 힘 제어를 유지합니다.
   - 선택한 B/C 각 1회, 첫 plan 이후 공통 0–37.5초 수치와 Fz·SD·n·τ3·τ50·XY 거리·추론 지연의 정의를 포함합니다.

2. **E2_ppt_summary.png — R/T/C 방법 정의와 실행 기록**
   - R: 편도 규칙 경로, 가공 기준 힘 18 N.
   - T: episode_29의 위치·힘·시간 재생.
   - C: 측정 힘을 입력받고 동작·목표 힘을 출력하는 9/16 ON IL 정책.
   - R/T 각 1회와 C 두 번을 따로 표기하고, R 복귀 미완료·T 목표/접촉 해제 미확인·C 중단 상태를 표시합니다.

모든 수치는 상위 폴더의 검증된 CSV에서 읽어 소수 둘째 자리로 표시했습니다. 평균±SD는 한 실행 내 시간 변동으로, 반복 간 불확실성이나 통계적 우열을 의미하지 않습니다. 제거율·물리 작업 성공률은 미평가입니다.

PPT에서는 [삽입 → 그림]으로 PNG를 넣고 가로세로 비율을 유지해 크기를 조정하면 됩니다.
재생성: `python3 scripts/render_e1_e2_ppt_slides.py`
""")
print(json.dumps(dict(output=str(OUT),images=["E1_ppt_summary.png","E2_ppt_summary.png"],resolution=[3840,2160]),ensure_ascii=False))
