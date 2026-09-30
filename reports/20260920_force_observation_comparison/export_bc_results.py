"""Export existing analysis as B/C tables, without changing the original runs."""
import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

OUT = Path(__file__).resolve().parent
analysis = json.loads((OUT / "comparison.json").read_text())
B, C = analysis["runs"]["OFF_E1"], analysis["runs"]["ON_baseline"]
columns = ["분류", "지표", "단위", "B — Force-obs OFF", "C — Force-obs ON (기존 baseline)", "집계 범위 / 설명"]
rows = []
def row(group, label, unit, b, c, note):
    rows.append([group, label, unit, b, c, note])
def values(key):
    return B[key], C[key]
def forces(key):
    return B["common_force"][key], C["common_force"][key]

row("실행", "실행 시각 (2026-09-20 KST)", "시각", "16:23:41–16:24:24", "16:06:36–16:07:19", "로그 시작부터 키보드 중단까지")
row("실행", "정책 실행 시간", "s", *values("policy_duration_s"), "첫 plan 완료부터 중단까지; 초기 PTP 제외")
row("실행", "추론 횟수", "회", *values("inference_count"), "전체 실행")
row("실행", "서비스 성공 응답", "건", B["service_responses"]-B["service_failures"], C["service_responses"]-C["service_failures"], "명령 전달 성공; 물리 작업 완료 판정은 아님")
row("실행", "서비스 실패 응답", "건", *values("service_failures"), "전체 실행")
row("실행", "첫 회 제외 추론 시간 중앙값", "ms", *values("warm_inference_median_ms"), "inference_start → inference_end 이벤트 간격")
row("힘", "평균 측정 Fz", "N", *forces("mean_N"), "공통 0–37.5 s; receipt 시간 간격 가중")
row("힘", "측정 Fz 표준편차", "N", *forces("std_N"), "공통 0–37.5 s; receipt 시간 간격 가중")
row("힘", "관측 최대 Fz", "N", *forces("peak_N"), "공통 0–37.5 s; 최대 20Hz 기록에서 관측한 값")
row("힘", "최대 Fz 발생 시각", "s", *forces("peak_t_s"), "각 실행 첫 plan 완료를 0 s로 설정")
row("힘", "Fz ≥ 3 N인 시간", "s", *forces("time_ge_3N_s"), "공통 구간의 분석용 임계값; 접촉 성공/실패 판정 아님")
row("힘", "Fz ≥ 3 N 구간 평균 Fz", "N", *forces("mean_when_ge_3N"), "공통 0–37.5 s; 해당 임계값 이상 구간만")
row("힘", "Fz ≥ 50 N인 시간", "s", *forces("time_ge_50N_s"), "공통 0–37.5 s; 샘플 사이 값을 유지하는 근사")
row("힘", "15–25 s 평균 Fz", "N", B["plateau_15_25s_force"]["mean_N"], C["plateau_15_25s_force"]["mean_N"], "동일한 중간 구간; receipt 시간 간격 가중")
row("힘", "15–25 s Fz 표준편차", "N", B["plateau_15_25s_force"]["std_N"], C["plateau_15_25s_force"]["std_N"], "이 구간에서는 B의 힘 수준이 높고 변동은 작음")
row("동작", "XY 이동 거리", "mm", *values("common_xy_path_mm"), "공통 0–37.5 s; 측정 TCP 위치 샘플 간 XY 거리 합")
row("힘 명령", "9–10.5 s 힘 요청 수", "건", B["force_command_burst_9_10_5s"]["requests"], C["force_command_burst_9_10_5s"]["requests"], "PTP9D_STREAM_SET_FORCE 요청만 집계")
row("힘 명령", "9–10.5 s 0/비0 전환", "회", B["force_command_burst_9_10_5s"]["zero_nonzero_switches"], C["force_command_burst_9_10_5s"]["zero_nonzero_switches"], "기존 접촉 판정에 따른 force-zero gate; B는 약 23–24 N과 0 N 사이 전환")
row("로그", "위치 / 힘 저장 행 수", "행", f"{B['counts']['tcp_pose']['written']} / {B['counts']['wrench']['written']}", f"{C['counts']['tcp_pose']['written']} / {C['counts']['wrench']['written']}", "전체 실행; 의도한 최대 20Hz 샘플링 적용")
row("로그", "commands 저장 행 수", "행", B["counts"]["commands"]["written"], C["counts"]["commands"]["written"], "정책 예측과 전송 명령을 포함; 서비스 호출 건수와 다름")
row("로그", "로그 쓰기 오류", "건", len(B["logger_write_errors"]), len(C["logger_write_errors"]), "전체 실행")
row("로그", "로거 큐 드롭", "건", *values("logger_drops"), "의도한 샘플링 생략과 구분; DDS 손실은 미측정")
row("결과 판정", "물리 작업 성공률 / 제거율", "", "미평가", "미평가", "완료 신호·정렬된 평가 ROI 없음; 자동 removal heatmap은 실제 제거량이 아님")

conditions = [
    ["측정 힘 observation", "X", "O", "정책 입력 기준"],
    ["힘 history", "정규화 후 0", "측정 힘 사용", "B의 현재 구현은 상수 0 입력을 유지"],
    ["목표 힘 action 학습·출력", "O", "O", "교시 목표 힘을 학습; 위치와 함께 9D 출력"],
    ["교시의 목표 힘을 학습 정답으로 사용", "사용", "사용", "A는 미사용이나 이번 표에 A 실행 결과는 없음"],
    ["로봇의 목표 힘", "정책 예측값", "정책 예측값", "접근·이탈 등 기존 실행 규칙 유지"],
    ["하위 힘 제어 / 안전 감시", "유지", "유지", "제어기 ON/OFF 실험이 아님"],
    ["사용 ckpt", "9/16 E1 OFF", "9/10 기존 성공 ckpt", "C는 대응하는 9/16 E1 ON ckpt 실행 결과가 아님"],
    ["실행 모드", "service_stream", "service_stream", "동일"],
    ["Grad-CAM / 실시간 영상 창", "OFF / OFF", "OFF / OFF", "자동 녹화 유지"],
    ["측정 로그 / 추가 고주파 구독", "최대 20Hz / OFF", "최대 20Hz / OFF", "정책 관측 및 제어 주기는 유지"],
    ["시편 ID", "default", "default", "동일 시편·얼룩 재준비 여부는 로그만으로 확인할 수 없음"],
    ["A — Motion-only", "실행하지 않음", "실행하지 않음", "별도 학습 + 외부 기준 힘 F0; 하위 힘 제어 유지"],
]
notes = [
    ["비교 범위", "B와 기존 C baseline 각각 1회. 서로 다른 학습 실행의 ckpt이므로, 힘 observation의 인과 효과 또는 통계적 우열을 단정하지 않는다."],
    ["정식 E1 비교", "대응하는 9/16 E1 ON ckpt로 C를 실행하고 시편 준비·시작 상태·평가 구간을 맞춰야 한다."],
    ["시간 정렬", "각 실행의 첫 inference_end를 0초로 설정. 공통 비교 구간은 0–37.5초이며 초기 PTP 이동은 제외한다."],
    ["시간 가중", "기록 receipt 시각의 연속 구간을 가중치로 사용. 마지막 샘플 밖으로 외삽하지 않으며 0.2초 초과 간격은 제외(이번 공통 구간에서 제외 시간 0초). 실제 포함 시간은 B 37.4708초, C 37.4616초."],
    ["힘 의미", "Fz는 base 좌표 축의 측정 성분. 표면 법선·부호·접촉 면적 미검증으로 압력 또는 검증된 표면 법선력으로 해석하지 않는다."],
    ["샘플링 한계", "최대 20Hz 기록이므로 고주파 피크와 빠른 접촉 판정 변화를 모두 포착하지 못한다. Fz ≥ 3 N 시간은 분석 지표이며 접촉 성공률이 아니다."],
    ["초기 큰 Fz", "두 조건 모두 약 7.5초에 100 N 이상 관측. 그때 마지막 전송 힘은 B -3.19 N, C -1.37 N이므로, 정책이 100 N 목표를 전송했다는 뜻이 아니다. 실제 적용값/부호/초기 진입 동작은 별도 확인 대상."],
    ["힘 명령 집계", "commands.csv의 PTP9D_STREAM_SET_FORCE를 사용. STREAM_APPEND의 힘 열 0은 사용되지 않는 자리이며 legacy.csv에 이 0이 섞여 있으므로 전체 cmd_fz 열로 추종 오차를 계산하지 않는다."],
    ["힘 명령 전환", "9–10.5초 B의 0/비0 전환 30회, C 5회. 0 요청은 기존 접촉 판정의 force-zero gate로 표시됨. hard safety stop 건수로 해석하지 않는다."],
    ["중간 구간", "15–25초 B는 23.96 ± 2.61 N, C는 18.17 ± 7.00 N. B가 모든 구간에서 더 불안정하다는 결론은 맞지 않는다."],
    ["얼룩 제거", "최종 이미지에 검은 자국이 남아 있으나 전후 카메라 자세가 달라 제거 면적률은 미계산. 시편 재준비 여부도 미확인."],
    ["자동 removal heatmap", "미보정 k=1과 TCP 속도 기반 상대 지표. 회전 공구 RPM/실제 제거량 미측정 및 격자 영역 차이 때문에 실제 제거율 비교에 사용하지 않는다."],
    ["B 원본 로그", B["path"]], ["C 원본 로그", C["path"]],
    ["B ckpt", B["checkpoint"]], ["C ckpt", C["checkpoint"]],
    ["상세 분석", str(OUT / "report.md")], ["계산 결과 JSON", str(OUT / "comparison.json")],
    ["비교 그래프", str(OUT / "comparison.png")],
    ["B 영상", "/home/eunseop/Videos/Screencasts/polishing_inference_20260920_162346_FLOW_IL_minus_F_E1_off.webm — 10fps, 38.5초 정상 저장"],
    ["C 영상", "/home/eunseop/Videos/Screencasts/polishing_inference_20260920_160641_FLOW_rel_single_90deg_best.webm — 종료 마무리 오류로 끝부분 불완전; CSV 로그는 정상"],
]

csv_path = OUT / "BC_results_20260920.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerow(["비교 전제", "B: 9/16 E1 OFF / C: 9/10 기존 baseline", "", "1회", "1회", "동일 학습 실행의 B/C 비교 아님. 공통 0–37.5초; 자세한 조건은 XLSX의 실험조건·해석과원본 시트 참조."])
    writer.writerows([[round(v, 4) if isinstance(v, float) else v for v in r] for r in rows])

# A small dependency-free workbook: inline strings, numeric cells, three worksheets.
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("", NS)
ET.register_namespace("r", REL)
def element(parent, tag, attrs=None, text=None):
    item = ET.SubElement(parent, f"{{{NS}}}{tag}", attrs or {})
    if text is not None:
        item.text = str(text)
    return item
def xml(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
def col(index):
    return chr(65 + index)  # all sheets have at most 6 columns
def sheet(title, subtitle, headers, content, widths):
    root = ET.Element(f"{{{NS}}}worksheet")
    n = len(headers)
    element(root, "dimension", {"ref": f"A1:{col(n-1)}{len(content)+4}"})
    views = element(root, "sheetViews")
    view = element(views, "sheetView", {"workbookViewId": "0"})
    element(view, "pane", {"ySplit": "4", "topLeftCell": "A5", "activePane": "bottomLeft", "state": "frozen"})
    element(root, "sheetFormatPr", {"defaultRowHeight": "30"})
    cols = element(root, "cols")
    for i, width in enumerate(widths, 1):
        element(cols, "col", {"min": str(i), "max": str(i), "width": str(width), "customWidth": "1"})
    data = element(root, "sheetData")
    for index, values in enumerate([[title], [subtitle], [], headers, *content], 1):
        height = 32 if index == 1 else 46 if index == 2 else 12 if index == 3 else 40 if index == 4 else 48
        r = element(data, "row", {"r": str(index), "ht": str(height), "customHeight": "1"})
        for j, value in enumerate(values):
            style = 1 if index == 1 else 2 if index == 2 else 3 if index == 4 else 5 if isinstance(value, float) else 4
            cell = element(r, "c", {"r": f"{col(j)}{index}", "s": str(style)})
            if isinstance(value, (int, float)):
                element(cell, "v", text=value)
            else:
                cell.set("t", "inlineStr")
                element(element(cell, "is"), "t", text=value)
    element(root, "autoFilter", {"ref": f"A4:{col(n-1)}{len(content)+4}"})
    merges = element(root, "mergeCells", {"count": "2"})
    for i in (1, 2):
        element(merges, "mergeCell", {"ref": f"A{i}:{col(n-1)}{i}"})
    element(root, "pageMargins", {"left": "0.25", "right": "0.25", "top": "0.4", "bottom": "0.4", "header": "0.2", "footer": "0.2"})
    element(root, "pageSetup", {"orientation": "landscape", "paperSize": "9", "fitToWidth": "1", "fitToHeight": "0"})
    return xml(root)

styles = f'''<styleSheet xmlns="{NS}">
<fonts count="4"><font><sz val="11"/><name val="Malgun Gothic"/></font><font><b/><sz val="18"/><color rgb="FFFFFFFF"/><name val="Malgun Gothic"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Malgun Gothic"/></font><font><sz val="11"/><color rgb="FF92400E"/><name val="Malgun Gothic"/></font></fonts>
<fills count="5"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF16324F"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FF256B83"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFF3CD"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left/><right/><top/><bottom style="thin"><color rgb="FFD8E1E8"/></bottom><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="6"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center"/></xf>
<xf numFmtId="0" fontId="3" fillId="4" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf>
<xf numFmtId="0" fontId="2" fillId="3" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf>
<xf numFmtId="2" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment wrapText="1" vertical="center" horizontal="right"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
workbook = ET.Element(f"{{{NS}}}workbook")
sheets = element(workbook, "sheets")
for i, name in enumerate(["B C 결과표", "실험조건", "해석과원본"], 1):
    element(sheets, "sheet", {"name": name, "sheetId": str(i), f"{{{REL}}}id": f"rId{i}"})
pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
relationships = f'<Relationships xmlns="{pkg}">' + ''.join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,4)) + f'<Relationship Id="rId4" Type="{REL}/styles" Target="styles.xml"/></Relationships>'
types = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,4)) + '</Types>'
xlsx_path = OUT / "BC_results_20260920.xlsx"
with ZipFile(xlsx_path, "w", ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", types)
    z.writestr("_rels/.rels", f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    z.writestr("xl/workbook.xml", xml(workbook))
    z.writestr("xl/_rels/workbook.xml.rels", relationships)
    z.writestr("xl/styles.xml", styles)
    z.writestr("xl/worksheets/sheet1.xml", sheet("B·C 실험 결과 — 2026-09-20", "B: 9/16 E1 OFF / C: 9/10 기존 baseline. 각 1회이며 동일 학습 실행의 B/C 비교가 아닙니다. 공통 비교 구간: 첫 plan 이후 0–37.5초.", columns, rows, [13, 35, 12, 27, 32, 66]))
    z.writestr("xl/worksheets/sheet2.xml", sheet("실험 조건과 분류", "B=(힘 observation X, 학습된 힘 action O), C=(O,O). 두 조건 모두 하위 힘 제어와 안전 감시를 유지합니다.", ["항목", "B — Force-obs OFF", "C — Force-obs ON (기존 baseline)", "설명"], conditions, [42, 33, 38, 78]))
    z.writestr("xl/worksheets/sheet3.xml", sheet("해석·집계 방법·원본 경로", "단일 실행의 기술적 비교입니다. 실제 가공 완료/제거율/통계적 우열은 판정하지 않았습니다.", ["항목", "내용"], notes, [30, 148]))

# Verify archive, XML, row counts and numeric B/C placement before copying out.
with ZipFile(xlsx_path) as z:
    assert z.testzip() is None
    for name in z.namelist():
        ET.fromstring(z.read(name))
    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    actual = root.find(f"{{{NS}}}sheetData")
    assert len(actual) == len(rows) + 4
    for i, expected in enumerate(rows, 5):
        for j in (3, 4):
            cell = root.find(f".//{{{NS}}}c[@r='{col(j)}{i}']")
            if isinstance(expected[j], (int, float)):
                assert float(cell.find(f"{{{NS}}}v").text) == expected[j]
with csv_path.open(encoding="utf-8-sig", newline="") as f:
    reread = list(csv.reader(f))
assert len(reread) == len(rows) + 2 and all(len(r) == 6 for r in reread)
print(f"Validated {len(rows)} result rows, {len(conditions)} condition rows, {len(notes)} notes.")
for p in (xlsx_path, csv_path):
    print(p, p.stat().st_size, "bytes")
