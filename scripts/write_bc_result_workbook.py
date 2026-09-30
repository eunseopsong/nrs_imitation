"""Render a B/C table_spec.json into validated Excel and UTF-8 BOM CSV files."""
import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("table_spec", type=Path)
args = parser.parse_args()
OUT = args.table_spec.resolve().parent
spec = json.loads(args.table_spec.read_text())
columns, rows, conditions, notes = (spec[k] for k in ("columns", "rows", "conditions", "notes"))
csv_path = OUT / "BC_results_20260920.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerow(["비교 전제", spec["subtitle"], "", "1회", "1회", "조건·집계 방법은 XLSX의 실험조건 및 해석과원본 시트 참조"])
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
    z.writestr("xl/worksheets/sheet1.xml", sheet(spec["title"], spec["subtitle"], columns, rows, [13, 35, 12, 27, 32, 66]))
    z.writestr("xl/worksheets/sheet2.xml", sheet("실험 조건과 분류", "B=(힘 observation X, 학습된 힘 action O), C=(O,O). 두 조건 모두 하위 힘 제어와 안전 감시를 유지합니다.", ["항목", columns[3], columns[4], "설명"], conditions, [42, 33, 38, 78]))
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
