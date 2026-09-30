import csv,datetime,hashlib,json,shutil
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path('/home/eunseop/nrs_imitation')
OUT=Path(__file__).resolve().parent
REPORT=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
device=json.loads((OUT/'controller_readback.json').read_text())
runtime=json.loads((OUT/'runtime_readback.json').read_text())
protection=json.loads((OUT/'protection_status.json').read_text())
if not (OUT/'verification_before_linking.json').exists():
 shutil.copy2(REPORT/'verification.json',OUT/'verification_before_linking.json')
existing=json.loads((REPORT/'verification.json').read_text())
basis=json.loads((REPORT/'settings_basis.json').read_text())
config=ROOT/'experiments/e2_A_pilot_20260927/config.json'
record=ROOT/'experiments/e2_A_pilot_20260927/commissioning_request.json'
untouched={str(p):sha(p) for p in (config,record)}
# Extract the actual remote snapshots, never substitute a local driver copy.
remote_sources=[]
for f in device['files']:
 if f.get('exists') and 'text' in f:
  remote=Path(f['path'])
  rel=remote.relative_to('/home/nrs_forcecon')
  dest=OUT/'robot_pc_files'/rel
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(f['text'])
  assert sha(dest)==f['sha256']
  remote_sources.append(dict(remote_path=f['path'],local_snapshot=str(dest),sha256=f['sha256']))
write(OUT/'remote_source_manifest.json',dict(host=device['host'],captured_at_utc=device['captured_at_utc'],files=remote_sources))
manual_url='https://www.universal-robots.com/download/manuals-cb-series/user/ur10/315/user-manual-ur10-cb-series-sw315-english-international-en/'
accepted=dict(reused_without_retest=True,F0_candidate_N=23.,command_frame='controller_tcp',command_axis='+Fz',
 teacher_mean_N=existing['teacher_mean_reproduced_N'],training_episodes=38,
 F0_selection_scope='F0 scalar only; not the full command sequence or measured loads',
 BC_matching_runs=10,offline_mock_tests=140,
 previous_verification=ref(OUT/'verification_before_linking.json'),settings_basis=ref(REPORT/'settings_basis.json'),
 calculation_sources=basis['F0'],archive_config_sources=basis['BC_archive_checks'],
 test_results=ref(REPORT/'tests.xml'),test_log=ref(REPORT/'tests.log'),
 config=ref(config),commissioning_record=ref(record),
 execution_contract_hash=json.loads(config.read_text())['execution_contract_hash'],
 not_claimed=['process performance validation','hardware safety validation','F0 freezing'])
write(OUT/'established_evidence_links.json',accepted)
# Three groups, exactly the 12 unresolved fields. No guessed values.
rows=[
 ('실제 하중 보호','measured_force_abs_limits_N','N / 선택한 감시축','실측 [Fx,Fy,Fz] 절대 중단 한계: [____, ____, ____]','로봇 PC 하중감시 설정 / UR 화면 A'),
 ('실제 하중 보호','measured_torque_abs_limits_Nm','N·m / 감시축·기준점','실측 [Tx,Ty,Tz] 한계: [____, ____, ____]; 기준점: ____','로봇 PC 6축 감시 / 센서 장착도 / UR 화면 D'),
 ('실제 하중 보호','sensor_raw_force_abs_limits_N','N / 센서 원시축','영점·필터·중력보정 전 축별 한계: [____, ____, ____]','센서 제어기·드라이버 원시 패킷 / 하중감시 설정'),
 ('실제 하중 보호','sensor_raw_torque_abs_limits_Nm','N·m / 센서축·원점','원시 토크 축별 한계: [____, ____, ____]','센서 제어기·드라이버 원시 패킷 / 센서 장착도'),
 ('실제 하중 보호','measured_wrench_frame','좌표계 / 토크 기준점','sensor / robot_base / controller_tcp 중 실제 감시축과 토크 원점 확인: ____','센서 장착도 / 드라이버 축변환 / UR 화면 D'),
 ('실제 하중 보호','limits_evidence','설정·기록 ID','기존 보호를 재사용할 경우 적용 하중·축·정지 경로의 동등성 근거: ____','UR 화면 A·C / 기존 하중감시 설정·확인 기록'),
 ('실제 하중 보호','sensor_overload_evidence','포화·복합하중·고장 검출','포화/과부하 검출과 정지 연결의 근거: ____; raw 명칭 토픽만으로 확정하지 않음','센서 제어기·드라이버 / 제조사 문서 / 현장 확인 기록'),
 ('실제 센서 감시','acquisition_max_age_s','s / 원본 수신 시각','실제 UDP 수신 나이 상한: ____ s; 선정·시계/지연 근거: ____','로봇 PC 수신루프·watchdog / 양쪽 PC 시각 상태'),
 ('실제 센서 감시','sensor_driver_deployment_evidence','실행 버전·활성 상태','현재 버전은 기록 완료. 미관측 감시 토픽 2종의 활성화 또는 동등 감시 근거: ____','nrsgene 실행 PID·바이너리 SHA / ROS 발행자·수신 기록'),
 ('현장 동작 확인','workspace_evidence','mm·rad / 기준 프레임','현재 공구·지그의 경계 적용과 TCP 보정 일치 확인. 시작 로그의 보정 불일치 경고 처리 근거: ____','UR 화면 B·D / 현재 공구·지그 / 드라이버 교정 로그'),
 ('현장 동작 확인','automatic_stop_evidence','정지 조건·시각·관측 기록','고장별 명령 억제 → 큐 취소 → 실제 정지의 확인 기록: ____','UR 화면 C / 실제 정지 로그 / singleArm 명령 서비스'),
 ('현장 동작 확인','approach_exit_evidence','mm·rad / 작업 순서','시작 자세·가공 marker·접촉 해제·이탈·스핀들 절차의 적용 기록: ____','현재 공구·지그 / 로봇 프로그램 / 기존 운전 절차')
]
assert len(rows)==12
remaining=set(x.split(':',1)[0] for x in existing['remaining_hardware_blockers'])
assert {x[1] for x in rows}==remaining
fields=[dict(group=g,field=f,unit_and_frame=u,needed=v,device_or_screen=s,
 status='field_confirmation_required',selected_value=None) for g,f,u,v,s in rows]
write(OUT/'field_confirmation.json',dict(rows=fields,no_values_inferred=True,manual=manual_url,
 grouping_only=True,blank_fields_do_not_prove_absence_of_protection=True))
with (OUT/'field_confirmation.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['묶음','필드명','단위·좌표계','필요한 값 또는 확인 사항','확인할 장비·화면','담당자 기입/기록 ID'])
 for row in rows:w.writerow([*row,''])

controllers=[dict(name=x['name'],state=x['state']) for x in runtime['controllers']['response']['controller']]
facts=[
 dict(item='UR safety state',value='NORMAL',evidence='protection_status.json: queries./dashboard_client/get_safety_mode and dynamic_joint_states',
      scope='current reported state; not a trip test or validation of configured limits'),
 dict(item='UR software',value='3.15.7',evidence='protection_status.json: dynamic_joint_states software-version interfaces'),
 dict(item='current application control mode',value='Position',evidence='runtime_readback.json: /ur10skku/ctlMode'),
 dict(item='feedback streams',value=['/ur10skku/currentF','/ur10skku/currentP','/ur10skku/ftdata','/ur10skku/ftdata_tcp','/ur10skku/ftdata_base','/ur10skku/ftdata_tcp_raw'],
      evidence='runtime_readback.json: nonzero received messages',scope='delivery observed; does not establish ADC freshness'),
 dict(item='configured and loaded robot YAML',value='/home/nrs_forcecon/dev_ws/install/Y2RobMotion/share/Y2RobMotion/config/setup_parameters.yaml',
      evidence='controller_readback.json: current PID startup log + installed file SHA; runtime_readback.json: setup_file parameter'),
 dict(item='sensor calibration setting',value='gravity_compensation_enabled=true; startup matrix_loaded=true',
      evidence='runtime_readback.json: /FTGetMain parameters; controller_readback.json: PID 601830 startup log',
      scope='loaded/active configuration, not a new calibration validation'),
 dict(item='controller_stopper',value='node and live parameters observed; joint_controller_active=true',
      evidence='protection_status.json: controller_stopper',scope='not a demonstrated response to an AIDIN sensor fault'),
 dict(item='A source provenance interfaces',value={'/ur10skku/ft_acquisition':0,'/ur10skku/currentF_provenance':0},
      evidence='runtime_readback.json: publisher_count=0',scope='not observed active in this ROS domain during capture; not proof that all protection is absent'),
 dict(item='driver calibration warning',value='connected robot calibration differs from supplied kinematics config',
      evidence='controller_readback.json: current ur_ros2_control_node PID 601293 startup log',
      scope='requires field review of TCP/workspace calibration; no correction performed'),
 dict(item='raw topic meaning',value='ftdata_tcp_raw uses filtered, zero-offset sensor values rotated to TCP before gravity compensation',
      evidence='robot_pc_files/.../FTGetMain.cpp lines 648-663 and 756-764; FT_EtherGet.cpp lines 150-155',
      scope='does not supply the required pre-zero/pre-filter sensor-axis overload evidence')
]
summary=dict(captured_from='nrsgene (192.168.0.151), live ROS_DOMAIN_ID=30',read_only=True,
 established_items_retested=False,mock_tests_rerun=False,hardware_commands_sent=False,controller_restarted=False,sensor_zeroed=False,
 evidence_files={name:ref(OUT/name) for name in ('controller_readback.json','runtime_readback.json','protection_status.json','remote_source_manifest.json','established_evidence_links.json','field_confirmation.json')},
 actual_device_findings=facts,controller_states=controllers,
 running_binaries=[dict(pid=p['pid'],path=p['executable'],sha256=p['running_image']['sha256'],started_at_utc=p['started_at_utc'])
                   for p in device['processes'] if 'executable' in p],
 existing_protection_reuse=dict(partial_evidence_preserved=True,equivalence_to_AFT_six_axis_and_source_watchdog='not_established',
  not_concluded='Missing commissioning values do not show that the robot has no protective functions. NORMAL state does not prove that every requested protection requirement is satisfied.'),
 remaining_groups={group:[r[1] for r in rows if r[0]==group] for group in dict.fromkeys(r[0] for r in rows)},
 runtime_config_updated=False,F0_frozen=False,quality_validation='pending',
 manual_reference=dict(url=manual_url,local=ref(OUT/'UR10_CB3_SW315_manual.pdf'),
  pages='85–86, 91, 95, 102, 125; only screen locations used, no safety limits copied'))
write(OUT/'device_findings.json',summary)

pdfmetrics.registerFont(TTFont('Nanum','/usr/share/fonts/truetype/nanum/NanumGothic.ttf'))
pdfmetrics.registerFont(TTFont('NanumBold','/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'))
style=ParagraphStyle('cell',fontName='Nanum',fontSize=7.8,leading=10.1,wordWrap='CJK',spaceAfter=0)
small=ParagraphStyle('small',parent=style,fontSize=8.2,leading=11)
title=ParagraphStyle('title',parent=style,fontName='NanumBold',fontSize=15,leading=19)
def p(text,st=style):return Paragraph(escape(text).replace('\n','<br/>'),st)
pdf=OUT/'field_confirmation_one_page.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=landscape(A4),leftMargin=22,rightMargin=22,topMargin=19,bottomMargin=19)
story=[p('E2 A | 장비·제어 담당자 현장 확인표 (12개 필드)',title),Spacer(1,5),
 p('실제 조회: 2026-09-27 18:58–19:03 KST · nrsgene / ROS domain 30 · UR 3.15.7 · Safety NORMAL / 제어 모드 Position',small),
 p('힘·TCP 수신 및 실행 버전은 기록됨. 원본 수신 감시 토픽 2종은 발행자 0개. 아래 빈칸은 보호 기능 부재 판정이 아니며, 현장 근거를 기입하는 자리임.',small),Spacer(1,7)]
table=[[p(x) for x in ['필드명','단위·좌표계','필요한 값 또는 확인 사항','확인할 장비·화면']]]
groups=[];prev=None
for group,field,units,needed,screen in rows:
 if group!=prev:
  groups.append(len(table));table.append([p(group),p(''),p(''),p('')]);prev=group
 table.append([p(field),p(units),p(needed),p(screen)])
width=landscape(A4)[0]-44
t=Table(table,colWidths=[179,108,294,width-581],hAlign='LEFT')
commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#a8b4bf')),
 ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce5ee')),('LEFTPADDING',(0,0),(-1,-1),5),
 ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
for row in groups:commands.extend([('SPAN',(0,row),(-1,row)),('BACKGROUND',(0,row),(-1,row),colors.HexColor('#eef2f5')),('TOPPADDING',(0,row),(-1,row),3),('BOTTOMPADDING',(0,row),(-1,row),3)])
t.setStyle(TableStyle(commands));story.append(t);story.append(Spacer(1,7))
story.append(p('UR 화면: A = Installation → Safety → General Limits / B = Safety → Boundaries / C = Safety → Safety I/O / D = Installation → TCP Configuration',small))
story.append(Paragraph('화면명 출처: <link href="'+manual_url+'" color="#24557a">UR10 CB3 SW3.15 공식 매뉴얼</link> · 기존 F0/140개 mock 근거는 재검사 없이 연결. 품질 pending, F0_frozen=false 유지.',small))
story.append(Spacer(1,6));story.append(p('확인자(장비·제어 담당): ____________________    확인 일시: ____________________    설정/시험 기록 ID: ____________________',small))
def make_canvas(*args,**kwargs):
 c=canvas.Canvas(*args,**kwargs)
 c.setTitle('E2 A - device and field confirmation - 2026-09-27')
 c.setDateFormatter(lambda y,m,d,hh,mm,ss: f"D:{y:04d}{m:02d}{d:02d}{hh:02d}{mm:02d}{ss:02d}+09'00'")
 return c
doc.build(story,canvasmaker=make_canvas)

# Append provenance to the existing accepted record; do not touch runtime guards.
existing['established_evidence_links']=ref(OUT/'established_evidence_links.json')
existing['actual_device_readback']=ref(OUT/'device_findings.json')
existing['field_confirmation_table']=dict(pdf=ref(pdf),csv=ref(OUT/'field_confirmation.csv'))
existing['followup_scope']='Read-only actual-device evidence; accepted F0, archived-setting comparison and 140 mock tests were not rerun.'
write(REPORT/'verification.json',existing)
assert all(sha(p)==h for p,h in untouched.items())
write(OUT/'artifact_manifest.json',dict(files=[ref(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'],
 runtime_configuration_hashes_unchanged=untouched,existing_record=ref(REPORT/'verification.json')))
print(json.dumps(dict(pdf=str(pdf),csv=str(OUT/'field_confirmation.csv'),actual_device_evidence=str(OUT/'device_findings.json'),
 existing_record_updated=str(REPORT/'verification.json'),runtime_settings_changed=False),ensure_ascii=False))
