# 실행 경로 감사

근거는 2026-09-26 로컬 working tree이다. HEAD는 `a556c58333f53c265a1939a1a8e88b48242652ed`이며, 이미 수정·미추적 파일이 다수 있다. 적용 가능한 상위/프로젝트 `AGENTS.md`는 발견하지 못했다. 다른 프로젝트의 AGENTS는 적용하지 않는다. 코드·설정 해시는 `source_hashes.json`, 기존 변경은 `git_status_at_audit.txt`에 보존했다.

아래 약어는 실제 파일을 가리킨다. 줄 번호는 이 working tree 기준이다.

| 약어 | 경로 |
|---|---|
| IC | [inference_core.py](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py) |
| PR | [e2_providers.py](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/nrs_imitation/e2_providers.py) |
| EX | [e2_executor_node.py](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/nrs_imitation/e2_executor_node.py) |
| TE | [e2_timed_execution.py](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/nrs_imitation/e2_timed_execution.py) |
| IM | [inference_metrics.py](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/nrs_imitation/inference_metrics.py) |
| EM | [execution_metrics.py](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/nrs_imitation/execution_metrics.py) |
| LC / LG | [clean launch](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py) / [gradcam launch](/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation/launch/inference_gradcam_single_cam.launch.py) |
| CFG | [기존 config.json](/home/eunseop/nrs_imitation/experiments/e2_rule_replay_20260920/config.json) |
| RC | [robot_command.cpp](/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/src/robot_command.cpp) |
| SA | [singleArm_cmd.cpp](/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/src/singleArm_cmd.cpp) |
| RM | [robot_motion.cpp](/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/src/robot_motion.cpp) |
| FC | [force_control.cpp](/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/src/force_control.cpp) |
| SP | [setup_parameters.yaml](/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/config/setup_parameters.yaml) |

**실행 경로 세 가지를 구분해야 한다**

1. 과거 C `service_stream`: IC `_on_infer_timer`(4930) → FLOW forward·denormalization(5137–5170) → `_postprocess_provider_action`(4791) → `Plan` → `_ptp9d_stream_topup`(6222), `_ptp9d_stream_update_force`(6331) → `/singleArm_cmd/single_arm_command` → SA service 분기(491–541) → RC `streamLoop`(404) → `/ur10skku/cmdMotion` → RM `cmdMotionCB`(200), `main_control`(543) → FC `control_force`(576), `execute_force_control` → IK(570) → RM `state_publisher`(384) → `/forward_position_controller/commands` → 외부 UR driver/로봇. 마지막 driver 및 실제 로봇 동작은 미확인이다.
2. 기존 R/T 및 CFG의 C `timed_topic`: R `PR.rule_actions`(135) / T `PR.TimedActions.load`(76), `export_episode`(207) / C 기존 FLOW → IC `_postprocess_provider_action`(4791) → `_e2_publish_plan`(4869) → `/e2/plan` → EX `on_plan`(161) → TE `accept`(71), `tick`(100) → EX `tick`(295)에서 125 Hz `/ur10skku/cmdMotion` 직접 publish → 위와 같은 RM/FC 이후 경로. 가공 중 SA/RC service 큐·보간을 거치지 않는다.
3. launch 기본값 `service_call`: LC 267–269의 기본값은 `service_call`이다. IC `_ptp9d_advance`(5996) → `PTP9D` → RC `PTP9D_command_gen`(216)의 MotionBlender9D 경로다. `service_stream`과 다른 배치 실행이며 이번 공통 경로로 자동 선택하지 않는다. 기본 checkpoint도 LC 184–188의 `20260802_1549`이므로 무인자 launch는 이번 E1에 부적합하다.

**R / T / C 감사표**

| 항목 | R (기존 CFG) | T (기존 CFG) | C (기존 서비스 / CFG) | 근거·판정 |
|---|---|---|---|---|
| 생성부 | 명시 endpoint·rotvec, quintic 가감속·force ramp, line만 지원 | HDF5 고유 시점을 복원한 TimedActions, 네트워크 없음 | FLOW 반복 추론, 실제 측정 force observation, position6+force3 | PR 135–260; IC 2254–2261, 4930–5206. 생성부 재사용 가능, 이번 recipe/episode/ckpt 미선택 |
| 단위·변환 | mm, rotvec rad, N; 상대 XY→base XY | 동일 | denormalization 후 동일; service payload 자세는 rotvec **성분을 degree로 변환**, Euler로 해석하지 않음 | IC 4791–4813, 6281–6289; RC 360–368에서 다시 rad. relative frame은 XY 평행이동, Z/자세는 base 유지 |
| 서비스·mode | 시작 PTP·정지 STOP만 service, 가공 cmdMotion topic, Force mode | 동일 | service_stream은 START/APPEND/SET_FORCE/STOP; CFG의 C는 R/T와 같은 topic | EX 92–103, 181–205, 371; SA 491–541. service type은 `y2_rob_motion_interfaces/srv/SingleArmCommand` |
| timestamp·전송·horizon | 전체 유한 plan 한 번, 125 Hz 샘플링 | 전체 episode 한 번; 본체는 원본 비균일 `time` 사용 | 모델 horizon 128, 30 Hz 가정, knot span 127/30≈4.233 s, replan 120/30=4 s; legacy stream은 최대 40점, stride2, 예상 lookahead2.5 s | IC 4874–4911; TE 43–53, 100–122; CFG.il/ replay; 실제 과거 C runtime은 log_transport_evidence.json. 4.27은 metadata chunk duration이며 실행 확인 아님 |
| queue·chunk·replace | TE final plan은 교체 금지 | 동일, 순차 chunk 공급 아님 | timed C는 새 plan으로 교체; legacy는 최근 plan에서 tail XYZ에 가장 가까운 index 다음부터 append | TE 71–86; IC 6255–6260. 왕복·체류에서 nearest XYZ는 시점/방향 식별을 보장하지 않음 |
| buffer 소진·초기화 | startup Idling/STOP/Position 정지 확인; 유한 plan 끝은 endpoint 전송 뒤 정지 | 동일 | timed C horizon 소진은 plan_underrun. legacy는 큐가 비면 마지막 pose를 유지하며 **마지막 힘 target도 유지**; START는 측정 pose·0 force로 seed | EX 189–224, 295–306; TE 114–121; RC 302–337, 469–478. legacy 실제 queue depth·소진 telemetry 없음; IC 6311은 추정치 |
| 위치·자세 보간 | TE linear XYZ + SO(3) Slerp, Python gain/rate cap | 동일 | legacy: Python pose6 이동평균 window35 + stride2 → C++ 위치/rotvec 성분 선형 보간, 시간=max(거리/속도, 각성분거리/각속도,0.05 s) → low-pass/rate cap | PR 44–65; TE 43–53, 141–155; IC 6237–6244; RC 435–499; robot_command.hpp 123–126. 동일 보간이라고 볼 수 없음 |
| 힘 보간·적용 | 같은 elapsed sample에서 선형 force → contact gate → Python low-pass/rate cap | 동일 | legacy APPEND 힘 열은 무시; 측정 XYZ의 nearest plan index로 Fz 선택 → 독립 SET_FORCE → C++ latest-target low-pass/rate cap | TE 122–159; IC 6331–6412; RC 453–499. 원본 pose–force–time 대응을 서비스에서 보존할 수 없음 |
| clipping·gate·게인·안전 | TE signed measured Fz 3/1.2 N hysteresis, Fx/Fy=0, 구성값 Fz clip, stale/tick/workspace/watchdog 검사 | 동일 | legacy도 gate/clip 존재. 단, `_publish_cmd`의 workspace 검사는 stream TRACK 경로에서 호출되지 않음 | TE 103–155; IC 4685, 4791, 5327, 6676–6686. CFG fz_hard_limit_N=0은 clip 해제이지 0 N 안전상한이 아님. 이 과거 값을 이번 안전승인으로 간주하지 않음 |
| 접근·가공·종료·이탈 | 공통 demo-start PTP 후 source waypoint approach, 정상 완료에만 별도 lift/home | T 원본 접근·종료 포함, 정지 hold; phase unknown | C 정책 접근·반복 추론, operator finish/timeout; legacy service TRACK은 일반 PRELOAD/RELEASE 분기 앞에서 return | IC 5788–5954, 6676; EX 256–375; CFG.common/recipe.return_home. R만 home 복귀, R 별도 복귀예산50 s이므로 공통 종료 조건 아님 |
| 수동 중단 | EX SIGINT/SIGTERM→Idling→STOP ACK→Position·fresh stationary 검증 | 동일 | timed C 동일. legacy IC KeyboardInterrupt는 로그·destroy만; service stream 종료/정지 확인이 보장되지 않음 | EX 398–416, 189–224, 302–340; IC 2763–2777, 7050–7073. STOP 자체도 RC thread/queue 정리이며 물리 정지 완료 의미 아님 |
| feedback·로그 | currentP/currentF 실제 feedback, sampled/gated/sent 구분 | 동일, 긴 plan snapshot 별도 | IM prediction/postprocess/sent/ACK 기록. 실제 controller-applied force는 미확인 | EX 118–139, 239–254; IM 224–286, 340–440. EM bounded queue writer. 기본 최대20 Hz feedback, acquisition timestamp 없음 |

SP의 기존 설정은 CONTROL_PERIOD=0.008, ANGULAR_VELOCITY_LIMIT=40 deg/s, Force_Con_Coordinate=1(TCP), Force_Con_Mode=3(NAF mdGradi)이다. RC stream gain15/s, force rate30 N/s는 헤더 상수다. 이들은 **관찰한 로컬 값**이며 이번 실험 승인값이 아니다. 하위 힘 제어는 adaptive MDK를 포함하므로 상수 게인 하나로 동일성을 설명할 수 없다. 현재 배포된 controller 모델·파라미터·초기 상태·안전한계의 고정 근거가 필요하다.

**실제 과거 로그로 확인한 경로**

`log_transport_evidence.json`은 로그 row 수와 command_id 기반 요청 수를 구분한다. APPEND 한 요청의 waypoint 여러 행을 요청 여러 번으로 세지 않는다.

| 실제 폴더 접두사 (모두 logs/inference_metrics 아래) | 기록된 전송 |
|---|---|
| C_force_obs_ON_E1_20260916_20260920T171726 | runtime ptp9d_use_stream=true, APPEND와 SET_FORCE 존재 |
| E2_line_R_validation_02_executor_20260920T195830 | e2_timed_topic_v1, cmdMotion |
| E2_line_T_r01_executor_20260920T201023 | e2_timed_topic_v1, cmdMotion |
| E2_line_C_r01_executor_20260920T201420 | e2_timed_topic_v1, cmdMotion |

즉 과거 9/20 후반 R/T/C는 서로 timed 실행기를 공유했지만 그 C는 앞선 service C와 실행 조건이 다르다. 구번호 결과를 이번 E1의 독립 trial로 자동 편입하지 않는다. 로그상의 송신 또는 service 성공을 물리 실행 완료·스크래치 제거 성공으로 바꾸지 않는다.

CFG의 R geometry는 episode_29의 힘 threshold 기반 구간에서 도출됐고, 18 N은 과거 C 로그의 contact 구간 통계에서 정한 후보였다. `main_experiment_ready=false`가 남아 있다. 이를 이번 E1의 별도 calibration/validation 완료 recipe 또는 평가용 교시 F_ref로 취급하지 않는다. 기존 planner는 명시 endpoint를 받으며 mask→경로 생성 기능이 이미 완성됐다고 주장하지 않는다.

**교시 및 checkpoint 검사**

- 지정 dataset에는 episode_0–41의 42개 compact HDF5가 있다. action/position은 [N,6], action/force는 [N,3]의 고유 시점 배열이다. loader가 `_slice_pad`로 겹치는 horizon을 만든다([dataset.py](/home/eunseop/nrs_imitation/source/data/dataset.py:714)). 따라서 학습 batch를 연결하지 않고 HDF5 원시 시점 배열을 읽는 기존 exporter를 재사용할 수 있다.
- `episode_29`의 source는 `20260819_1618/merged_hdf5/hdf5_recorder_single_cam_20260819_1618.hdf5:episodes/ep_0003`이다. 443개 시점, 14.9104065895 s, 최대 간격 0.0670065880 s. 시간은 `subscriber_receive_time_unix`이며 센서 hardware acquisition time이 아니다. 중간 gap을 일률적으로 30 Hz라고 덮어쓰면 안 된다.
- action/force는 source `ft`와 일치한다. source는 `/ftsensor/measured_Cvalue`를 EMA(alpha0.2) 처리한 힘이다. `raw_ft`와 구분한다. [converter](/home/eunseop/nrs_imitation/source/custom/_imitation_form_converter.py:1759)는 이 값을 `command_target_force`로 복사한다. 로봇에서 과거 실제 적용한 목표 힘을 기록한 채널이라는 뜻은 아니다.
- seed0·lexical episode order로 train38/validation4를 재구성해 saved normalizer extrema와 timestep count를 대조했다. 결과는 dataset_verification.json에 있다. historical split ID 저장본이나 독립 평가 데이터의 존재를 주장하지 않는다. validation도 모델 선택에 사용되므로 자동으로 독립 test reference가 되지 않는다.
- 기존 episode contact sheet의 23–37 및 3번에서 한 개 긴 직선 표식을 확인했다. target dataset의 2점 task를 확인하지 못했으므로 line task 후보로만 제한한다. 다른 폴더의 `2dot` checkpoint 존재는 이번 데이터의 2점 교시 근거가 아니다.
- C 후보와 normalizer를 CPU `torch.load(weights_only=True, mmap=True)`로 읽어 실제 state input [256,9], action output [9,256,1], metadata history30/horizon128/flow steps10을 확인했다. 추론을 실행하거나 후보를 선택하지 않았다. 9/10은 ON 플래그 미기록, 9/16 ON1531은 명시 true, OFF1531은 명시 false이며 C에서 제외해야 한다. 전체 path/hash/epoch는 checkpoint_candidates.json 참조.
- `scripts/e2_experiment.py` 22, 31, 184–187에는 9/16 ON checkpoint 상수가 남아 있다. 새 명시 선택을 이 스크립트가 존중한다고 가정하지 않는다. 구 manifest의 “not trained” 상태도 현재 실제 checkpoint보다 우선하지 않는다.

**힘 좌표계와 적용 명령의 구분**

`Y2FT_AQ/src/FTGetMain.cpp` 605, 659–736에서 sensor filtering → TCP 변환 → 조건부 중력 보정 → base 변환 → `/ur10skku/ftdata` 발행을 확인했다. RM `ftsensorCB`(315)는 이를 저장하고 `state_publisher`(410)가 header 없는 `currentF`로 재발행한다. 따라서 현재 로그의 Fz는 base Fz이며 원시 센서 Fz가 아니다. 원본 센서 시각도 소실된다. ftdata의 header도 driver 발행 시각이며 장치 acquisition 시각으로 확인되지 않았다.

FC 280–335는 base 실측 힘을 TCP로 투영하지만 Fd(366)는 요청된 force 성분을 직접 사용한다. SP의 coordinate=1 아래에서는 `SET_FORCE.fz`를 base Fz로 단정하면 안 된다. 법선 n, 압축 부호, TCP/교시 센서 변환을 검증한 뒤 같은 frame에서 projection해야 한다. 기존 Fz가 양수라는 사실만으로 동일 법선력임이 증명되지 않는다. 중력 보정과 tare는 당시 활성 상태가 확인될 때만 적용하며 이중 보정하지 않는다.

또한 FC 366–374에는 작은 비영 목표힘/미접촉 조건에서 내부 `commanded_Fd`를 별도 precontact hold로 바꾸는 로직이 있다. RM targetF(413)는 `FC_AC_desX`이므로 내부 최종 `commanded_Fd`까지 입증하지 않는다. 따라서:

- F_tar: 선택된 실행 계획의 사전 gate/limit 목표. 아직 실행에 선택되지 않은 예측 horizon 전체는 실행 프로파일이 아니다.
- F_cmd_sent: 실제 SET_FORCE 또는 Force-mode cmdMotion 전송 값. APPEND의 0 force 자리, Position pose 명령, 준비만 된 prediction은 제외한다.
- controller_reported_target: targetF를 확보해도 위 변환 전 단계로 구분한다.
- controller_applied: 적용 ID·controller clock·내부 commanded_Fd telemetry가 없어 현재 unavailable.
- F_meas: 실제 wrench에서 검증된 법선으로 투영한 측정값. command로 대체하지 않는다.

**기존 logger의 재사용 범위와 부족분**

EM 70–80의 stamp는 source header/ROS receipt/local monotonic receipt를 분리한다. EM 102–147, 234–340은 비동기 bounded queue, drop·write error·max gap·retrograde 기록을 제공한다. IM 212–265는 최대20 Hz sample gate를 control observation과 분리한다. `normal_force_signed`, velocity 필드는 있으나 현재 일반 경로에서 법선력·실제 TCP velocity 값이 채워지는 것은 아니다.

IM `reference_fixed`는 **stain XY 작업 원점**이 고정됐다는 이벤트다. 평가용 F_ref가 고정됐다는 뜻이 아니므로 새 `evaluation_reference_fixed`와 hash가 필요하다. legacy trial_start, 정지/이탈 완료, queue depth/underflow, 적용 명령 timestamp/ID는 충분하지 않다. 기존 metrics_extra_telemetry=false이며 이를 자동 true로 바꾸거나 저장률을 높이지 않는다. 20 Hz에서 관측되지 않은 힘 peak는 복원할 수 없다.

**미확인 및 현장 근거 요구**

| 항목 | 필요한 근거 | 현재 상태 |
|---|---|---|
| 실제 배포 버전 | active binary/build ID, service schema, launch args, loaded YAML/model hash | 로컬 소스만 확인; 원격 접속·실행 안 함 |
| 보간·큐 동등성 | 동일 서비스에서 소비한 action ID/time, queue depth/underflow·완료 telemetry | legacy mock 또는 source만으로 현장 동등성 확인 불가 |
| 힘 법선/부호/보정 | 교시 센서→공통 frame, TCP→base, 표면 법선, tare/gravity epoch | unverified; 정량 법선 비교 제한 |
| 실제 적용 목표힘 | 내부 commanded_Fd와 적용 timestamp/ID, clock mapping | unavailable; last-sent 오차와 구분 필요 |
| RPM·공구·재료·영역 | 동일 설정 RPM 출처, actual 측정 가능 여부, 시편/영역 ID | 이번 E1 미고정; 수치 생성 금지 |
| recipe·힘/속도·timeout | 별도 calibration/validation 기록과 사전 고정 예산 | 과거값만 존재; 이번 승인으로 승계 안 함 |
| 종료·수동 중단 | queue cancel 후 fresh mode/pose/force와 접촉 해제 확인 | timed 경로 일부 구현; legacy와 공통화 필요 |

결론은 `execution_equivalence=unverified`이다. 현재 제약에서 설정만 고쳐 동일 service·보간과 T 원래 시간 보존을 동시에 달성할 수 없다. 코드 변경 없이 여기까지 감사했으며 변경 내역은 신규 감사 문서·증거 파일뿐이다.
