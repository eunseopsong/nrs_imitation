# E2 A 첫 예비시험 준비 — 실기 준비 미완료

2026-09-27. 로봇 PC에는 소스 해시·실행 프로세스를 읽는 SSH만 수행했다. 로봇, 스핀들, homing, zeroing 명령을 보내지 않았다. 프로젝트와 상위 디렉터리에 적용되는 AGENTS.md는 없었다. 기존 변경, checkpoint, 9/26 결과는 보존했다.

## 실제 차단 조건과 변경

| 위치 / 함수 | 조사 당시 실제 조건과 의미 |
|---|---|
| `launch/e2_abc.launch.py::configure` | `hardware_blockers(cfg, 'il', enabled=True)` 결과가 있으면 attempt를 `preflight_blocked`로 저장하고 예외 발생. ROS 노드를 포함하기 전이다. `mode=check`는 노드를 시작하지 않는다. |
| `e2_providers.py::hardware_blockers` | common 설정, 모델/normalizer SHA, timed_topic 설정 및 `runtime_contract_errors`를 검사한다. 센서 과부하 보호의 존재를 증명하는 검사는 기존에 없었다. |
| `e2_ablation.py::runtime_contract_errors` | 기존 A 분기: `f0.status != 'frozen' or not isinstance(f0.command_fz_N, (int,float))`이면 차단. 이후 finite, freeze_evidence, calibration hash, candidate SHA 검사. master 값은 `status=unresolved`, `command_fz_N=null`이었다. |
| `e2_ablation.py::scheduled_force` | `processing=False`이면 0. 가공 중에는 `status != 'frozen' or not isfinite(command_fz_N)`이면 예외. 실행기에서도 이 두 번째 동결 조건을 사용했다. |
| `scripts/e2_ablation.py::main`, `freeze-f0` 분기 | candidate의 calibration_hash 불일치 또는 `not calibration.normal_force_verified`이면 동결 거부. 명령 범위·근거도 요구한다. 이는 **본 실험 레시피 확정** 기능이며 예비시험 준비와 구분했다. |
| `e2_offline.py::OfflineNode` | in-memory publisher/service로 실제 inference 함수를 바인딩하는 fixture. 동결·품질 승인 조건문은 없고 ROS context를 만들지 않는다. 이번에 등록 A 모델 검사 함수를 추가했다. |
| `experiments/e2_force_ablation_20260926/f0_candidate_status.json` | 상태 보고 sidecar. 런타임은 이 파일의 `F0_frozen`을 읽어 차단하지 않았다. 수정하지 않았다. |
| `reports/20260927_E2_F0_offline_assumption/verification.json::physical_validation`, `reports/20260927_E2_A_preparation/trained_A_validation.json::physical_quality_validated` | 오프라인 보고서의 false 값. 실기 조건문에서 읽지 않는다. 이를 true로 바꾸어도 원래 F0 차단은 해결되지 않는다. |
| `e2_ablation.py::readiness`의 reference/normal-force/stylus 결과 | 정규 힘 지표·기준 프로파일·스타일러스 평가 준비 상태. 실기용 F0 목표, 실측 힘 중단값과 별개이다. 이번에도 분석 준비 미완료를 예비시험 차단값으로 복사하지 않았다. |

변경 후 `trial_stage='pilot'`의 A는 `e2_pilot.py::pilot_errors`를 사용한다. F0는 candidate, `F0_frozen=false`, `quality_validation=pending`, `start_mode=operator_existing_procedure`를 유지해야 한다. 해당 장비·공구·작업에 맞는 승인된 commissioning 기록과 실제 보호 기능을 모두 요구한다. 품질 완료나 main F0 동결은 요구하지 않는다.

`installed_protection_errors()`는 조사한 배포에서 없는 기능 네 가지를 명시적으로 차단한다: 실측 힘·토크 중단, 원본 센서 수신 시각 감시, 센서 overrange/포화 감시, 실제 TCP 및 시작 정렬을 포함하는 작업영역 감시. 이 함수는 JSON 승인 플래그로 해제되지 않는다. 실제 기능의 구현·배포·검증 후 소스 검토가 필요하다. 테스트에서는 이 함수만 메모리에서 보호 backend mock으로 교체한다. 실기용 mock/skip/force 옵션은 없다.

main A는 기존 동결 조건에 더해 commissioning 근거, `f0.pilot_result_evidence`, 실제 예비시험 결과의 `quality_validation=reviewed`를 요구한다. 예비시험 준비만으로 본 실험에 승격하지 않는다. main 레시피 자체는 unresolved로 보존했다.

`E2Executor.__init__`에서 A의 허용 후보를 한 번 검사하고 저장한다. `tick`에서 가공 중에만 그 값을 기존 공통 실행기에 전달하고, 나머지 구간에는 0을 전달한다. 승인 파일을 125 Hz 루프에서 읽지 않는다. 기존 힘 변화율 제한 때문에 가공 종료/gate-off 직후의 **전송값**은 순간 0 대신 감쇠할 수 있다. 목표 0과 전송값을 별도로 기록한다. 정지 시에는 기존 terminal/queue 계약을 그대로 사용한다.

## 9/26 B/C 실제 설정·코드·로그 대조

B 5회, C 5회의 `launch_context/config.json`, 보관된 executor/engine, provider의 controller/FT 소스를 조사했다. 10회의 executor 설정과 motion postprocessor는 모두 같고 현재 설정과도 같다. 상세 run ID·설정·정지 이벤트는 `BC_runtime_audit.json`에 있다.

| 보호/제어 항목 | 설정 존재 및 런타임 적용 | 현재 작업에 대한 근거와 한계 |
|---|---|---|
| 공통 transport/평활화 | timed_topic 125 Hz, pose MA 35, 계획 전환 0.5 s, 가속 25 mm/s² 및 100 deg/s². IL 경로에서 활성 | 보관된 B/C engine과 현재 engine으로 동일한 525 tick/조건의 계획 교체·접촉 변화 fixture를 비교. 요청/평활/gate/전송이 비트 단위 동일. 장비 검증 아님. |
| 속도/힘 변화율 | 축별 10 mm/s, 40 deg/s, 30 N/s; gain 15/s 활성 | 코드·설정 근거. 실측 힘 초과 중단과 다르다. 이번 값 변경 없음. |
| 접촉/force-zero gate | base Fz의 3 N on / 1.2 N off, XY 목표 힘 0 활성 | 센서 자체는 A에서도 유지. 압축 법선 힘 교정이나 과부하 중단 기준이 아니다. 빈번한 gate만으로 비정상 판정하지 않았다. |
| Fz 명령 cap | `executor.fz_hard_limit_N=0`으로 비활성 | **측정 힘 중단값이 아니다.** 23 N 후보를 이 값에 복사하지 않았다. |
| 측정 힘·토크 초과 | 적용 가능한 per-axis/norm abort 설정·자동 검사 없음 | `on_force`는 finite/길이 검사, engine은 접촉 판정에 force[2] 사용. torque 값이 로그에 있어도 torque 중단 검사라는 뜻은 아니다. UR 자체 안전 설정/검증 기록은 별도로 확보되지 않았다. |
| ROS 피드백 감시 | pose/force/mode 수신 나이 0.2 s, tick gap 0.1 s, plan age 0.5 s, provider heartbeat 2 s; 실행 중 검사 활성 | acquisition freshness가 아니다. `on_force` invalid 즉시 stop은 running/returning에 한정. 시작 정렬 전체에 동일 보호가 보장되지 않는다. |
| 센서 stale/포화 | 원본 packet timestamp/sequence/overrange를 실행기에 전달하는 경로 없음 | 아래 Ethernet cache 재사용 문제. B/C 성공 로그는 이 고장 시나리오의 검증이 아니다. |
| Force 제어모드 상실 | running에서 `mode != 'Force' or now-mode_at > feedback_max_age_s` → stop; arming timeout 2 s | 런타임 코드와 mock 오류 주입 확인. 현장 모드 상실 시험 근거 없음. |
| 작업영역 | XY 140, 하강 85, 상승 95, 목표-현재 거리 200 mm; raw/conditioned/sent 목표 검사 활성 | 실제 TCP 자체의 공구/지그 경계 또는 시작 PTP 경로의 보호가 아니다. 현재 작업에 적용할 경계·검증 근거 미확인. 시작 PTP의 설정 속도는 공통 timed 루프와 별개이다. |
| 작업 제한 시간 | 50 s 이후 timeout → stop | 가공 품질 또는 힘 안전 기준과 다르다. |
| 중단·큐·정지 확인 | engine.stop()으로 plan 제거/재수락 금지 → Idling → PTP9D_STREAM_STOP → ACK 후 Position → fresh pose+mode 0.4 s 정지 확인 | controller `stopStream()`은 thread join 및 queue/force target 제거. ACK만으로 실제 정지라 하지 않는다. 15 s 안에 확인 못하면 stop_failed, 자동 재개 없음. |
| 현장 종료 관측 | 10회 모두 queue ACK 및 physical_hold_verified 이벤트 | 전부 operator_finish에 따른 정상 종료 관측. 센서 고장/통신 장애/과부하의 자동 보호 시험으로 대체하지 않는다. 독립적인 안전 인증 증거도 아니다. |
| 스핀들·이탈 | 자동 스핀들 정지/자동 A home·retract 없음 | 기존 현장 수동 절차 유지. finish는 정지/hold이며 접촉 해제·가공 품질 완료가 아니다. 접근·이탈 및 fault-stop 시 절차 근거 필요. |

센서의 정격, F0 배수, 과거 피크, 분석용 50 N은 중단값 산정에 사용하지 않았다. 하위 `PRECONTACT_FORCE_HOLD=15`는 제어 전환용 값이며 안전 중단값이 아니다. Q-coordinate용 내부 포화 설정도 현재 TCP-coordinate 설정의 일반 힘 제한으로 간주하지 않았다.

## 장비·통신·축 확인

로봇 PC의 실행 프로세스는 `ur10_update_rate.yaml`을 사용하고 remote launch는 `ur_type='ur10'`, robot IP 192.168.0.47이다. setup의 kinematics도 ur10, 주기 0.008 s로 UR10/CB3 설정이다. 실물 명판을 새로 확인했다는 의미는 아니다. 이 설정을 사용자에게 다시 물을 필요는 없다.

`FTGetMain.cpp`는 `FT_SOURCE_MODE=0`, 192.168.0.100:8890의 Ethernet 경로다. `FT_EtherGet.cpp`는 UDP로 받아 24 byte 이상에서 6개 network-order float32 힘·토크를 해석하고, 48 byte 이상에서 IMU를 읽는다. [AIDIN Ethernet 매뉴얼](https://emanual.oopy.io/aft200-d80-en)의 float32/big-endian 형식과 일치한다. 제조사 범위 수치로 가공 중단 기준을 만들지 않았다.

**확인된 결함:** `FT_EtherGet::FTGet()`는 `bytesReceived < 24`이면 `last_ftdata_`를 반환한다. `FTGetMain::transferData()`는 이를 필터링하고 새 publish 시각을 붙인다. `robot_motion::ftsensorCB`/`state_publisher()`는 cache된 wrench를 headerless `currentF`로 계속 재발행한다. 따라서 `e2_executor_node::on_force`의 수신 나이 검사로 원본 센서 단절을 검출할 수 없다. 실제 센서 입력 단절 실험은 하지 않았다. 원격 FT/robot 소스 6개 해시는 로컬과 모두 일치했지만, 실행 중 바이너리와 소스의 빌드 재현성까지 증명한 것은 아니다.

명령축은 controller TCP Fz, 단위 N이다. 실제 측정은 중력 보상된 base wrench이며, 하위 force controller가 실제 TCP 회전으로 변환한다. 사용자의 기존 ‘공구/TCP/지그/제어기 변경 없음’ 확인과 E1 C의 양의 명령 관례를 재사용한다. `existing_command_convention.json`은 이 **기존 근거**만 보관한다. teacher 채널을 교정된 압축 법선 힘으로 승인하거나 +23 N의 허용 범위를 승인한 기록이 아니다. 20초 unloaded 관측 역시 가공 허용 범위·고장 시 자동 정지를 검증하지 않는다.

## A 모델·검사와 로깅

A checkpoint `A_c05e06da163127f1/policy_best.ckpt`는 본학습 epoch index 499의 등록 모델이다. SHA는 `c05e06da163127f15b31e1ed77cc083688d830dc87abc857fc0be5d7031c90d6`. 실제 startup loader로 읽고 모든 weight가 등록 checkpoint와 같은지 검사했다.

state/action 6차원, force observation/history/action false, motion_only true이다. 학습 dataset도 6차원 action/normalizer이며 force target을 NaN으로 바꾼 테스트에서도 읽는 동작 target은 같다. loss는 이 6차원 동작 출력에만 적용된다. B/C 출력의 힘을 버린 모델이 아니다. 실제 validation image/pose를 넣어 128×6을 출력하고, 관측 힘을 바꿔도 출력이 정확히 같았다. transport용 3개의 0 slot과 ‘정책 힘 예측 없음’ 로그를 구분했다.

`bash /home/eunseop/nrs_imitation/scripts/e2_A_pilot_offline.sh`는 실제 A inference 뒤 동일 engine에 23 N **오프라인 가설**을 넣어 500 tick을 검사한다. 가공 marker, gate-off/on, 힘 slew, 종료 시 plan 제거를 확인한다. 피드백 pose/force는 synthetic이다. rclpy.init 및 실제 Node 생성은 코드에서 실패하도록 막았다. 실행마다 새 `results/YYYYMMDD/E2/A_offline/<id>/`에 원본 예측·피드백·명령 단계·시각·marker·종료 이유·config snapshot·후보 출처·manifest를 저장한다. 실패 기록도 남기며 실기 설정에는 쓰지 않는다.

실기 logging은 기존 wrench/TCP 20 Hz 표본, 명령 단계 125 Hz, 시각·가공 marker·중단 사유를 유지한다. headerless feedback에 acquisition 시각이 없고 controller 실제 적용 힘은 미관측이라는 한계도 유지한다. trial_stage/F0/commissioning 출처 및 승인 근거 사본을 추가한다. launch 전 차단 attempt도 기존 방식으로 보존한다.

시험: 관련 회귀검사 198개 통과. 이후 main 승격 시 결과 검토 요구 및 F0 기록 정합성 보완은 `final_targeted_tests.xml`의 관련 61개가 통과했다. mock 보호 backend에서만 미검증 품질 pilot 허용, main 차단, 보호 필드 누락 차단, A sensor NaN/stale/mode loss/abort의 공급 중단·queue ACK·fresh hold 확인을 검사했다. 승인 mock은 pytest 임시 디렉터리에만 존재하며 실제 config 해시는 그대로 유지됐다. 이는 보호 backend의 실기 검증이 아니다.

## 변경 파일과 남은 입력

- 기존 코드 수정: `e2_ablation.py`, `e2_executor_node.py`, `e2_run_context.py`, `e2_offline.py`.
- 추가 코드: `e2_pilot.py`, `scripts/e2_A_pilot_offline.py`, `scripts/e2_A_pilot_offline.sh`, `test_e2_A_pilot.py`.
- 추가 설정: `experiments/e2_A_pilot_20260927/config.json`, `commissioning_request.json`.
- 기존 master config는 코드 계약 hash/cohort만 갱신했다. 기존 common/executor/motion/sampling/calibration/F0/model 값은 그대로다. controller/센서 소스, B/C 기록, checkpoint는 수정하지 않았다. B/C와 같은 **제어 설정**을 쓰되 A preflight/phase 계약 개정은 별도 cohort로 기록한다.

남은 실제 입력은 `commissioning_request.json`에 null로 남겼다. 기존 command_direction/applicability 근거는 채웠으므로 로봇 모델·통신 형식·동일 공구 여부를 다시 요청하지 않는다.

| 정확한 필드 | 단위 / 필요한 근거 |
|---|---|
| `processing_command_fz_range_N` | signed [min,max] N. 현재 공구·지그·작업에 승인된 시운전 범위. 23 N이 범위 안인지 확인 필요. |
| `measured_force_abs_limits_N`, `measured_torque_abs_limits_Nm`, `measured_wrench_frame`, `limits_evidence` | [Fx,Fy,Fz] N, [Tx,Ty,Tz] N·m와 보호 판정 좌표계. 실측 과다하중 중단의 공학적 선정·적용 근거 및 실제 자동 중단 구현/배포 근거. |
| `acquisition_max_age_s`, `sensor_overload_evidence` | 원본 센서 수신 나이 s, 포화/overrange 감지 및 중단 근거. cached ROS 재발행의 freshness와 구별. 드라이버 경로 보완·검증 필요. |
| `workspace_evidence`, `automatic_stop_evidence`, `approach_exit_evidence` | 실제 TCP/공구 및 초기 정렬을 포함한 작업영역, 센서/제어 고장 때 명령 억제·큐 취소·실제 정지, 접근·가공 marker·접촉 해제·스핀들 취급의 현장 절차/시험 기록. B/C 정상 finish 기록만으로 충족하지 않음. |
| `status`, `approved_by`, `approved_at` | 위 실제 근거를 검토한 시운전 승인 기록. 현재 unconfirmed/null/null이며 생성하지 않았다. |

각 `*_evidence`는 실제 파일 `path`와 `sha256`를 요구한다. scope는 코드가 현재 설정에서 산출해 두었다. 값이나 승인 기록 입력만으로 현재 없는 보호 기능이 생기지 않으므로, 이 상태에서는 실기 1회 명령을 제시하지 않는다. 가공 품질 결과·main용 F0 freeze·평가 기준 프로파일·스타일러스 측정은 이번 첫 예비시험 준비의 선행 입력으로 요청하지 않는다.
