# E2 A 보호 기능 패치 — 실기 준비 미완료

2026-09-27. 실행기 패치는 현재 `behavior_ws`에서 사용되는 소스에 적용했다. 로봇 측 패치는 `experiments/e2_A_protection_20260927/driver_src/`에서 별도 빌드했다. 실행 중인 로봇 PC와 기존 `dev_ws` 소스·설치는 변경하지 않았다. 로봇, 스핀들, homing, zeroing을 실행하지 않았다.

## 해소한 소프트웨어 차단

- `e2_pilot.py::installed_protection_errors()`가 보호 기능 미구현을 이유로 항상 반환하던 네 가지 오류를 실제 기능 검사로 교체했다. 승인 기록만으로 준비 완료가 되지 않는다.
- 새 `e2_protection.py::ProtectionMonitor`는 명시적으로 주어진 실측 힘·토크 중단값, 센서 원시값 허용 범위, 원본 수신 갱신, 실제 TCP 경계를 검사한다. 중단값·timeout 기본값이나 F0에서 계산하는 값은 없다.
- `e2_executor_node.py::tick()`의 `waiting_feedback`에서 원본 센서와 제어기 사용 데이터가 모두 확인되기 전에는 초기 명령도 보내지 않는다. ready/초기 정렬, arming, running, stopping에 보호 검사를 연결했다. 오류는 latch하며 명령 공급 중단 → Idling → 기존 큐 취소 → ACK 뒤 Position → fresh feedback의 정지 확인 절차를 유지한다. ACK 누락·정지 확인 실패는 정상 종료로 기록하지 않는다.
- staged `FT_EtherGet.cpp/.hpp`, 새 `FT_Acquisition.hpp`, `FTGetMain.cpp`: 실제 UDP 수신에만 sequence/수신 시각을 갱신하고 raw 6축을 별도로 전달한다. 짧은/비유한 패킷 이력을 latch하여 다음 정상 패킷이 오류를 숨기지 못한다. cache 재사용은 zeroing의 새 표본으로 세지 않는다.
- staged `robot_motion.cpp/.hpp`: `/ur10skku/currentF_provenance`에 제어기가 사용하는 6축 값과 원본 수신 시각을 전달한다. 주기적 재발행으로 원본 시각을 갱신하지 않는다. 기존 6축 `/currentF`와 제어 계산은 유지한다.
- `e2_ablation.py`, `e2_run_context.py`, pilot/master config: 소스 계약·별도 cohort를 갱신하고 향후 시도에 보호 코드, 필요한 driver 소스, 후보 F0와 commissioning 근거를 함께 보존한다. 실패·시작 전 차단을 삭제하지 않는다.

예비시험은 기존 분리된 경로대로 `F0_frozen=false`, `quality_validation=pending`을 유지한다. 본 실험은 실제 예비시험 결과 검토와 F0 확정 없이는 차단된다. `physical_validation` 등의 보고서 플래그를 승인값으로 바꾸지 않았다. 차단 조건 전체는 `status.json`에 저장했다.

## 검사 결과와 적용 범위

- Python 회귀검사 **287 passed** (`regression_tests.xml`), 최종 driver hash 갱신 후 관련 검사 **114 passed** (`final_targeted_tests.xml`). mock 승인·수치는 pytest 임시 디렉터리에만 존재한다.
- raw force/torque 양·음 방향 한계, stale/cache 재발행, 불량 패킷, producer 재시작, 원본 시각 역행, 실제 TCP 이탈과 모든 시작/실행/정지 단계 오류를 검사했다. 실제 executor callback/state machine에 연결한 mock이며 ROS context를 열지 않는다.
- 순수 C++ 가짜 datagram 검사 통과. 두 ROS C++ 패키지 격리 빌드 완료: `driver_build_final.log`, `install/`. **빌드 성공은 배포나 실기 검증이 아니다.** 기존 클래스 초기화 순서 경고 등은 build log에 보존했다.
- 실제 등록 A checkpoint를 로드한 오프라인 검사 통과. 입력·출력 6차원, 힘 observation/history/output/loss 채널 없음, 가짜 측정 힘을 바꾸어도 같은 motion 출력, 공통 gate/30 N/s 변화율 제한 보존. 결과는 `results/20260927/E2/A_offline/173440_4bbb8f2f/attempt.json`에 있다. 이 검사는 로컬 checkpoint로 완료되었으며 선택적 Hugging Face import의 기존 OpenSSL 경고는 log에 남겼다.
- 9/26 결과·영상·checkpoint 등 **2,526개 파일 SHA 동일**, 기존 local driver 원본도 보존 (`preservation.json`). B/C의 timed_topic·평활화·계획 연결·속도·가속·gate·힘 변화율 값은 그대로이며 보관 엔진과 출력 비교도 통과했다. **A에 추가한 보호 revision은 과거 B/C와 동일 안전 조건이라고 표시하지 않는다.**

원본 시각은 센서 내부 ADC 시각이 아니라 **호스트 UDP 수신 시각**이다. 두 PC의 ROS 시계가 맞아야 하며 미래/오래된 시각은 차단한다. 유효한 UDP가 계속 도착해도 센서 내부 ADC가 고장났는지는 이 정보만으로 보증할 수 없다. raw per-axis 범위 검사만으로 복합하중·센서 포화 보호를 검증 완료로 판단하지 않는다. 실제 TCP 경계는 기존 140/85/95 mm를 시작 피드백 기준으로 검사하며 공구/지그의 전체 충돌 형상이나 PTP 경로 검증을 대체하지 않는다. TCP 좌표계의 토크는 축 회전만 적용하며 기준점은 센서 원점이다. 독립된 하위 안전 정지나 통신 완전 단절 시 물리 정지를 새로 보증하는 패치가 아니다.

## 실행 가능한 오프라인 명령

```bash
cd ~/nrs_imitation
bash scripts/e2_A_pilot_offline.sh
```

후보 23 N과 등록 A 모델을 실제 로봇 명령 없이 검사한다. 실기 승인이나 F0 확정은 하지 않는다.

## 남은 실기 입력·근거 — 한 번에 확인할 목록

`experiments/e2_A_pilot_20260927/commissioning_request.json`의 아래 값은 미확인 상태 그대로다. 수치를 사용자에게 추정하도록 요청하는 것이 아니라, 현장 책임자의 적용 가능한 운전 기준/시험 기록이 필요하다. 각 `*_evidence`는 실제 기록의 `path`와 `sha256`이다.

| 정확한 필드 | 단위 / 필요한 근거 |
|---|---|
| `processing_command_fz_range_N` | TCP Fz의 부호 있는 [최소, 최대] N. 현재 공구·지그·작업의 허용 시운전 범위이며 후보 23 N이 그 안에 있어야 한다. |
| `measured_force_abs_limits_N`, `measured_torque_abs_limits_Nm`, `measured_wrench_frame`, `limits_evidence` | [Fx,Fy,Fz] N, [Tx,Ty,Tz] N·m의 자동 중단 기준과 좌표계/토크 기준점. 장비·작업에 적용할 선정 및 확인 근거. F0·센서 정격·과거 peak·분석용 50 N에서 생성하지 않는다. |
| `sensor_raw_force_abs_limits_N`, `sensor_raw_torque_abs_limits_Nm`, `sensor_overload_evidence` | 영점/필터/중력 보정 전 센서축 3개씩 N/N·m. 포화·과부하 및 복합하중을 포함한 감시 방식의 적용 근거. |
| `acquisition_max_age_s`, `sensor_driver_deployment_evidence` | 실제 UDP 수신 허용 나이 s. staged driver가 로봇 PC에서 실행 중임을 확인한 소스/바이너리/프로세스·두 provenance topic 및 시계/통신 지연 확인 기록. |
| `workspace_evidence`, `automatic_stop_evidence`, `approach_exit_evidence` | 실제 TCP·공구/지그 및 초기 정렬 경계, 고장 시 명령 억제·큐 취소·실제 정지 확인, 접근/가공 시작·종료 marker/이탈/스핀들 절차. 이전 정상 finish 기록만으로 대체하지 않는다. |
| `status`, `approved_by`, `approved_at` | 위 기준과 근거에 대한 실제 시운전 승인. 현재 `unconfirmed/null/null`. |

검토용 로봇 변경은 `robot_driver.patch`, 각 원본/변경 SHA는 `driver_sources.json`에 있다. 로봇 드라이버 시작에는 기존 자동 zeroing이 포함되어 있어 자동 배포·재시작하지 않았다. 배포와 위 실제 근거가 확보되기 전에는 A 실기 명령을 제공하지 않는다. 23 N은 계속 후보이며 실기 안전 확인값이 아니다.

격리 `install/`은 로컬 컴파일 확인용이다. 그 안의 config를 원격 장비의 현재 교정 설정으로 간주하지 않는다. 실제 배포 검토 대상은 여섯 소스 파일의 패치이며, 원격 장비의 현재 calibration/config와 실행 환경은 별도로 보존·대조해야 한다.
