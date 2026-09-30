# E2 A: B/C 공통 설정 및 F0 검사 분리 확인

실행 로직과 수치 설정은 변경하지 않았다. 설정에 F0 검사의 적용 대상과 계산 출처를 명시하고 mock 회귀검사를 추가했다.

- F0: 기존 controller TCP +Fz, 23 N 후보. A 학습 38개 교시의 시간 평균 23.0182261091 N을 원본에서 재현했다. 법선력 보정·실기 검증값은 아니다.
- `processing_command_fz_range_N=[23,23]`은 `e2_pilot.commissioning_errors()`에서 `config.f0.command_fz_N` 하나만 검사한다. `E2Executor.__init__()`에서 한 번 준비한 뒤 `tick()`은 가공 중 F0, 그 외 0 N을 공통 실행기에 전달한다.
- 접근·이탈/해제 요청 0 N, 가공 23 N, gate와 slew를 거친 중간 명령값은 F0 선택 범위와 별개다. 감쇠 중 송신값이 즉시 0 N이 되지는 않는다. [23,23]을 센서값에 적용하지 않는다.
- 9/26 B 5회+C 5회의 보관 config와 executor, 후처리, sampling, 작업영역, 접근·이탈 절차, 시간 상한이 모두 같다. timed_topic/125 Hz, gain 15/s, 축별 10 mm/s 및 40 deg/s, 힘 30 N/s, 위치 가속 25 mm/s², MA35, 연결 0.5 s, 접촉 ON/OFF 3.0/1.2 N을 유지했다.
- `fz_hard_limit_N=0`도 보관 config와 같다(기존 명령 cap 비활성). 이 값과 F0 범위를 측정 하중 중단값으로 바꾸지 않았다. 보호 코드 revision 전체가 과거 B/C와 동일하다는 의미는 아니다.
- 힘/토크 중단, raw 센서 감시, 피드백·제어모드·작업영역 감시와 명령 중단/큐 취소/정지 확인 조건은 유지했다. 관측 peak, 센서 정격, 분석용 50 N으로 값을 만들지 않았다.
- 실제 보호 수치/근거와 driver 활성화는 미확정이다. A 실기 preflight는 이를 이유로 차단되며, 품질 pending 또는 F0_frozen=false 때문이 아니다.

## 검증

140 passed. ROS context·로봇·스핀들을 실행하지 않았다. mock 승인값은 임시 디렉터리에만 만들고 실제 config와 commissioning 기록의 SHA 불변을 검사했다.

| Mock 단계 | tick 수 | 요청 Fz (N) | 송신 Fz 범위 (N) |
|---|---:|---:|---:|
| approach | 12 | [0.0] | 0 ~ 0 |
| approach_contact | 4 | [0.0] | 0 ~ 0 |
| processing | 160 | [23.0] | 0.24 ~ 22.9998108 |
| contact_gate | 6 | [23.0] | 22.3881415 ~ 22.7598108 |
| retract | 220 | [0.0] | 6.62403508e-08 ~ 22.2215645 |
| release | 12 | [0.0] | 1.4286133e-08 ~ 5.82915087e-08 |

finish와 abort 각각에서 명령 공급 중단 → Idling → PTP9D_STREAM_STOP → ACK 뒤 Position → fresh stationary feedback 확인을 검사했다. 종료 명령 후 추가 cmdMotion은 없었다. 실제 물리 정지를 시험했다는 뜻은 아니다.

F0_frozen=false, quality_validation=pending을 유지했다. 신규 실기 승인이나 품질 승인 이력을 생성하지 않았다.

## 남은 미확정 필드

- `measured_force_abs_limits_N`: three approved positive per-axis limits required
- `measured_torque_abs_limits_Nm`: three approved positive per-axis limits required
- `sensor_raw_force_abs_limits_N`: three approved positive per-axis limits required
- `sensor_raw_torque_abs_limits_Nm`: three approved positive per-axis limits required
- `measured_wrench_frame`: explicit protection frame required
- `acquisition_max_age_s`: approved source-acquisition timeout required
- `limits_evidence`: pinned evidence path + sha256 required
- `sensor_overload_evidence`: pinned evidence path + sha256 required
- `workspace_evidence`: pinned evidence path + sha256 required
- `automatic_stop_evidence`: pinned evidence path + sha256 required
- `approach_exit_evidence`: pinned evidence path + sha256 required
- `sensor_driver_deployment_evidence`: pinned evidence path + sha256 required

## 파일

- `config_changes.patch`: 설정만의 전후 diff.
- `changes.patch`: 설정 및 테스트 전체 diff.
- `settings_basis.json`: 교시 계산 재현값·입력 출처 SHA·보관 B/C 10회 대조.
- `tests.xml`, `tests.log`, `verification.json`: mock 결과와 보존 확인.
