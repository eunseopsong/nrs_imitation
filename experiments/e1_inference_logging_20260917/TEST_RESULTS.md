# Offline 검증 결과

## 2026-09-17 실제 실행 로그 조사 후 수정

17:55:57 실행의 `python3_27008_1789635357850.log`에서 `[METRICS] setup failed ... no attribute 'list_parameters'`를 확인했다. 이전 mock이 실제 ROS Humble Node에 없는 메서드를 제공하여 이 오류를 놓쳤다. 파라미터 열거를 설치된 rclpy의 `get_parameters_by_prefix("")`로 수정했다. 테스트도 실제 Node의 열거 메서드를 바인딩하여 사용하고, 점이 포함된 이름과 빈 RPM까지 전체 metadata 보존을 확인한다.

수정 후 `test_execution_metrics.py`: **12 passed in 3.12s**. ON/OFF 로깅 유무 출력 비교, 실제 rclpy 파라미터 열거, `service_call`/`service_stream` launch 전달 및 GradCAM 비활성화 검증 포함. `git diff --check` 통과. 설치 경로가 수정된 source를 참조하는 것도 확인했다. 로봇 노드 실행이나 서비스 호출은 하지 않았다. 수정 전 실패한 실행의 센서 로그는 복구되지 않으며, 수정 후 실제 수신은 다음 운영자 실행에서 확인해야 한다.

기존 성공 모델과 E1 ON의 체크포인트 비교: 모델 key/shape는 동일, 336개 state tensor 중 174개 값이 다름. best epoch 필드는 19 / 39. 데이터·정규화 수치는 같고 policy_config는 명시적인 `use_force_observation=True` 추가만 다르다. 따라서 같은 학습 구성의 재학습 결과이며 동일한 가중치 파일은 아니다.

9월 10일 해당 모델의 추론 로그 12개에서 모두 `PTP9D_STREAM_START`, `GRADCAM enable=0`을 확인했다. 최근 E1은 구간 대기 방식인 `service_call`, `GRADCAM enable=1`이었으며 README의 ON/OFF 명령을 과거 실행 방식으로 정정했다. 새 계획 생성 간격 설정은 두 실행 모두 120 steps / 30 Hz = 4초였다. GPU 계산 시간 자체를 별도로 측정한 결과는 아니다.

## 최초 검증 기록 (위 후속 수정 이전)

2026-09-17, `/usr/bin/python3` 3.10 / ROS Humble import 환경 / torch 2.5.1+cu121. 로봇 Node 생성·ROS spin·service 호출·spindle 명령은 실행하지 않았다. 테스트의 observer/transport는 mock이며 실제 inference 함수·normalizer·Flow sampler·SRF helper·명령 계산을 호출했다.

검증 결과: **13 passed** (새 logger 테스트10개 + 기존 recorder_sync 회귀3개, 약3초).

| 항목 | 결과/범위 |
|---|---|
| OFF 전체 history force perturbation | 실제 `_on_force` → `_on_infer_timer` 정규화/masking → eval Flow sampling, force history 전체를 다른 값으로 변경, noise/cache 동일 초기화: 정책 9D 출력 allclose(atol=1e-6, rtol=0). raw 센서 Fx/Fy/Fz/Tx/Ty/Tz는 원값으로 기록. |
| ON conditioning | 실제 Flow `_condition` 인자에서 현재 힘/전체 history 변화 도달 확인. action이 반드시 달라야 한다는 기준은 사용하지 않음. |
| 로깅 유무 | ON/OFF 각각 동일 입력·초기 상태·seed에서 실제 Flow 출력 및 `_publish_cmd` payload **array_equal**. torch/numpy/Python 전역 RNG 상태 불변. 목표 Fz 출력 비제로 보존, 기존 전송 Fx/Fy=0 유지. |
| pose ≠ command | feedback x=420과 다른 command x를 별도 파일/stage에 저장. rotvec→xyzw quaternion, 부족한 필드/NaN을 0으로 보충하지 않음. |
| 서로 다른 수신률/clock | wrench10행, pose2행 fixture. 힘 source stamp ns 보존, header 없는 pose source stamp는 빈칸. Fn unknown, 음수 Fz=-7 보존. |
| reference·이미지 | 첫 reference 고정 후 원래 객체를 바꿔도 roi 불변. 최초/최종 RGB 저장. 인코딩·모든 파일 쓰기가 writer thread에서만 수행됨. |
| queue/충돌/flush | queue size1에 writer를 잠시 막고100행 enqueue:99 drop/1write 명시. 같은 tag도 다른 폴더, 정상 close drain. 크기0(unbounded)는 거부. |
| 저장 오류 | mock disk-full 주입: write error warning/artifact, 기존 raw/metadata 보존, 제어 예외 없음. 실제 디스크 장애 시험은 아님. |
| 서비스 경로 | 실제 `_ptp9d_advance`의 mock request를 logging ON/OFF 비교. 모든 waypoint/원 payload/인덱스 동일. 응답 success도 physical completion=unknown. 실제 controller 호출 없음. |
| subscriber | 독립 observer Node/executor를 mock하여 실제 소스에서 확인한6 topic 및 BEST_EFFORT/depth100 확인. 실제 DDS 연결/수신 시험 아님. |
| RPM | 수동1234는 **테스트 값**. setpoint/manual만 기록, measured_rpm=null, spindle 명령 없음. nan/inf/음수/문자열 값 거부. 실험 RPM을 정한 것이 아님. |
| launch | clean include의 실제 parameter 평가로 ON/OFF bool, 로그 옵션, RPM string, stain/canon false, flow steps10/chunk128 전달 확인. Node.execute/launch 실행 없음. `ros2 launch ... --show-args`도 확인. |
| 재로딩/불완전 종료 | 모든 파일 run_id 일치. summary를 다른 이름으로 보존한 crash 모사에서도 raw CSV/JSONL 검사 가능하고 summary 부재 보고. |
| 기존 회귀 | `test_recorder_sync.py` 3개 통과. 기존 동기화 구현 수정 없음. |

별도 검사: Python compileall, `git diff --check` 통과. 설치 환경에서 새 logger 모듈 import 확인(source-linked build 경로). `scripts/check_inference_log.py`가 SYNTHETIC ON/OFF 파일을 읽어 force10/pose5/command1, 알려진5ms/10ms grid, drop0/write error0, middleware 손실 unknown과 부족한 분석 항목을 보고함. 예시 clock/rate는 합성 값이지 실제 센서 성능이 아니다.

최신 예시 run_id:

- `SYNTHETIC_OFF_20260917T163903_1789630743700352795_krjsuhiz`
- `SYNTHETIC_ON_20260917T163903_1789630743728988069_cccar3kz`

각 파일은 이 디렉터리의 `examples/<run_id>/`에 있다. 이전 예시도 보존했다.

초기 실행의 pytest 외부 plugin 충돌은 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`로 격리했다. 테스트 harness의 resize 타입, sampler 다중 conditioning call 비교, LaunchConfiguration 정규화는 수정 후 재실행 통과했다. 생산 sampling/config는 바꾸지 않았다.

## 수행하지 않은 것 / 실험 전 확인

- 실제 센서 수신·실제 TCP/FK calibration·배포된 외부 controller 소스/QoS 및 nominal 2000/100/125Hz의 실측 수신률.
- 생산 DINOv3 ON/OFF checkpoint의 로봇/GPU 실행, 제어 deadline/GIL/CPU/GPU 부하/장시간 저장 성능.
- Ethernet sensor scale/torque 기준점, gravity 보정의 active flag 및 영점 epoch, acquisition timestamp와 host clock 동기화.
- 현재 유효한 표면 법선·압축 부호·TCP/접촉점 offset·고정 평가 ROI·회전 ON/OFF 상태·실측 RPM·footprint·Preston 계수.
- targetF는 controller 보고값이지 내부 최종 commanded_Fd 확인이 아니다. 실제 도달/가공 완료 신호도 없다.

따라서 “센서 로깅까지 검증 완료” 또는 “Preston 제거량 계산 가능”으로 판단하면 안 된다. README의 명령으로 운영자가 실행한 뒤 각 run의 source별 gap/drop/unknown을 확인하고 필요한 수동 calibration/가공 구간을 확보해야 한다. 새 학습이나 checkpoint 생성은 하지 않았다.
