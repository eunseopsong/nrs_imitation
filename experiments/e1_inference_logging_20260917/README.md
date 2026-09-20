# E1 실행 로그 (2026-09-17)

ON/OFF는 **힘 관측값을 제공해 학습한 체크포인트 / 힘 관측값을 가리고 학습한 체크포인트**를 뜻한다. 각각의 학습 설정에 맞춰 추론에서도 ON은 measured Fx/Fy/Fz history를 정책에 제공하고, OFF는 **정규화 후 qpos[6:9]와 모든 force history를 0**으로 만든다. 둘 다 위치·자세·목표 힘 action을 출력한다. 기준은 성공한 `20260910_90deg_rel_single/20260910_1606`의 90deg 데이터·얼룩 중심 상대 위치·모델 구성이다. 로깅은 정책으로 값을 돌려주지 않는다.

실제 로봇/센서로 검증하지 않았다. 아래 launch는 로봇을 움직일 수 있으므로 운영자가 기존 준비·안전 절차 후 **한 조건씩** 실행한다. agent는 실행하지 않았다.

## 실행

```bash
cd /home/eunseop/nrs_imitation
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source behavior_ws/install/setup.bash
E1_CKPT=/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/e1_force_observation_20260916
read -r -p '동일하게 사용할 실제 설정 RPM (미확인이면 Enter): ' E1_RPM
E1_RPM_ARGS=()
if [[ -n "$E1_RPM" ]]; then
  E1_RPM_ARGS+=("metrics_rpm_setpoint:=$E1_RPM")
fi
```

`E1_RPM`은 **로그 전용 수동 설정값**이며 spindle 명령/실측 RPM이 아니다. 숫자를 추정하지 않는다. 빈 값이면 위 Bash 배열은 RPM 인자를 생략하고, launch 기본값에 따라 null+경고로 기록한다. `metrics_rpm_setpoint:=`처럼 빈 값을 CLI 인자로 직접 전달하면 ROS 2가 `malformed launch argument`로 거부한다. 실측 없이 일정 회전으로 해석할 경우만 `metrics_rpm_assumed_constant:=true`를 명시한다. 기본 false. RPM이 실제로 동일했는지는 운영자가 확인해야 한다.

ON:

```bash
ros2 launch nrs_imitation inference_clean_single_cam.launch.py \
  ckpt_dir:="$E1_CKPT/on/20260916_1531" use_force_observation:=true \
  use_stain_mask:=false stain_canon_enable:=false inference_mode:=service_stream gradcam_enable:=false \
  metrics_log_enable:=true metrics_run_tag:=E1_on_r01 \
  metrics_repeat_id:=r01 \
  "${E1_RPM_ARGS[@]}" \
  overlay_record_enable:=true removal_viz_enable:=false \
  visualize_flow_vector:=false visualize_modality_importance:=false
```

OFF (ON 종료 후):

```bash
ros2 launch nrs_imitation inference_clean_single_cam.launch.py \
  ckpt_dir:="$E1_CKPT/off/20260916_1531" use_force_observation:=false \
  use_stain_mask:=false stain_canon_enable:=false inference_mode:=service_stream gradcam_enable:=false \
  metrics_log_enable:=true metrics_run_tag:=E1_off_r01 \
  metrics_repeat_id:=r01 \
  "${E1_RPM_ARGS[@]}" \
  overlay_record_enable:=true removal_viz_enable:=false \
  visualize_flow_vector:=false visualize_modality_importance:=false
```

2026-09-17 실행 로그 조사 후 위 명령의 실행 모드를 정정했다. 성공 기준인 9월 10일 `20260910_90deg_rel_single` 실행 로그들은 `PTP9D_STREAM_START`(연속 `service_stream`)와 `gradcam_enable=0`을 사용했다. 앞서 안내한 `service_call`은 구간 완료 응답을 기다리는 별도 실행 방식이다. ON/OFF 모두 위 두 옵션을 동일하게 사용한다. 2026-09-20에 polishing single-cam launch와 실행 entrypoint의 Grad-CAM 기본값도 `false`로 맞췄다. `inference_mode:=service_stream`은 위 명령처럼 명시한다.

E1 ON은 같은 데이터·모델 구성으로 재학습한 체크포인트이며 9월 10일 파일과 동일하지 않다. `policy_best.ckpt`의 저장된 epoch는 기존 19 / E1 ON 39이며, 모델 state 336개 중 174개 tensor 값이 다르다. 정규화 수치와 상대좌표 설정은 같다. 기존 성공 체크포인트 재현과 E1 ON/OFF 비교는 실행 태그로 구분한다.

시편 ID는 ON/OFF 모두 기본값 `default`로 기록하므로 지정할 필요 없다. 다음 반복은 `E1_on_r02`/`E1_off_r02`, `r02`. 실시간 뷰어는 기본으로 끄고, `overlay_record_enable:=true`로 flow-vector와 modality 영상을 `~/Videos/Screencasts/*.webm`에 자동 저장한다. 녹화는 두 영상 토픽의 첫 프레임을 모두 받은 뒤 시작하고, 종료 시 파일을 마무리한다. 창을 열지 않아도 녹화되며 영상 생성·인코딩 연산은 계속 수행된다. `removal_viz_enable:=false`는 별도의 Preston 결과 생성을 끈다. 이 옵션들은 로봇 제어 설정이 아니다.

현재 설치는 source-linked이다. 다른 설치에서 새 모듈이 import되지 않으면 `cd behavior_ws` 후 기존 절차대로 `colcon build --symlink-install --packages-select nrs_imitation` 및 `source install/setup.bash`가 필요하다.

## 저장·점검

출력의 `[METRICS] run directory:`를 복사한다. tag가 같아도 timestamp(ns)+임시 디렉터리 난수로 **새 폴더만** 만들며 이전 로그/체크포인트는 덮어쓰지 않는다.

```text
logs/inference_metrics/<tag>_<timestamp_ns>_<unique>/
  metadata.json        # 실제 node parameters/resolved 설정, checkpoint 식별, config
  wrench.csv           # source별 실제 수신 힘/토크, OFF에서도 비제로 원값
  tcp_pose.csv         # 실제 joint-feedback FK TCP; 명령 아님
  commands.csv         # prediction / node_sent / controller_reported_target
  events.jsonl         # inference/phase/reference/send/response/error/drop
  roi.json             # 최초 고정 reference, 초기/종료 image 정보
  legacy.csv           # 기존 비교 CSV 열 + 시간/age; 최신값 snapshot일 뿐
  logger_status.json   # 실행 중 enqueue/write/drop/error
  summary.json         # 종료 drain 결과; 강제 종료시 없어도 raw 파일은 읽을 수 있음
  snapshots/           # 초기·종료 RGB, 기존 mask가 수신된 경우 최초 mask
  artifacts/           # normalizer·소스·기존 calibration/config 사본 및 SHA256
  code_changes.patch   # 작업트리 변경 식별; 미추적 신규 logger도 artifacts에 복사
```

```bash
read -r -p '[METRICS]가 출력한 실행 폴더 절대경로: ' E1_RUN_DIR
watch -n 2 "wc -l '$E1_RUN_DIR/wrench.csv' '$E1_RUN_DIR/tcp_pose.csv' '$E1_RUN_DIR/commands.csv'; cat '$E1_RUN_DIR/logger_status.json'"
# 종료 후 (ROS 환경 불필요)
/usr/bin/python3 scripts/check_inference_log.py "$E1_RUN_DIR"
```

검사기는 source별 rate/median·maximum gap, 결측, NaN/Inf, 시각 역행, source stamp 반복, drop/error, run_id 일치를 보고한다. **수신 이벤트 수 ≠ 센서 취득 횟수**(`/currentF`는 재발행). middleware 손실은 unknown이다. gap/stale 합격 기준은 기본적으로 설정하지 않는다. 실제 관측 주기와 실험 허용오차를 정한 뒤 `--max-gap-ms`(수신 시각 기준, 모든 source 공통), `--max-age-ms`(legacy cached 값의 동일 host monotonic age)에 명시한다. 보간·삭제하지 않는다. 종료 코드 2는 로그 오류/명시 기준 위반이며, 0도 Preston 분석 준비 완료를 뜻하지 않는다.

주기가 다른 source는 `--source-max-gap-ms SOURCE=MS`를 반복 지정해 각각 판단할 수 있다(예: SOURCE는 실제 `/ur10skku/ftdata`, MS는 운영자가 정한 허용값). 원본 acquisition age를 모르는 wrench/pose의 stale 판정은 여전히 null이다.

## Offline 테스트·예시

```bash
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source behavior_ws/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/behavior_ws/src/nrs_imitation:$PWD/behavior_ws/src/stain_relative_frame:${PYTHONPATH}" \
  /usr/bin/python3 -m pytest -q behavior_ws/src/nrs_imitation/test/test_execution_metrics.py
/usr/bin/python3 scripts/make_inference_log_example.py --force-observation OFF
/usr/bin/python3 scripts/make_inference_log_example.py --force-observation ON
```

예시의 `SYNTHETIC_*`는 실제 실행·가공 결과가 아니다. 테스트는 실제 callback·정규화·history masking·Flow sampler·전송 계산을 작은 untrained ResNet Flow로 실행하며 ROS transport만 mock한다. 생산 DINOv3 checkpoint의 하드웨어 실행/실시간 주기 검증은 아니다. pytest의 설치된 외부 plugin 버전 충돌 때문에 autoload를 끈다.

추가 로그 전용 옵션: `metrics_queue_size`(양수, 기본 8192), `metrics_snapshot_enable`(기본 true), `metrics_context_file`(JSON 문서의 수동 calibration/시편/공구 메모를 `operator_context`에 보존; 제어/변환에 적용하지 않음). 유효 접촉 면적/반경, 외경, Preston 계수, 현재 표면 법선·압축력 부호, 접촉점 offset, 회전 상태는 확인 전 null이다. 수동 자료는 출처/단위/좌표계/측정 날짜를 함께 남긴다.

상세 경로·schema·미확인 사항: [DATA_PATHS.md](DATA_PATHS.md). 테스트 결과: [TEST_RESULTS.md](TEST_RESULTS.md).
