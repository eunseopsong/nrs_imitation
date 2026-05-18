# vr_calibration

## 바로 실행

orientation 오차가 크지 않을 때, 기존 `T_SA`를 유지하고 position/base calibration만 갱신:

```bash
cd /home/eunseop/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run vr_calibration vr_calibration
```

orientation 오차가 클 때, `/calibrated_pose` 기준 rotation도 다시 맞춤:

```bash
cd /home/eunseop/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run vr_calibration vr_calibration --ros-args \
  -p t_sa_mode:=update \
  -p t_sa_max_delta_deg:=180.0
```

`vr_calibration`은 UR robot EE pose와 Vive tracker raw pose를 같은 target waypoint에서 수집한 뒤, `vive_tracker_ros2` 런타임이 사용할 calibration YAML을 생성하는 ROS 2 패키지다.

이 문서는 `nrs_imitation` 전체가 아니라 `behavior_ws/src/vr_calibration` 패키지 기준으로만 정리한다.

## 입력과 출력

입력 topic:

- `/ur10skku/currentP`: `Float64MultiArray`, robot current pose `[x y z wx wy wz]`
- `/raw_pose`: `PoseStamped`, Vive tracker raw pose
- `/calibrated_pose`: `Float64MultiArray`, `T_SA` update 모드에서 현재 calibrated rotation을 읽기 위해 사용

주요 파일:

- `data/for_vr_calibration_point_v3.txt`: target waypoint 파일
- `/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/vr_calibration/ur10_ee.txt`: 캡처된 EE pose 기록
- `/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/vr_calibration/ur10_vr.txt`: 캡처된 VR pose 기록
- `/home/eunseop/nrs_imitation/behavior_ws/src/vive_tracker_ros2/yaml/calibration_matrix.yaml`: 최종 calibration YAML

생성되는 YAML 행렬:

- `T_AD`: Vive world/raw frame을 robot base frame으로 올리는 base calibration
- `T_BC`: robot EE에서 tracker/tool frame까지의 offset
- `R_Adj`: VR point cloud와 robot point cloud의 미세 기울어짐/축 정렬 보정
- `T_FIX`: z-plane residual을 줄이기 위한 left-multiplied rigid correction
- `T_CE`: legacy constant offset
- `T_SA`: orientation display/alignment용 right-multiplied rotation correction

## 기본 실행

빌드:

```bash
cd /home/eunseop/nrs_imitation/behavior_ws
colcon build --packages-select vr_calibration
source install/setup.bash
```

캘리브레이션 실행:

```bash
ros2 run vr_calibration vr_calibration
```

현재 기본값은 다음과 같다.

```text
t_sa_mode = keep
radj_sample_count = 0        # 0 또는 음수면 전체 captured sample 사용
capture_hold_time_s = 2.0
capture_window_s = 0.5
capture_min_clean_samples = 20
vr_capture_age_s = 0.2
max_capture_sync_dt_s = 0.05
capture_max_vr_std_mm = 10.0
z_fix_enable = true
max_calib_position_rms_mm = 50.0
```

orientation까지 새로 맞춰야 하면 `T_SA` update를 켠다.

```bash
ros2 run vr_calibration vr_calibration --ros-args \
  -p t_sa_mode:=update \
  -p t_sa_max_delta_deg:=180.0
```

## 캡처 로직

노드는 waypoint 파일에서 `holding_time_s > 0`인 point만 target으로 사용한다. 각 target마다 robot이 다음 조건을 만족하면 hold 상태로 들어간다.

- position error <= `pos_enter_mm_`
- orientation error <= `ori_enter_deg_`
- robot linear velocity <= `vel_thresh_mms_`
- robot angular velocity <= `angvel_thresh_dps_`

패치 이후에는 hold가 끝나는 순간의 단일 샘플을 바로 쓰지 않는다. hold 중 다음 조건을 만족하는 clean sample만 buffer에 쌓는다.

- `/ur10skku/currentP`가 fresh
- `/raw_pose`가 `vr_capture_age_s` 이내
- `abs(currentP_time - raw_pose_time) <= max_capture_sync_dt_s`
- robot이 target region 안에 있음
- robot이 stopped 상태임

그 뒤 clean sample이 최소 `capture_min_clean_samples`개 이상이고, buffer 시간 폭이 `capture_window_s` 이상이면 평균 pose를 하나 만든다.

- robot pose: clean sample 평균
- VR position: clean sample 평균
- VR orientation: quaternion sign-align 평균
- VR position std가 `capture_max_vr_std_mm`를 넘으면 캡처를 보류

캡처 로그 예:

```text
[CLEAN_CAPTURE] averaged 42 samples over 0.510s | dist=0.03mm ang=0.01deg vr_std=1.25mm
[CAPTURE] target 12/32 ...
```

## Calibration 계산 순서

캡처된 sample은 내부적으로 다음 의미를 가진다.

```text
T_AB[i] = robot base(A) -> EE(B)
T_DC[i] = VR world(D) -> tracker(C)
```

전체 계산 흐름:

```text
1. clean sample set 수집
2. R_Adj 계산
3. T_DC_adj[i] = T_Adj * T_DC[i]
   where T_Adj rotation = R_Adj.T
4. hand-eye solve로 T_BC 계산
5. 각 sample에서 T_AD_i = T_AB[i] * T_BC * inv(T_DC_adj[i]) 계산
6. T_AD_i 평균으로 T_AD 생성
7. T_FIX 계산
8. runtime-chain residual 검증
9. YAML 저장
```

`radj_sample_count=0`이면 `R_Adj` 계산에 captured sample 전체를 사용한다. 특정 개수만 쓰고 싶으면 양수로 지정한다.

```bash
ros2 run vr_calibration vr_calibration --ros-args \
  -p radj_sample_count:=32
```

## Runtime pose 의미

`vr_calibration`은 YAML만 만든다. `/calibrated_pose`를 어떤 의미로 publish할지는 `vive_tracker_ros2/vive_tracker_node.py`의 `tool_correction_mode`가 결정한다.

```text
tool_correction_mode=none
  -> calibrated tracker/world pose publish
  -> EE와 tracker 사이 offset이 position에 남아 있음

tool_correction_mode=t_bc
  -> T_BC inverse를 적용해서 EE/TCP pose publish
  -> robot currentP와 position이 거의 같아지는 것이 정상

tool_correction_mode=t_ce
  -> legacy T_CE offset 사용
```

현재 기본값은 `none`이다. 따라서 아무 인자 없이 `vive_tracker_node`를 실행하면 `/calibrated_pose`는 robot EE pose가 아니라 tracker pose로 나온다. EE/TCP pose가 필요하면 명시적으로 `t_bc`를 켠다.

```bash
ros2 run vive_tracker_ros2 vive_tracker_node --ros-args \
  -p tool_correction_mode:=t_bc
```

## 확인 포인트

캘리브레이션이 정상적으로 끝나면 다음 로그를 확인한다.

```text
[R_ADJ_DONE] multi-point position fit using N/N samples
[T_FIX] z-plane rigid fix computed ...
[CALIB_VALIDATE] runtime-chain position fit: rms=... max=...
[YAML_SAVED] ...
```

`T_SA` update를 켠 경우에는 다음 로그가 있어야 한다.

```text
[T_SA_DONE] ...
[T_SA] Pre-capture update done.
```

다음 로그가 반복되면 clean sample 조건이 너무 빡빡한 것이다.

```text
[WAIT_CLEAN_CAPTURE] clean_samples=...
[WAIT_CLEAN_CAPTURE] VR position std ... exceeds ...
```

이 경우 먼저 `/raw_pose` publish rate와 tracking 상태를 확인하고, 필요하면 `max_capture_sync_dt_s`, `capture_min_clean_samples`, `capture_max_vr_std_mm`를 완화한다.
