# 실제 경로 및 schema 조사

조사 기준 commit: `0c2cff6cff0268a289fbe9ed396165faabc58bd0` + 기존 사용자/E1 변경을 보존한 작업트리. 상위/저장소에서 적용되는 AGENTS.md는 발견되지 않았다. root/behavior_ws/nrs_imitation README를 확인했다. 기존 데이터셋, 체크포인트, 과거 CSV와 사용자 학습 로그를 수정하지 않았다.

## 데이터 경로 (모두 소스 코드 기준; 실행중 publisher/배포 설정 미검증)

공통 prefix: `/home/eunseop/dev_ws/src/y2_ur10skku_control/`는 **읽기만** 했다.

| stream | 실제 source/함수 | 의미 및 제한 |
|---|---|---|
| 원 센서 | `Y2FT_AQ/src/FT_EtherGet.cpp::FTGet`, `FTGetMain.cpp::transferData`, `filtering` | Ethernet 6축 float → 영점 offset → MOV(size=2) → sensor→TCP 축 회전 diag(1,-1,-1). **영점/필터 이전 센서 frame의 원본 topic은 없다.** 새 topic/보정을 만들지 않음. |
| timestamp 있는 base 힘 | `/ur10skku/ftdata`, WrenchStamped, `FTGetMain.cpp` | 기존 MOV + 조건부 gravity matrix 보정 + TCP→base 회전. 소스 루프 nominal 2000Hz, publisher RELIABLE depth10. header stamp는 publisher `now()`, 하드웨어 취득 clock 아님. frame_id 빈 값. |
| pre-gravity 힘 | `/ur10skku/ftdata_tcp_raw`, WrenchStamped | nominal 100Hz, frame_id=`tcp`. 이름은 raw지만 이미 zero/MOV/축회전 후. torque는 축만 회전, 접촉점으로 torque reference 이동 없음. |
| TCP 보정값 | `/ur10skku/ftdata_tcp`, WrenchStamped | nominal 100Hz, gravity 보정 여부는 런타임 flag·matrix·pose 수신 여부에 따라 달라짐. 이를 logger에서 재보정하지 않음. |
| 기존 정책 힘 | `/ur10skku/currentF`, Float64MultiArray, `Y2RobMotion/src/robot_motion.cpp::ftsensorCB/state_publisher` → `inference_core.py::_on_force` | `[Fx,Fy,Fz,Tx,Ty,Tz]`, `/ftdata` 최신값 재발행. source timestamp/seq/frame 없음. 새 센서 측정으로 간주하면 안 됨. 정책은 `_extract_force3`의 기존 force_indices(기본 0,1,2)로 Fx/Fy/Fz만 사용. torque는 정책 미사용, 로그에는 보존. 대체 Wrench 입력도 `_on_force_wrench`에서 force+torque를 로그 보존. |
| 실제 TCP | `/ur10skku/currentP`, Float64MultiArray, `robot_motion.cpp::JointStateCB/state_update/state_publisher` → `_on_pose` | joint feedback → `Y2Kinematics/src/KinematicsUR10.cpp::forwardKinematics`: `T06 * EE2TCP`. base frame mm, rotation vector(rad). `Y2Matrix/src/RotationTransform.cpp::fromSpatialAngle`은 Rodrigues/axis-angle. 따라서 Euler가 아니다. quaternion xyzw는 기록용 변환만 수행. 실제 속도는 publish되지 않음. |
| policy prediction | `inference_core.py::_on_infer_timer`, `_denorm_action_seq` 직후 | 기존 9D x,y,z,rotvec[3],Fx,Fy,Fz. SRF/offset/clip/anchor 이전 전 action 저장. E1은 stain-relative XY, Z/자세/force는 기존 의미 유지. action horizon128/history30, 실제 runtime 전체값은 metadata. |
| topic 전송 | `_publish_cmd` → `/ur10skku/cmdMotion` | 기존 XY force disable, 안전 hold 적용 후 실제 publish payload. 원 정책 예측과 구분. |
| service 전송 | `_start_ptp_alignment`, `_ptp9d_advance`, stream start/stop/topup/set-force → `/singleArm_cmd/single_arm_command` (`SingleArmCommand`) | 실제 Request `target_pose`/mode/velocity. PTP/PTP9D는 rotvec **성분을 degree로** 바꿔 보냄. segment의 모든 waypoint/action index 보존. 서비스 응답=응답일 뿐 실제 도달/가공 완료 아님. |
| controller 보고 target | `/ur10skku/targetP`, `/ur10skku/targetF`, Float64MultiArray, `robot_motion.cpp::state_publisher` | targetP=controller `target_pose`, targetF=`FC_AC_desX[6:9]`. 둘을 별도 수신 레코드로 저장. targetF는 `force_control.cpp` 내부 `commanded_Fd`(precontact hold/Q frame 등의 변경)의 **최종 값이 아니다**. `controller_applied_final_force=null`. targetP도 실제 위치 아님. header 없어 명령 ID와 직접 대응 불가. |
| 상태 | `/ur10skku/ctlMode`, String | controller 모드 전환만 기록. spindle ON/OFF 신호 아님. |
| 작업 기준 | `/stain_relative_frame/stain_origin`, `stain_relative_frame/inference_adapter.py::StainOriginClient._on_origin` | 기존 transient-local reliable depth1, 최초 중심 XY mm/선택 angle rad 고정. logger는 정책이 실제 고정한 `_srf`를 읽는다. 별도 재검출/마스크 활성화 없음. source stamp 없는 값이므로 최초 ready 관찰 시각이라고 명시. |
| RGB/mask | `_on_img`/`_on_stain_mask` | 기존 RGB decode 후 preprocessing 이전 이미지의 최초/종료 각1장. mask는 **이미 사용중인 subscriber에 수신될 때만** 최초1장. 기존 `use_stain_mask=false`, `stain_canon_enable=false` 유지. |

F/T force 단위 N / torque 단위 N·m는 ROS Wrench SI 관례이며 Ethernet sensor scale/교정의 현재 유효성은 하드웨어 검증이 필요하다. header frame과 소스에서 확인한 semantic frame을 별도 열로 보존한다. `Fz`를 Fn으로 해석하지 않고 `normal_force_signed`, normal, compression sign, transform timestamp는 unknown이다. 원본 부호와 비유한 값을 보존한다.

`setup_parameters.yaml`에는 UR10, CONTROL_PERIOD=0.008, Force_Con_Coordinate=1 및 EE2TCP translation z=320mm + 회전이 있다. 이것은 **보관한 설정 후보**이지 배포된 controller 설정 확인 결과가 아니다. 접촉 기준점과 유효 footprint offset을 이 숫자로 대체하지 않는다. 기존 `spindle_gravity.yaml`도 사본/해시만 보존한다.

`checkpoints/stain_relative_frame/homography.json`에는 `K_fxfycxcy`, `R_cb`, `t_cb_base_mm`, `n_cam_plate_normal`, home_pose 등 기존 calibration이 있다. metadata의 `camera_calibration_candidate`와 artifacts에 보존하나 현재 표면/카메라에 유효한 normal 또는 TF로 적용하지 않는다. 실시간 CameraInfo/TF 조회·새 보정은 하지 않는다. 확인된 RPM publisher/실측 API/회전 상태는 없다. 수동 설정 RPM만 받아 provenance=`manual:metrics_rpm_setpoint`, measured_rpm=null.

## Schema와 clock

모든 CSV는 `run_id,logger_seq,source_seq,source_stamp_ns,receipt_ros_ns,receipt_monotonic_ns,stamp_basis,source_clock,frame_id,source,validity`로 시작한다. ROS2 header의 sequence는 없으므로 source_seq=null. CSV null은 빈 칸, JSON null은 null. raw 배열/추가 정보는 JSON cell. 값이 NaN/Inf이면 invalid 표시와 원값을 남기며 0으로 대체하지 않는다.

source_header timestamp와 receipt ROS clock의 동기화는 미확인이다. 서로 빼서 latency를 계산하지 않는다. monotonic은 logger host에서만 의미 있다. header 없는 currentP/currentF를 이전 값의 새로운 취득 시각으로 위장하지 않는다. callback 수신 시각만 있고 취득 age는 unknown이다. writer diagnostic 이벤트는 ROS clock을 얻지 않으므로 writer monotonic과 별도 source를 사용한다.

`wrench.csv`: Fx/Fy/Fz/Tx/Ty/Tz, units, representation(재발행/base보정/TCPpregravity/TCP보정), semantic_frame, 보정 상태, 원 배열. 여러 source를 시간 정렬/중복 제거하지 않고 따로 기록한다.

`tcp_pose.csv`: raw x/y/z/rotvec, xyzw quaternion, feedback FK TCP 표시, pose/frame/unit. reference가 준비되면 기존 `_srf_observation_pose6`의 복사본 결과를 `work_pose`에 저장. 이것은 XY 기준 정책 좌표계이지 임의의 full rigid transform이 아니다. 실제 속도/접촉점 offset은 null.

`commands.csv`: `policy_prediction`(미사용 horizon 포함), `node_sent`(실제 전송), `controller_reported_target`(관측된 targetP/F)를 구분. inference_id/plan_id는 생성 계획 연결, command_id는 전송별, action_index는 service가 실제 선택한 인덱스. controller topic에는 ID가 없고 topic interpolation/aggregation의 정확한 선택 인덱스도 unknown으로 표시한다. force-only stream service에 실제 pose를 끼워 넣지 않는다. `legacy.csv`만 기존 비교 스크립트 호환용 최신 snapshot을 유지하며 별도 source/수신 age를 표시한다.

`events.jsonl`: run_start/end/interrupted/error, reference_fixed, first_command_sent, inference_start/end/error, wait, stage_observed, command_sent, service_response/error, safety_limit, policy_force_postprocess, controller_mode, logger_overflow_or_drop/write_error. APPROACH/PRELOAD/TRACK/RELEASE는 코드 단계이지 검증된 가공 구간이 아니다. 두 점 사이 의도적 이탈/비의도적 접촉 상실의 의미 신호는 없어 수동 구간 지정 필요. force threshold로 가공 구간을 새로 만들지 않는다.

추론별 OFF=constant_zero / ON=measured, CPU 원 history norm(패딩/정규화 이전), OFF conditioned norm=0. ON normalized norm은 GPU 동기화를 피하려고 null. 실제 force stream은 그 전에 독립적으로 기록한다. logger에서 policy history/cache에 쓰는 경로는 없다.

## 변경 및 한계

- `nrs_imitation/inference_core.py`: 작은 로깅 hook, 기존 동기 CSV를 비동기 backend로 교체. 기존 계산/주기/안전 로직은 유지.
- `nrs_imitation/inference_metrics.py`: 읽기 전용 bridge, sensor/controller 별도 observer executor, metadata/고정 ROI/서비스 단계.
- `nrs_imitation/execution_metrics.py`: bounded queue, writer, 원본 CSV/JSONL, 충돌 없는 폴더, 주기 flush/원자적 status/summary, 사본·해시.
- `launch/inference_gradcam_single_cam.launch.py`: 새 **로그 전용** argument 선언/전달. clean launch가 기존 방식으로 포함한다.
- `test/test_execution_metrics.py`, `scripts/check_inference_log.py`, `scripts/make_inference_log_example.py`: offline 검증·품질 검사·명시적 synthetic 예시.

추가 subscriber는 소스의 RELIABLE publisher에 호환되는 BEST_EFFORT/depth100으로 관찰하여 새 reliable backpressure를 만들지 않는다. callback의 가벼운 snapshot만 bounded queue로 전달한다. 파일 write/flush/image encode는 writer에만 있다. OS scheduling/GIL/하드웨어 수신 손실·실시간 간섭은 이 offline 테스트로 보장할 수 없다. 이 이유로 원본 source별 gap/drop/오류를 실험마다 확인해야 한다.

강제 종료시 마지막 flush 이후 약0.5초 및 queue 대기 데이터는 사라질 수 있다. 오류·overflow는 제어를 임의 중단하지 않고 warning/event/status/summary에 표시한다. 데이터가 손실된 실행은 실험 분석에서 별도로 표시해야 한다. 저장장치 전체 오류라면 disk에 오류 자체도 쓸 수 없어 ROS 경고가 최종 증거일 수 있다.

향후 공간 노출량에는 현재 유효한 법선/좌표계·압축력 부호, contact footprint/offset, 설정/실측 회전 조건, 수동 구간, clock/동기화 검증이 더 필요하다. TCP 이송속도를 회전 slip 속도로 대체하지 않으며 정지 TCP를 0 노출로 처리하지 않는다. 목표 힘으로 실제 가공량을 계산하거나 미보정 μm/제거율을 출력하지 않는다.
