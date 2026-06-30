# umi_ros2

`umi_ros2`는 Dynamixel 기반 UMI 그리퍼를 ROS 2에서 제어하기 위한 Python 패키지입니다. 현재 패키지에는 세 가지 실행 방식이 포함되어 있습니다.

- `umi_gripper_pico`: 키보드 입력과 `/gripper/command` 토픽을 이용한 그리퍼 제어
- `umi_gripper_sub`: `/gripper/command` 토픽과 조이스틱 토픽을 이용한 그리퍼 제어
- `umi_gripper_sub_pwm`: 터미널 키보드 입력을 이용한 PWM 모드 테스트

기본적으로 Dynamixel을 Current-based Position Control Mode(Mode 5)로 사용하며, 현재값과 위치를 ROS 토픽으로 publish합니다.

## 1. 패키지 위치

이 패키지는 워크스페이스 루트의 `src/umi_ros2` 아래에 두면 됩니다.

```bash
<workspace_root>/src/umi_ros2
```

## 2. 의존성

`setup.py` 기준 Python 의존성:

- `dynamixel-sdk`
- `opencv-python`
- `pyserial`

ROS 2 의존성:

- `rclpy`
- `std_msgs`
- `launch`
- `launch_ros`
- `ament_index_python`

## 3. 빌드

워크스페이스 루트에서 빌드합니다.

```bash
cd <workspace_root>
colcon build --packages-select umi_ros2
source install/setup.bash
```

## 4. 실행 가능한 노드

`setup.py`에 등록된 ROS 2 실행 엔트리는 아래 세 개입니다.

- `umi_gripper_pico`
- `umi_gripper_sub`
- `umi_gripper_sub_pwm`

직접 실행:

```bash
ros2 run umi_ros2 umi_gripper_pico
ros2 run umi_ros2 umi_gripper_sub
ros2 run umi_ros2 umi_gripper_sub_pwm
```

런치 파일 실행:

```bash
ros2 launch umi_ros2 umi_grp.launch.py
ros2 launch umi_ros2 umi_grp_sub.launch.py
```

## 5. 각 노드 설명

### 5.1 `umi_gripper_pico`

관련 파일:

- [umi_ros2/umi_grp.py](umi_ros2/umi_grp.py)
- [config/umi_grp.yaml](config/umi_grp.yaml)
- [launch/umi_grp.launch.py](launch/umi_grp.launch.py)

특징:

- 기본 노드 이름: `umi_gripper`
- 기본 시리얼 포트: `/dev/ttyUSB0`
- 기본 보드레이트: `57600`
- 기본 그리퍼 ID: `0`
- `/gripper/command` (`std_msgs/Int32`)를 subscribe
- `/gripper/present_current_mA` (`std_msgs/Float32`) publish
- `/gripper/present_position` (`std_msgs/Int32`) publish
- 키보드 입력으로 open/close 및 미세 조정 가능

기본 키보드 조작:

- `o` 또는 `O`: open
- `c` 또는 `C`: close
- `+` 또는 `=`: 목표 tick 증가
- `-` 또는 `_`: 목표 tick 감소
- `ESC`: 종료

실행 예:

```bash
ros2 launch umi_ros2 umi_grp.launch.py
```

직접 노드 실행 예:

```bash
ros2 run umi_ros2 umi_gripper_pico --ros-args --params-file \
  $(ros2 pkg prefix umi_ros2)/share/umi_ros2/config/umi_grp.yaml
```

### 5.2 `umi_gripper_sub`

관련 파일:

- [umi_ros2/umi_grp_sub.py](umi_ros2/umi_grp_sub.py)
- [config/umi_grp_sub.yaml](config/umi_grp_sub.yaml)
- [launch/umi_grp_sub.launch.py](launch/umi_grp_sub.launch.py)

특징:

- 기본 노드 이름: `umi_gripper_sub`
- 기본 시리얼 포트: `/dev/ttyUSB0`
- 기본 보드레이트: `57600`
- 기본 그리퍼 ID: `0`
- `/gripper/command` (`std_msgs/Int32`)를 subscribe
- `/ur10skku/joy_move` (`std_msgs/Float64MultiArray`)를 subscribe
- `/gripper/present_current_mA` (`std_msgs/Float32`) publish
- `/gripper/present_position` (`std_msgs/Int32`) publish
- 조이스틱 축 값을 그리퍼 tick 범위로 매핑

조이스틱 기본 파라미터:

- `joystick.enabled: true`
- `joystick.command_topic: /ur10skku/joy_move`
- `joystick.axis_index: 5`

실행 예:

```bash
ros2 launch umi_ros2 umi_grp_sub.launch.py
```

직접 노드 실행 예:

```bash
ros2 run umi_ros2 umi_gripper_sub --ros-args --params-file \
  $(ros2 pkg prefix umi_ros2)/share/umi_ros2/config/umi_grp_sub.yaml
```

## 6. 주요 토픽

### Subscribe

- `/gripper/command` (`std_msgs/Int32`)
  - 목표 그리퍼 위치 tick 전달
- `/ur10skku/joy_move` (`std_msgs/Float64MultiArray`)
  - `umi_gripper_sub`에서 사용
  - 지정한 `axis_index` 값을 `gripper.min_tick` ~ `gripper.max_tick`으로 매핑

예시:

```bash
ros2 topic pub /gripper/command std_msgs/msg/Int32 "{data: 1500}" -1
```

### Publish

- `/gripper/present_current_mA` (`std_msgs/Float32`)
- `/gripper/present_position` (`std_msgs/Int32`)

모니터링 예시:

```bash
ros2 topic echo /gripper/present_position
ros2 topic echo /gripper/present_current_mA
```

## 7. 파라미터 파일

기본 파라미터 파일:

- [config/umi_grp.yaml](config/umi_grp.yaml)
- [config/umi_grp_sub.yaml](config/umi_grp_sub.yaml)

런치 파일은 기본적으로 위 YAML 파일을 사용하며, `config_file` 인자로 다른 파일을 넘길 수 있습니다.

예시:

```bash
ros2 launch umi_ros2 umi_grp.launch.py config_file:=./config/custom.yaml
ros2 launch umi_ros2 umi_grp_sub.launch.py config_file:=./config/custom.yaml
```

주요 파라미터 예:

- `dxl.port`: Dynamixel 연결 포트
- `dxl.baud`: 보드레이트
- `dxl.gripper_id`: Dynamixel ID
- `gripper.min_tick`, `gripper.max_tick`: 동작 범위
- `dxl.goal_current_mA`: 목표 전류
- `gripper.close_current_stop_mA`: 물체 파지 시 정지 기준 전류
- `gripper.command_topic`: 명령 토픽 이름

`umi_gripper_pico` 전용 파라미터:

- `trigger.min_tick`, `trigger.max_tick`
- `gripper.invert`
- `monitor.enabled`
- `monitor.print_period_sec`
- `keyboard.step_size`

`umi_gripper_sub` 전용 파라미터:

- `gripper.close_increases_tick`
- `joystick.enabled`
- `joystick.command_topic`
- `joystick.axis_index`

## 8. PWM 테스트 노드

[umi_ros2/umi_grp_sub_pwm.py](umi_ros2/umi_grp_sub_pwm.py)는 PWM 모드 테스트용 노드입니다.

주의:

- 터미널 키 입력을 사용하므로 실행 터미널이 TTY여야 합니다.
- 기본 파라미터는 코드 내부 기본값을 사용합니다.

실행:

```bash
ros2 run umi_ros2 umi_gripper_sub_pwm
```

기본 키:

- `o`: open
- `c`: close
- `s`: stop
- `q`: quit

## 9. 실행 전 확인 사항

- Dynamixel 장치가 `dxl.port`에 실제로 연결되어 있어야 합니다.
- 장치 ID와 보드레이트가 YAML 설정과 일치해야 합니다.
- 현재 설정은 기본적으로 `/dev/ttyUSB0`, ID `0`, baud `57600`을 가정합니다.
- 시리얼 장치 권한 문제로 실행이 실패할 수 있습니다.
- 전류 제한, 위치 제한, stop threshold는 실제 하드웨어 기준으로 반드시 재확인해야 합니다.

## 10. 빠른 시작

토픽 기반 제어를 가장 빠르게 시험하려면:

```bash
cd <workspace_root>
colcon build --packages-select umi_ros2
source install/setup.bash
ros2 launch umi_ros2 umi_grp_sub.launch.py
```

다른 터미널에서:

```bash
cd <workspace_root>
source install/setup.bash
ros2 topic pub /gripper/command std_msgs/msg/Int32 "{data: 1500}" -1
```
