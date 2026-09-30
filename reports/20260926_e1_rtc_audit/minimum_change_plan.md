# 감사 후 최소 변경 계획 — 구현 전 검토본

이번 범위는 paper E1 R/T/C이다. 구번호 경로는 유지한다. 재학습·새 2점 과제·unseen 일반화·새 blending 알고리즘은 포함하지 않는다. 아래 파일/명령은 **계획**이며 구현·실행 가능한 상태로 표시하지 않는다.

**설정 변경만으로 해결할 수 없는 이유**

현재 service는 timestamp 배열이 없고, 위치 큐의 force 열을 소비하지 않는다. 위치는 거리/속도로 retime되고 힘은 최신 nearest-XYZ index에서 따로 전송된다. 따라서 기존 T를 그대로 service adapter에 넣으면 원본 시간축, 왕복 방향, 정지 체류의 위치–힘 대응을 잃는다. 반대로 C를 기존 timed_topic에 연결하는 것은 사용자 프롬프트가 금지한 실행 경로 변경 방향이다.

권장 방향은 **C++ 서비스 실행기를 유지하면서 명시적인 시간·force 일정과 실행 상태를 전달하는 별도 버전의 서비스 규약을 추가**하는 것이다. 이것은 단순 logger 수정이 아니고 실행 시간·큐 규약의 변경이다. 사용자 승인이 필요하다. 기존 legacy service 명령과 동작은 호환 경로로 보존한다. 새 실험의 세 방법이 모두 새 버전을 사용해야 하며 과거 C와 동일 실행이었다고 주장하지 않는다.

**승인할 구체적인 변경 범위**

| 부분 | 재사용·변경 | 확인할 기준 |
|---|---|---|
| R/T 생성부 | PR.rule_actions, TimedActions, export_episode 재사용. exporter의 특정 training-split/상수 checkpoint 결합만 명시 spec 입력으로 분리 | 신경망 없이 생성, 원본 timestep 고유성·pose/force/time 보존. validation reference 허용 여부를 명시 |
| C 생성부 | 사용자가 선택한 checkpoint·normalizer·ON obs/history·seed·flow steps·horizon·반복 추론 유지. _postprocess_provider_action 경계 재사용 | 같은 observation/noise에서 로깅 전후 action 일치. ckpt/normalizer/schema 불일치와 누락은 즉시 실패, fallback 없음 |
| 시간 service interface | 기존 SingleArmCommand와 구분되는 versioned timed chunk 계약: run/plan/command/action ID, pose6/force3, 명시 상대 실행시각, frame/단위, end-of-plan·취소 정보. 형식은 승인 후 실제 인터페이스 패키지에 맞춰 구현 | 호스트 간 monotonic 직접 차감 금지. controller가 승인한 run-clock/sequence를 기준으로 처리하고 송신·수신·예정·적용 시각 분리 |
| controller-side queue | 기존 RC service→queue→streamLoop→cmdMotion 경로 재사용. 유한 R/T는 전체 episode를 순서대로 chunk 공급; C는 반복 추론하며 이미 commit된 prefix와 새 suffix 경계를 명시 | duplicate/재전송/중복 horizon 방지, accepted와 applied 구분, queue depth·wait·underflow·end 확인, 실패 시 조용한 fallback 없음 |
| 위치·힘 시간 규약 | 기존 위치/자세 보간 계산 및 gain/rate cap을 조사한 기준으로 보존하면서 명시 waypoint duration을 소비. force schedule은 같은 plan clock/index로 선택하고 공통 gate가 우선 | 시간 소비·force 선택의 변경은 실행 의미 변경임을 version에 기록. 기존 0.05 s 최솟값·속도 제한과 원본 시간의 충돌을 검사. 불가능한 입력을 묵시적 retime하지 않음 |
| 승인 후 feasibility 판정 | 원본 T 시간이 현재 속도·최소구간 조건에서 수용 가능한지 전체 궤적 및 phase 경계 검사. 공통 grid로 바꿀 때 총시간과 대응 오차 공개 | 정확한 시간과 기존 제한을 동시에 만족하지 못하면 실기 차단 및 구체적 충돌 보고. 제한 완화·보간 변경을 자동 수행하지 않음 |
| safety·시작/종료 | 기존 alignment, measured contact gate, workspace/age watchdog, EX의 stop verifier를 공통 service 경로에 연결. R만 home으로 가는 비대칭 종료 해소는 승인된 공통 절차 사용 | 하위 FC 알고리즘·게인·안전 수치 변경 없음. 검증되지 않은 Ctrl-C/ACK를 정지 완료로 간주하지 않음. 새 안전한계는 calibration 근거 필요 |
| logger | EM/IM 비동기 writer 유지. F_tar/F_cmd_sent/controller-reported/applied/F_meas, 예정시각·consumed index, trial/phase·evaluation-reference hash, queue 이벤트 추가 | 기존20 Hz measurement 유지. drop/clock/gap/stale/write error를 품질 판정에 반영. 원래 sensor time이 없으면 receipt-only 명시 |
| 평가·운영 | 별도 offline 도구와 버전/해시 spec·manifest·annotation·stylus template 추가 | 실행 loop 밖에서 평가, 실제 로그 부족 시 NA. 실험값·성공률 생성 금지 |

위 범위는 **로컬 구현 및 offline 검증 승인**을 요청하는 범위다. 빌드 결과의 실기 배포나 로봇 launch·명령·스핀들 구동 승인은 포함하지 않는다. 외부 제어기를 수정할 수 없는 조건이면 “현 상태에서 프롬프트의 실행·시간 요구사항 동시 충족 불가”로 남긴다. 임의로 timed_topic으로 우회하지 않는다.

**사전에 고정할 입력**

- C: 정확한 checkpoint 파일, 함께 쓸 normalizer. 파일/정규화 통계/config hash, seed/history/flow steps/horizon·실제 적용 설정을 고정한다. 후보 선택이 아직 없으면 실기 준비는 중단한다.
- T: 명시 episode ID와 선택 근거. episode_29를 과거 성능이나 과거 설정만으로 새로 선택하지 않는다.
- Reference: 같은 작업 조건의 교시 ID·전처리·normal/frame·선정 근거. 독립 자료가 없으면 기존 교시 기반 재현 평가로 표시한다. T-source와 같은 경우 이를 공개하고 E_target≈0의 독립성 한계를 적는다.
- 실험 조건: 공구·재료·형상·frame/TCP/normal/tare/gravity calibration, R validation recipe, 설정 RPM 및 출처와 actual RPM 측정 가능 여부, 힘·속도 제한, 시작/종료/timeout, phase·pass·방향·dwell 정의.
- 설계: 실제 specimen/독립 region/block, 반복 수, seed와 block 내 무작위 실행 순서. 본 실험과 smoke/validation 분리. 같은 영역 연속 R→T→C 금지. 반복 수의 통계적 충분성을 자동 주장하지 않는다.
- 분석: phase별 progress grid·가중치·센서 허용 gap/age·reference version·완료/coverage 기준. 시간 gap 허용치를 이번 감사에서 임의 확정하지 않았다.

**오프라인 평가 구현 명세**

E_profile = sqrt(sum(w·(F_meas−F_ref)²)/sum(w)), E_target = sqrt(sum(w·(F_tar−F_ref)²)/sum(w)). 사전 정의한 stage+stroke ID+방향+진행률로 같은 grid에 대응한다. 실제 위치·영상·수동 annotation을 사용하며 force가 큰 구간만 택하지 않는다. Dwell은 별도 phase 시간·노출량을 유지한다. 대응 불명확·pass 수 차이·중단/누락 시 전체 점수는 unavailable이고 부분 진단과 coverage를 별도 출력한다. 조기 종료 구간을 0–1 전체로 늘이지 않는다.

E_track = sqrt(sum((F_meas−F_cmd)²·dt)/sum(dt)). 같은 실제 시간축에서 명령의 유효 시각을 사용한다. 내부 applied setpoint가 없으면 별도 `last_sent_tracking_rmse_N`과 근거를 출력하고 적용값 기준 오차로 표기하지 않는다. 서로 다른 force frame은 검증된 변환 없이는 계산하지 않는다. APPEND unused force=0·prediction-only·Position-mode pose 명령을 배제한다. force별 정규화, 최적 지연 제거, unconstrained DTW를 기본 분석에 넣지 않는다.

허용 gap 밖은 보간하지 않고 제외 길이/비율·coverage·재측정 필요를 함께 출력한다. 정상/수동중단/timeout/안전중단/계측불가를 분리한다. method mean±SD는 독립 완료 trial별 점수 기준이며 partial과 섞지 않는다. 원시 force mean/SD는 설명용이다.

trial/method CSV의 주 표는 `method | independent_trials | completed/total | E_profile_N | E_target_N | E_track_N | cycle_time_s | safety/gate_events | partial_or_invalid_runs`이다. 분석 가능한 trial 분모와 available/unavailable 사유도 보존한다. 그림은 같은 축/scale로 phase-progress의 F_ref/R/T/C 실측값 및 실제 시간의 F_tar/F_cmd/F_meas·gate/phase를 제시한다. 현재 실험 데이터가 없으므로 이 표에 결과 행을 채우지 않는다.

Stylus 양식은 specimen/run/method/section, d_before_um/d_after_um/local_dishing_um/Ra, 기준면·측정 조건·불확도·정량한계를 받는다. eta=100·(before−after)/before는 before가 정량한계보다 클 때만 계산한다. after<한계는 censor 상태로 남기고 0으로 바꾸지 않는다. 음수 감소율은 유지하고 미측정은 NA. 이미지·서비스 ACK·Preston proxy로 실측을 대체하지 않는다.

**승인 후 검증 및 완료 기준**

| 단계 | 필요한 검증 |
|---|---|
| 생성부/입력 | R/T 네트워크 미사용; 실제 T unique-time 복원·resampling 총시간/대응; unknown 단위·없는 episode·OFF/schema mismatch 차단 |
| 공통 경로 | mock에서 같은 service/type·queue/frame/단위/time, 중복 chunk·underflow·stale/gap·취소·부분 완료, C 반복 추론과 실제 지연 보존 |
| 로깅 | 관측된 actual과 target 혼동 없음; unused0 배제; clock 역행/drop/write error; fixed seed C action 불변 |
| 계산 | 같은 synthetic 가변 reference RMSE≈0; force 규모 차이 검출; 높은 raw SD와 낮은 profile RMSE 양립; 좋은 자기추종과 나쁜 reference 재현 분리 |
| 집계 | 중단·missing phase·partial coverage·gap 은폐 없음; 독립 trial 분모와 mean±SD; C 우승을 요구하는 테스트 없음 |
| 운영 산출물 | spec/hash/run manifest·random order·stylus template·annotation, 실제 --help와 일치하는 offline/dry-run/작업자 실행/종료 후 분석 명령 |
| 현장 | active controller/schema/설정 대조, 적용 명령·queue·안전중단·이탈 확인. mock 통과를 하드웨어 확인으로 표시하지 않음 |

완료 보고는 구현 / offline 검증 / 하드웨어 확인 / 실험 완료를 분리한다. 현재 승인 전 단계에서는 앞의 두 단계도 완료로 표시하지 않는다.
