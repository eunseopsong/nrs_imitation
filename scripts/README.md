# 학습 스크립트 변경 이력

## Preliminary — 현재 Flow 학습 구조와 파라미터 이해

이 절은 Polishing과 Gripper Flow 모델을 다시 학습하거나 파라미터를 조정할
때 필요한 기본 개념을 정리한다. 두 task는 action 차원과 일부 observation만
다르고, observation을 하나의 condition vector로 만든 뒤 미래 action sequence를
생성하는 전체 구조는 같다.

### 1. 모델이 학습하는 것

한 학습 sample은 임의의 episode 시점에서 다음 두 부분으로 만들어진다.

- **Observation:** 현재 robot pose/force, 현재 RGB image, 최근 force history,
  그리고 task에 따라 stain mask, marker 또는 gripper state
- **Action target:** 그 시점부터 시작하는 길이 `chunk_size`의 미래 동작 sequence

전체 흐름을 단순화하면 다음과 같다.

```text
현재 pose + 현재 force ── qpos MLP encoder ─────────┐
최근 force sequence ───── Force GRU encoder ────────┤
RGB image(s) ───────────── ResNet18 image encoder ──┤
stain mask / marker ────── 해당 feature encoder ────┼─ concat + fusion MLP
gripper position/current ─ Gripper encoder ─────────┘          │
                                                               ▼
                                                  global condition vector
                                                               │
noise action sequence z(t) + Flow time t ──────────────────────┤
                                                               ▼
                                                  Conditional 1D U-Net
                                                               │
                                                               ▼
                                             action velocity sequence 예측
```

학습 시에는 실제 action sequence `z1`과 Gaussian noise `z0` 사이의 임의 지점
`z(t)`를 만들고, 모델이 `z0`에서 `z1`로 가는 velocity를 맞히게 한다. Inference
시에는 noise sequence에서 시작해 예측 velocity를 여러 번 적분하여 하나의
미래 action sequence를 만든다. 여기서 **Flow time `t`는 로봇 trajectory의 실제
시간이 아니라 noise를 action으로 변환하는 생성 과정의 진행도**이다.

따라서 파라미터는 크게 다음 세 부류로 구분해야 한다.

- `force_history_len`, `chunk_size`처럼 모델에 넣는 **시간 범위**를 정하는 값
- `*_feature_dim`, `*_hidden_dim`, `flow_down_dims`처럼 **encoder와 생성 모델의
  표현 용량**을 정하는 값
- `lr`, `batch_size`, scheduler처럼 같은 모델을 **어떻게 최적화할지** 정하는 값

### 2. Observation과 action 차원

| 파라미터 | 현재 기본값 | 기능과 주의점 |
|---|---:|---|
| `state_dim` | 9 | 이름은 qpos지만 현재 데이터에서는 robot pose 6차원과 현재 force 3차원을 합친 observation이다. Gripper state는 여기에 넣지 않고 전용 encoder로 별도 처리한다. |
| `force_dim` | 3 | Force GRU가 매 시점 받는 force vector 크기다. 현재 Fx, Fy, Fz 세 축을 뜻한다. |
| `action_dim` | Polishing 9 / Gripper 11 | Polishing action은 pose 6 + force 3이다. Gripper는 여기에 gripper position 1 + current 1이 추가된다. 따라서 Gripper의 `state_dim`이 아니라 `action_dim`이 11이다. |
| `marker_dim` | 14 | marker observation을 펼친 vector 크기다. marker mode에서만 marker encoder 입력 크기로 사용된다. 일반 single/dual camera mode에서는 실질적으로 사용되지 않는다. |
| `norm_mode` | `minmax_m11` | qpos, action, force history, marker 및 gripper position/current를 training dataset min/max 기준으로 정규화한다. `minmax_m11`은 `[-1,1]`, `minmax_01`은 `[0,1]` 범위다. 범위 밖 validation/inference 값은 경계로 clip된다. 학습과 inference가 반드시 같은 stats와 mode를 사용해야 한다. |

`state_dim=9` 안에는 이미 **현재 force 3축**이 들어 있다. Force history를 켜면
동일한 현재 force를 포함한 최근 force sequence도 별도 GRU로 들어간다. 전자는
“지금 힘이 얼마인가”를, 후자는 “최근에 힘이 증가·유지·감소하고 있는가”를
표현하므로 단순 중복이라기보다 순간값과 변화 추세를 함께 제공하는 구조다.

### 3. Observation encoder와 관련 파라미터

#### 현재 state encoder

`state_dim=9` vector는 MLP를 거쳐 `flow_obs_hidden_dim` 크기의 feature가 된다.

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `flow_obs_hidden_dim` | 256 | 현재 pose/force를 표현하는 MLP의 hidden/output 크기다. 너무 작으면 상태 표현이 부족하고, 크게 하면 파라미터 수와 과적합 가능성이 증가한다. |

#### Force history encoder

Force history의 입력 shape은 `(batch, force_history_len, force_dim)`이며 현재는
GRU가 sequence를 앞에서부터 읽은 뒤 **마지막 hidden state 하나**를 force
feature로 사용한다.

| 파라미터 | 기본값 | Encoder와의 연결 |
|---|---:|---|
| `use_force_history` | true | false이면 Force GRU와 그 feature가 fusion에서 완전히 빠진다. |
| `dataset_hz` | 30.0 | history와 action horizon을 실제 초 단위로 환산하는 학습 데이터 row rate다. 센서의 원 publish rate가 아니라 동기화된 dataset의 sampling rate를 사용해야 한다. |
| `force_history_sec` | 1.0 | GRU가 볼 force 과거 구간을 초 단위로 정의한다. 30 Hz에서는 30 step이다. |
| `force_history_len` | 30 | GRU에 실제 입력되는 sequence row 수다. `force_history_sec>0`이면 `round(dataset_hz × force_history_sec)`로 다시 계산된다. |
| `force_encoder_hidden_dim` | 64 | GRU hidden state와 최종 force feature의 크기다. 길이와 달리 “얼마나 오래 보는가”가 아니라 그 구간을 “몇 개 숫자로 요약하는가”를 정한다. |
| `force_encoder_num_layers` | 1 | 쌓는 GRU layer 수다. layer를 늘리면 표현력과 계산량이 증가하지만 현재 dataset 규모에서는 과적합 위험도 커진다. |
| `force_encoder_dropout` | 0.0 | GRU layer 사이 dropout이다. PyTorch GRU 특성상 `num_layers=1`일 때는 값을 올려도 적용되지 않는다. 2개 이상의 layer에서만 의미가 있다. |

`force_history_len`을 10에서 30으로 늘려도 force feature 출력은 여전히 64차원이다.
달라지는 것은 GRU가 0.33초가 아니라 1초의 변화 패턴을 64차원으로 요약한다는
점이다. 길이를 과도하게 늘리면 오래된 힘까지 섞이고 GRU가 필요한 falling
edge를 보존하기 어려워질 수 있으며, 계산량도 sequence 길이에 비례해 증가한다.
Episode 시작 부분처럼 과거 sample이 부족한 경우에는 가장 오래된 force 값을
앞쪽에 반복해 항상 `force_history_len` 길이를 맞춘다.

#### Image encoder

각 camera image는 ImageNet 방식으로 정규화된 뒤 ResNet18을 통과한다.
ResNet18의 512차원 출력은 projection layer를 통해 `flow_image_feature_dim`으로
변환된다. Dual camera라면 각 camera feature를 따로 만든 뒤 이어 붙인다.

| 파라미터 | 기본값 | Encoder와의 연결 |
|---|---:|---|
| `camera_names` | mode에 따라 `cam0` 또는 `cam0 cam1` | 사용할 image stream과 개수를 정한다. 실제 HDF5 key 및 `obs_mode`와 일치해야 한다. |
| `no_pretrained` | false | 기본은 ImageNet pretrained ResNet18을 사용한다. 옵션을 켜면 image backbone을 random initialization부터 학습하므로 더 많은 데이터와 학습이 필요하다. |
| `flow_image_feature_dim` | 512 | camera 하나당 fusion으로 전달되는 image feature 크기다. Dual camera는 기본적으로 `512 × 2`가 된다. |

`flow_image_feature_dim`을 키우는 것은 image 해상도나 frame 수를 늘리는 것이
아니다. ResNet이 한 장에서 추출한 정보를 더 큰 vector로 전달하는 것이다.
데이터가 충분하지 않으면 dimension 증가가 feature 품질 개선보다 과적합으로
이어질 수 있다.

#### Stain mask encoder — Polishing 전용

Polishing에서는 RGB와 별도로 stain mask 위치에 해당하는 ResNet feature map을
masked mean pooling한다. 이 stain-local feature도 image projection을 거쳐 RGB
global feature와 함께 fusion된다. 즉 mask 자체를 별도 CNN으로 처리하는 것이
아니라, **RGB feature 중 얼룩 영역을 골라 요약하는 역할**이다.

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `use_stain_mask` | true | stain-local image feature를 observation condition에 포함한다. |
| `stain_mask_key` | `observations/images/stain_mask` | HDF5에서 mask를 읽을 dataset key다. |
| `stain_pooling_type` | `masked_mean` | mask에 포함된 spatial feature들의 평균을 사용한다. 현재 구현에서 지원하는 방식이다. |
| `empty_stain_feature_mode` | `zero` | mask가 비었을 때 stain feature를 0으로 둘지, `global` image feature를 대신 사용할지 정한다. |
| `stain_mask_threshold` | 0.5 | resize된 mask 값이 stain 영역인지 판정하는 threshold다. |
| `debug_stain_pooling` | false | 첫 batch에서 mask/feature 통계와 debug image를 생성해 pooling이 의도대로 작동하는지 확인한다. |

`use_stain_mask=true`이면 camera feature 외에 stain feature 한 개가 추가되므로
Polishing fusion 입력 차원도 `flow_image_feature_dim`만큼 커진다.

#### Marker encoder

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `flow_marker_feature_dim` | 128 | marker vector를 MLP로 변환한 feature 크기다. Gripper의 `single_cam_marker`처럼 marker observation mode일 때만 fusion에 추가된다. |

#### Gripper encoder — Gripper task 전용

Gripper position과 current 현재값은 각각 독립 MLP로 encoding한 뒤 current-state
feature로 합친다. 최근 position/current sequence는 별도 Joint GRU로 encoding하고,
두 branch를 다시 fusion해 하나의 gripper feature로 만든다. 이 값들은
`state_dim=9`에 포함되지 않으며 action의 마지막 두 차원을 예측하기 위한 gripper
상태 condition 역할을 한다.

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `gripper_encoder_hidden_dim` | 32 | position/current 각각의 scalar encoder 내부 hidden 크기다. |
| `gripper_feature_dim` | 64 | position feature와 current feature 각각의 출력 크기이자, 둘을 fusion한 최종 gripper feature 크기다. |
| `use_gripper_history` | true | Gripper task에서 최근 position/current causal history와 Joint GRU branch를 사용한다. `--no_gripper_history`이면 기존 MLP-only 구조로 학습한다. |
| `gripper_history_sec` | 0.5 | Dataset 30 Hz 기준 최근 0.5초를 observation으로 사용한다. |
| `gripper_history_len` | 15 | `round(dataset_hz × gripper_history_sec)`로 계산되는 history row 수다. |
| `gripper_history_hidden_dim` | 32 | Joint GRU hidden state 크기다. |
| `gripper_history_num_layers` | 1 | Joint GRU layer 수다. |
| `gripper_history_dropout` | 0.0 | GRU layer 사이 dropout이다. 1-layer에서는 PyTorch GRU 특성상 적용되지 않는다. |

#### Observation fusion

각 encoder 출력은 다음처럼 이어 붙인 후 fusion MLP로 압축한다.

```text
[state feature, camera feature(s), force feature,
 optional stain/marker feature, optional gripper feature]
                         ↓
             flow_global_cond_dim 차원
```

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `flow_global_cond_dim` | 256 | 모든 observation을 합친 최종 condition 크기다. 이 vector가 Conditional U-Net의 모든 residual block에 전달된다. |

각 개별 encoder dimension을 키워도 마지막에는 `flow_global_cond_dim=256`으로
압축된다. 따라서 앞단 feature만 무작정 크게 만들면 fusion layer의 입력과
파라미터 수만 증가하고, 최종 condition 병목은 그대로일 수 있다.

### 4. Action sequence와 Conditional 1D U-Net 파라미터

| 파라미터 | 기본값 | 기능과 Encoder 연계 |
|---|---:|---|
| `chunk_sec` | Polishing 4.27 / Gripper 5.33 | 한 번 학습·생성할 미래 action horizon을 실제 시간으로 표현한다. |
| `chunk_size` | Polishing 128 / Gripper 160 | Action sequence의 step 수이며 모델 내부의 `num_queries`가 된다. `chunk_sec>0`이면 dataset Hz로 다시 계산된다. 현재 U-Net down/up sampling 때문에 4의 배수여야 한다. |
| `train_seq_len` | 기본적으로 `chunk_size` | Dataset이 train action target으로 꺼내는 최대 길이다. 특별한 이유가 없다면 `chunk_size`와 같게 유지해야 한다. |
| `val_seq_len` | 기본적으로 `chunk_size` | Validation action target 길이다. 공정한 비교를 위해 train horizon과 같게 유지하는 것이 안전하다. |
| `flow_time_embed_dim` | 256 | Flow 생성 진행도 `t`를 sinusoidal embedding과 MLP로 encoding하는 크기다. 이 time feature와 observation의 global condition이 결합되어 U-Net block을 조건화한다. |
| `flow_down_dims` | `256,512,1024` | 1D U-Net 각 resolution의 channel 수다. action sequence의 시간축을 downsample하면서 넓은 temporal pattern을 학습한다. 가장 직접적으로 모델 크기와 GPU memory를 증가시키는 값 중 하나다. |
| `flow_kernel_size` | 5 | 1D convolution이 한 layer에서 인접 action step을 보는 범위다. 여러 layer와 downsampling을 거치므로 실제 receptive field는 이보다 훨씬 넓다. |
| `flow_n_groups` | 8 | U-Net의 GroupNorm group 기준값이다. batch size가 작아도 안정적으로 feature를 정규화하는 데 사용된다. |
| `flow_cond_predict_scale` | false | false이면 observation/time condition을 residual feature에 bias처럼 더한다. true이면 scale과 bias를 함께 예측하는 더 강한 conditioning이 되지만 파라미터와 불안정 가능성이 증가한다. |
| `flow_train_eps` | `1e-4` | 학습 시 Flow time을 정확한 0과 1에서 조금 떨어뜨려 endpoint 수치 문제를 피한다. 실제 robot 시간이나 control 주기와는 무관하다. |
| `flow_loss_type` | `mse` | 예측 velocity와 목표 velocity의 오차 함수다. `mse`는 큰 오차를 강하게 벌주고, `l1`은 outlier 영향이 상대적으로 작다. Padding step은 loss에서 제외된다. |
| `flow_infer_steps` | 10 | Inference 때 noise에서 action으로 적분하는 횟수다. 많을수록 근사가 세밀해지지만 한 번의 policy inference가 느려진다. 학습 action Hz나 ROS control Hz를 의미하지 않는다. |

`chunk_size`는 encoder feature 차원을 바꾸지는 않지만 U-Net이 동시에 생성해야
하는 시간축 길이를 바꾼다. 너무 길면 episode 끝의 padding 비율과 예측 난도가
커진다. Padding step 자체는 loss에서 제외되지만 끝부분 sample이 제공하는 유효
미래 target 수가 줄어든다. 반대로 너무 짧으면 물체 접근부터 파지·상승 또는
접촉부터 복귀까지 한 chunk에 담기지 않을 수 있다.

### 5. Dataset 구성과 sampling 파라미터

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `dataset_dir` | 자동 최신 dataset | 사용할 `imitation_form` directory를 직접 지정한다. 생략 시 task/camera 조건에 맞는 최신 dataset을 찾는다. |
| `num_episodes` | 0 | 0이면 발견된 episode를 모두 사용하고, 양수이면 최대 해당 개수만 사용한다. |
| `samples_per_episode` | 50 | 한 epoch에서 episode 하나당 뽑는 시작점 수다. Dataset 크기는 대략 `train episode 수 × samples_per_episode`가 된다. Action horizon 길이와는 다른 개념이다. |
| `resample_each_epoch` | true | true이면 다음 epoch에서 각 sample의 시작점을 다시 결정한다. sample 개수는 50으로 같지만 계속 동일한 50개 구간만 보지 않게 한다. Validation 시작점은 비교 일관성을 위해 고정된다. |
| `batch_size` | Polishing 12 / Gripper 8 | 한 optimizer step에서 동시에 처리하는 sample 수다. 이미지 encoder와 긴 action chunk 때문에 GPU memory 사용량에 큰 영향을 준다. |
| `seed` | 0 | episode train/validation 분할, sample 시작점, 모델 초기화 등 난수 재현에 사용된다. |
| `obs_mode` | `single_cam` | Observation 구성을 선택한다. Polishing은 single/dual, Gripper는 single/dual/single_cam_marker를 지원한다. |
| `camera_names` | mode에 따라 자동 | Dataset에서 읽고 image encoder에 전달할 camera 순서를 명시한다. |

`samples_per_episode`를 늘리면 encoder 크기는 변하지 않고 한 epoch의 batch 수가
증가한다. 반면 `chunk_size`를 늘리면 sample 하나의 action tensor와 U-Net 시간축
자체가 길어지므로 GPU memory와 학습 난도가 직접 증가한다.

### 6. Optimizer와 학습 안정화 파라미터

현재 optimizer는 AdamW다. 아래 값들은 observation/action의 의미나 encoder
shape를 바꾸지 않고, 모든 encoder와 U-Net weight가 업데이트되는 방법을 정한다.

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `num_epochs` | 500 | 전체 train dataset을 몇 번 반복할지 정한다. 기본 설정에서는 500 epoch 전체를 수행한다. |
| `lr` | `1e-4` | AdamW의 기준 learning rate다. 너무 크면 pretrained image encoder와 GRU가 불안정해지고, 너무 작으면 학습이 느리거나 underfit될 수 있다. |
| `weight_decay` | `1e-5` | AdamW가 큰 weight를 억제하는 정규화 강도다. 소규모 imitation dataset의 과적합 완화에 사용한다. |
| `beta1` | 0.95 | AdamW의 gradient 1차 moment 평균 계수다. 높을수록 update가 부드럽지만 최근 변화 반영은 느려진다. |
| `beta2` | 0.999 | Gradient 제곱의 2차 moment 평균 계수로 adaptive step 크기의 안정성에 관여한다. |
| `lr_scheduler` | `cosine` | Warmup 뒤 learning rate를 cosine 곡선으로 낮춘다. `none`이면 고정 LR을 사용한다. |
| `warmup_epochs` | 10 | 시작 LR을 10 epoch 동안 기준 LR까지 올린다. Random head와 pretrained backbone이 함께 업데이트되는 초반 충격을 줄인다. |
| `min_lr` | `1e-6` | Cosine decay 마지막의 최저 learning rate다. |
| `grad_clip_norm` | 1.0 | 모든 encoder와 U-Net gradient의 전체 norm이 기준을 넘으면 축소한다. Force transition/outlier batch의 폭발적인 update를 막는다. 0이면 비활성화된다. |
| `early_stopping_patience` | 0 | 기본적으로 early stopping을 사용하지 않는다. 양수 `N`을 지정할 때만 validation loss가 `N` epoch 연속 개선되지 않으면 종료한다. |

Scheduler와 gradient clipping은 feature를 직접 개선하는 encoder가 아니다. 다만
image backbone, force GRU, gripper/marker MLP와 U-Net이 같은 loss로 end-to-end
학습되므로, update가 안정되면 어느 한 encoder가 초기에 과도하게 변해 전체
condition을 망가뜨리는 현상을 줄일 수 있다.

### 7. DataLoader, checkpoint 및 실행 파라미터

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `num_workers` | 2 | CPU worker가 HDF5와 image sample을 병렬 준비한다. 모델 결과에는 영향을 주지 않지만 GPU 대기 시간을 줄일 수 있다. |
| `pin_memory` | true | CPU tensor를 page-locked memory에 두어 CUDA 전송을 빠르게 한다. |
| `persistent_workers` | true | epoch가 바뀌어도 worker process를 유지한다. 공유 epoch 값을 통해 train 시작점 재샘플링은 계속 적용된다. |
| `prefetch_factor` | 2 | worker 하나가 미리 준비할 batch 수다. 너무 크면 RAM 사용량이 증가한다. |
| `debug_batches` | 0 | epoch마다 앞쪽 몇 train batch의 loss를 추가 출력할지 정한다. Polishing에서는 `-1`로 모든 batch를 출력할 수 있다. |
| `save_every` | 50 | 매 50 epoch 완료 시 중간 checkpoint를 저장한다. 별도로 best와 last checkpoint는 관리된다. |
| `ckpt_root` | task별 기본 경로 | 새 학습 결과가 저장될 상위 directory다. |
| `ckpt_dir` | 없음 | 명시하면 train/eval checkpoint 위치를 직접 지정한다. Eval에서는 `policy_best.ckpt`와 `dataset_stats.pkl`을 찾는 기준이 된다. |
| `eval` | false | 새 학습 대신 checkpoint와 stats가 현재 모델 구성으로 load되는지 확인한다. |
| `train_all_obs_modes` | false | 지원되는 observation mode를 순서대로 모두 학습한다. |
| `shared_timestamp` | true | 여러 observation mode를 연속 학습할 때 동일한 timestamp 이름을 사용한다. |

#### DataLoader worker와 prefetch가 학습 속도에 미치는 영향

이 프로젝트의 sample 하나를 준비하려면 HDF5 file을 열고, 현재 시점의 RGB
frame과 수치 데이터를 읽고, image/tensor 변환과 normalization을 수행해야 한다.
`num_workers=0`이면 이 작업과 GPU 학습이 다음처럼 대부분 직렬로 진행된다.

```text
CPU: batch 읽기·변환 ────────────── 다음 batch 읽기·변환 ──────────────
GPU:                 forward/backward                 forward/backward
                     ↑ CPU가 끝날 때까지 대기         ↑ 다시 대기
```

GPU 계산이 빠르더라도 CPU가 HDF5와 image batch를 준비하는 동안 GPU가 쉬기 때문에
전체 속도는 model 연산이 아니라 data input pipeline에 의해 제한될 수 있다.

`num_workers=2`, `prefetch_factor=2`에서는 worker 두 개가 main training process와
별도로 sample을 읽는다. Worker 하나당 최대 2개 batch를 미리 준비하므로 정상
상태에서는 다음 batch들이 queue에 대기하고, 현재 batch의 GPU 연산과 다음 batch의
CPU 준비가 겹쳐진다.

```text
Worker 0: batch 1 준비 ─ batch 3 준비 ─ batch 5 준비 ─ ...
Worker 1: batch 2 준비 ─ batch 4 준비 ─ batch 6 준비 ─ ...
GPU:              batch 1 학습 ─ batch 2 학습 ─ batch 3 학습 ─ ...
                  ↑ 준비된 batch를 기다림 없이 소비
```

각 옵션의 역할은 서로 다르다.

- `num_workers=2`: HDF5 read와 image 변환을 두 CPU process에서 병렬 처리한다.
- `prefetch_factor=2`: 각 worker가 현재 요청보다 앞선 batch를 미리 준비한다. 현재
  설정에서는 이론적으로 worker queue에 최대 `2 workers × 2 batches`가 준비될 수
  있다.
- `pin_memory=true`: 준비된 CPU tensor를 page-locked memory로 옮겨
  `.to(device, non_blocking=True)` CUDA 전송이 CPU 작업과 더 잘 겹치게 한다.
- `persistent_workers=true`: epoch가 끝나도 worker를 종료하지 않아 다음 epoch마다
  process와 Dataset worker를 다시 만드는 초기화 비용을 제거한다. 따라서 첫
  epoch보다 이후 epoch에서 이점이 더 안정적으로 나타난다.

20260721 Gripper 재학습에서는 train sample 수가 2,250개, batch size가 8로
동일하여 epoch당 282 batch가 유지됐는데도 처리량이 약 `4.6 batch/s`로 상승하고
epoch 시간이 약 64초로 감소했다. 즉 sample을 생략해서 빨라진 것이 아니라 같은
batch를 더 빠르게 공급한 것이다. `chunk_size 200 → 160`으로 U-Net 시간축이
20% 짧아진 효과도 함께 존재하지만, 관측된 약 2배의 처리량 차이는 worker,
prefetch, pinned memory에 의한 GPU 대기 감소가 크게 기여한 것으로 판단한다.

다음 현상이 보이면 data loading 병목일 가능성이 높다.

- GPU utilization이 높은 값으로 유지되지 않고 주기적으로 0% 근처까지 떨어짐
- GPU memory는 할당돼 있지만 batch 진행 속도가 느림
- `num_workers=0`에서 CPU 사용은 낮고 batch 사이에 긴 공백이 있음
- Model 크기를 조금 줄여도 epoch 시간이 거의 개선되지 않음

반대로 worker 수를 무조건 늘리는 것은 좋지 않다. Worker마다 HDF5 handle, image
tensor와 prefetched batch를 가지므로 RAM 사용량이 증가한다. Storage가 하나의
느린 disk라면 worker가 너무 많을 때 random read 경쟁으로 오히려 느려질 수 있고,
HDF5/multiprocessing 환경에 따라 worker 오류가 발생할 수도 있다. 권장 tuning
순서는 `num_workers=0 → 2 → 4`이며, 각 설정에서 동일한 batch 수의 `it/s`, GPU
utilization과 RAM 사용량을 비교한다. 현재 single-camera HDF5 dataset에서는
`num_workers=2`, `prefetch_factor=2`를 보수적인 기본값으로 사용한다.

DataLoader 관련 값을 바꿔도 network 구조나 최종 policy 수식은 바뀌지 않는다.
다만 storage가 느리거나 RAM이 부족하면 worker/prefetch를 너무 크게 했을 때 오히려
느려질 수 있다. HDF5 오류나 메모리 부족이 발생하면 먼저 `num_workers=0`,
`--no_persistent_workers`, `--no_pin_memory` 조합으로 원인을 분리할 수 있다.

### 8. 파라미터를 바꿀 때 함께 확인할 조합

- `dataset_hz`를 바꾸면 같은 실제 시간을 유지하도록 `force_history_len`과
  `chunk_size`가 다시 계산되는지 확인한다.
- `force_history_len`은 GRU 입력 시간 길이, `force_encoder_hidden_dim`은 GRU가
  그 시간을 요약한 feature 크기다. 두 값은 역할이 다르다.
- `flow_image_feature_dim`, `flow_marker_feature_dim`, force/gripper feature는 모두
  concat된 후 `flow_global_cond_dim`으로 압축된다.
- `chunk_size`, `train_seq_len`, `val_seq_len`은 특별한 실험이 아니라면 동일하게
  유지한다.
- `action_dim`, `state_dim`, `marker_dim`은 HDF5 tensor와 checkpoint 구조를
  결정하므로 기존 checkpoint를 사용할 때 임의로 변경하면 안 된다.
- Network dimension 또는 encoder 사용 여부를 바꾸면 기존 checkpoint와 shape가
  달라질 수 있으므로 새 checkpoint directory에서 재학습한다.
- 학습 성능 비교 시 한 번에 여러 축을 바꾸기보다 dataset split과 `seed`를
  고정하고 변경 목적별 ablation을 남기는 것이 좋다.

## 20260721 — Flow 학습 파라미터 정비

적용 대상은 Polishing의 `flow_train_core.py`와 Gripper의
`train_flow_gripper.py` 및 두 코드가 공통으로 사용하는 데이터 로더이다.

`samples_per_episode=50`, `num_epochs=500`은 비교 조건을 유지하기 위해
변경하지 않았다. Force encoder의 `hidden_dim=64`, `num_layers=1`,
`dropout=0.0`도 기존 값이 이미 권장 구성과 같아서 그대로 유지했다.

### 기본값 변경

| 파라미터 | Polishing 이전 | Polishing 변경 | Gripper 이전 | Gripper 변경 | 근거 |
|---|---:|---:|---:|---:|---|
| `force_history_len` | 10 | 30 | 10 | 30 | 30 Hz 데이터에서 최근 1초의 힘 변화를 관찰해 접촉 진입·유지·해제 추세를 구분하기 위함 |
| `chunk_size` | 200 | 128 | 200 | 160 | 전체 episode보다 과도하게 긴 padding 구간의 학습 비중을 줄이고, 각 task의 유효 동작 구간에 맞춘 horizon 사용 |
| `weight_decay` | `1e-6` | `1e-5` | `1e-6` | `1e-5` | 소규모 imitation dataset에서 과적합을 완화하기 위함 |
| `save_every` | 100 | 50 | 100 | 50 | 장시간 학습 중간 결과의 보존 간격을 줄이기 위함 |
| `debug_batches` | 3 | 0 | 3 | 0 | 매 epoch 반복 출력과 I/O 부하를 제거하기 위함 |
| `num_workers` | 0 | 2 | 0 | 2 | HDF5/image batch 준비와 GPU 학습의 대기 시간을 줄이기 위함 |
| `pin_memory` | false | true | false | true | CUDA 전송 시 page-locked memory를 사용하기 위함 |
| `persistent_workers` | false | true | false | true | epoch마다 DataLoader worker를 다시 생성하는 비용을 줄이기 위함 |

### 추가된 시간축 파라미터

| 파라미터 | Polishing | Gripper | 근거 |
|---|---:|---:|---|
| `dataset_hz` | 30.0 | 30.0 | recorder가 만든 학습 row의 시간축을 명시 |
| `force_history_sec` | 1.0 | 1.0 | force history를 row 개수가 아닌 실제 시간으로 정의 |
| `chunk_sec` | 4.27 | 5.33 | Polishing 128 step, Gripper 160 step의 action horizon을 시간으로 정의 |

학습 시작 시 초 단위 값은 `dataset_hz`를 이용해 step 수로 환산한다.
기본 설정에서는 각각 `round(30×1.0)=30`, `round(30×4.27)=128`,
`round(30×5.33)=160`이 된다. 현재 Flow U-Net 구조 때문에 최종
`chunk_size`는 4의 배수여야 하며, 조건을 만족하지 않으면 즉시 오류를 낸다.

초 단위 설정이 0보다 크면 그것이 기준값이다. 기존처럼 step 수를 직접
지정하려면 해당 초 단위 옵션을 0으로 함께 지정한다. 예를 들어
`--force_history_sec 0 --force_history_len 45` 또는
`--chunk_sec 0 --chunk_size 200` 형태로 사용한다.

### 추가된 학습 안정화 파라미터

| 파라미터 | 기본값 | 근거 |
|---|---:|---|
| `lr_scheduler` | `cosine` | 고정 learning rate 대신 학습 후반의 파라미터 진동을 줄임 |
| `warmup_epochs` | 10 | pretrained image backbone을 포함한 초기 gradient 급변을 완화 |
| `min_lr` | `1e-6` | cosine decay의 최저 learning rate를 제한 |
| `grad_clip_norm` | 1.0 | force transition이나 outlier batch에서 발생할 수 있는 큰 gradient를 제한 |
| `early_stopping_patience` | 0 | 기본 비활성화. 500-epoch cosine schedule을 끝까지 수행하며 필요할 때만 양수 patience로 활성화 |
| `resample_each_epoch` | true | 매 epoch episode 내 시작 index를 다시 선택해 고정된 50개 crop만 반복 학습하는 현상을 방지 |

Warmup과 cosine scheduler 상태는 checkpoint에도 함께 저장한다. Validation은
각 epoch의 train update 이후 실행하며, 그 결과로 `policy_best.ckpt`를 갱신한다.
Early stopping은 `early_stopping_patience > 0`일 때만 판단한다. Validation
dataset의 시작 index는 비교 일관성을 위해 고정하고 train dataset만 epoch별로
재샘플링한다.

각 기능은 명령행에서 `--lr_scheduler none`, `--grad_clip_norm 0`,
`--no_resample_each_epoch`, `--no_pin_memory`, `--no_persistent_workers`로 개별
비활성화할 수 있다. Early stopping이 필요한 실험에서만
`--early_stopping_patience 100`처럼 양수를 지정한다.

## 20260722 Patch — Gripper history observation과 temporal encoder

> **상태:** 20260721에 적용한 gripper position observation 정규화에 이어,
> 20260722에 causal gripper history, Joint GRU encoder, 학습 metadata 및 ROS
> inference 동기화 buffer까지 구현을 완료했다. 이 구조는 새 checkpoint 재학습이
> 필요하며 기존 MLP-only checkpoint와 자동으로 혼합하지 않는다.

### 문제 정의

기존 Gripper Flow policy는 gripper observation으로 한 시점의
`present_position`과 `present_current_mA`만 사용했다. 두 scalar는 각각 독립
MLP를 거친 뒤 하나의 gripper feature로 합쳐졌기 때문에 observation encoder가
직전의 변화 방향을 알 수 없었다.

단일 시점의 같은 position/current 값은 다음처럼 서로 다른 상황에서 나타날 수
있다.

- 물체에 닿지 않은 채 손가락이 닫히는 중
- 물체에 처음 닿아 current가 증가하기 시작한 상태
- 물체를 정상적으로 잡고 position/current가 유지되는 상태
- 완전히 닫혔지만 물체를 잡지 못한 상태
- 파지한 물체가 미끄러지면서 position/current가 변하는 상태
- 높은 goal current, 기구부 마찰 또는 stall 때문에 current가 상승한 상태

Action chunk는 미래 sequence를 모델링하지만 과거 observation을 복원하지는
않는다. 따라서 최근 position/current 변화를 causal history로 제공하는 것은
현재 gripper 상태의 부분 관측성을 직접 줄이는 접근이다.

### 연구 사례와의 관계

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf)는
  단일 observation보다 짧은 observation history를 사용하는 것이 유리할 수
  있음을 보였고, 대부분의 실험에서 2 step을 좋은 절충값으로 보고했다. 동시에
  긴 history가 항상 더 좋은 것은 아니라는 결과도 제시한다.
- [Octo](https://octo-models.github.io/)는 image와 proprioception을 포함하는
  observation history를 Transformer가 처리할 수 있도록 설계됐다.
- [Bi-ACT](https://ras.papercept.net/images/temp/AIM/files/0129.pdf)는 joint angle,
  velocity, torque를 함께 입력하고 미래 position/velocity/torque chunk를
  예측한다. 또한 단위가 다른 각 신호를 개별 정규화해 입력 편향을 줄인다.
- NYU·UC Berkeley의
  [Feel the Force](https://hrcm-workshop.github.io/2025/abstracts/adeniji_abstract9.pdf)는
  gripper state와 연속 force history를 별도 token으로 encoding하고 미래
  gripper state와 force를 함께 예측한다. 현재 아이디어와 가장 가까운 공개
  사례지만, motor current가 아니라 tactile force를 사용하고 Transformer 및
  저수준 force controller를 사용한다는 차이가 있다.
- [ALOHA/ACT](https://arxiv.org/abs/2304.13705)는 현재 joint position과 image만으로
  action chunk를 예측해 높은 성공률을 보였다. 즉 history encoder가 모든 gripper
  IL 시스템에 필수인 것은 아니며, demonstration과 저수준 controller가 충분하면
  현재 상태만으로도 동작할 수 있다.

공개 연구를 종합하면 **position/load 계열 신호를 함께 사용하는 것**과
**필요할 때 짧은 observation history를 추가하는 것**은 모두 타당하다. 다만
Joint GRU 자체가 유일한 표준은 아니며, observation stacking, MLP token,
Transformer 또는 명시적인 velocity 입력으로 같은 문제를 풀기도 한다. 현재
dataset 규모와 기존 GRU 구현을 고려하면 작은 Joint GRU가 가장 단순한 첫
실험이다.

### 적용된 architecture

기존 현재값 MLP를 제거하지 않고, history branch를 병렬로 추가하는 hybrid
구조를 적용했다.

```text
현재 normalized position/current ── 기존 Gripper MLP ── current feature ─┐
                                                                          ├─ fusion
최근 normalized position/current ── Joint GRU ───────── history feature ─┘
                                                                              │
                                                                              ▼
                                                                  gripper feature 64
                                                                              │
                    robot state/image/force/marker feature와 함께 global fusion
```

현재값 branch는 절대적인 개방 폭과 부하를 명확하게 전달한다. History branch는
닫힘 방향, 접촉 시점, current 상승·유지·감소와 같은 동적 관계를 요약한다.
GRU 마지막 hidden state만 사용하고, 두 branch를 fusion MLP로 합쳐 최종
`gripper_feature_dim=64`를 유지하면 downstream global fusion의 입력 크기와
변경 범위를 최소화할 수 있다.

이번 구현의 history 입력은 다음 두 값으로 제한했다.

```text
gripper_history[t] = [normalized_present_position,
                      normalized_present_current]
```

후속 실험에서는 필요할 때만 아래 값을 추가한다.

```text
Δposition, Δcurrent,
previous_goal_position, previous_goal_current
```

특히 `present_current_mA`는 실제 파지력이 아니라 motor torque/load의 proxy다.
접촉뿐 아니라 이동 속도, 마찰, stall 및 명령한 goal current의 영향을 받으므로,
두 값만으로 접촉 원인을 구분하지 못한다면 **이전 executed command history**를
추가하는 것이 단순히 GRU hidden dimension을 키우는 것보다 우선이다.

### 20260721 선행 적용 완료: Gripper position observation 정규화

`source/data/dataset.py`에서 gripper position/current observation을 모두 training
split의 min/max로 정규화하도록 변경했다. 새 학습의 `dataset_stats.pkl`에는
`gripper_position_min/max`, `gripper_current_min/max`, `gripper_norm_mode`가 함께
저장되며 inference가 같은 통계를 사용한다.

```text
present_position → gripper_position_min/max로 정규화
present_current  → gripper_current_min/max로 정규화 유지
```

Position과 current의 scale이 다르면 한 modality가 fusion을 지배할 수 있고,
GRU가 시간 관계보다 절대 scale 차이를 먼저 학습할 수 있다. 기존 action은 이미
전체 action min/max로 정규화되므로 observation 쪽 position 처리도 일관되게
맞췄다.

이 정규화 변경은 기존 checkpoint의 입력 분포를 바꾸므로 old checkpoint에
강제로 적용하면 안 된다. Inference는 position stats가 없는 기존 checkpoint에는
raw position을 유지하고 경고하며, 새 checkpoint에는 저장된 position stats를
자동 적용한다.

### 적용된 기본 파라미터

30 Hz dataset에서 0.5초 history를 기본값으로 적용했다.

```text
use_gripper_history = true
gripper_history_sec = 0.5
gripper_history_len = 15       # round(30 Hz × 0.5 s)
gripper_history_input_dim = 2  # position + present current
gripper_history_hidden_dim = 32
gripper_history_num_layers = 1
gripper_history_dropout = 0.0
gripper_feature_dim = 64       # 기존 downstream interface 유지
```

권장 ablation 순서는 다음과 같다.

| 실험 | Position 정규화 | History | 목적 |
|---|---|---:|---|
| A | 기존 checkpoint의 raw 입력 | 없음 | 과거 checkpoint의 참고 결과 |
| B | 적용 | 없음 | 정규화 효과만 분리한 새 MLP baseline |
| C | 적용 | 6 step, 0.2초 | 매우 짧은 motor/contact transient 확인 |
| D | 적용 | 15 step, 0.5초 | 첫 권장 Joint GRU 설정 |
| E | 적용 | 30 step, 1.0초 | D가 유효할 때만 긴 context 효과 확인 |

이전에 force history를 과도하게 늘렸을 때 성능이 악화된 경험과 공개 연구의 짧은
observation horizon 결과를 고려하면, 처음부터 `L=30` 또는 그 이상을 기본값으로
두는 것은 권장하지 않는다.

### 적용 내역

#### 1. Dataset과 normalization

대상: `source/data/dataset.py`

1. Gripper position에 `gripper_position_min/max` 정규화를 적용했다.
2. `_force_history()`와 같은 causal helper로 `_gripper_history()`를 추가했다.
3. 출력 shape을 `(L, 2)`로 고정하고 episode 시작부는 가장 오래된 관측값을
   앞쪽에 반복해 padding한다.
4. Train과 inference가 같은 channel 순서와 normalization을 사용하도록
   `dataset_stats.pkl`에 history schema를 저장한다.
5. Train/validation 모두 현재 index까지의 과거 값만 사용하는 causal sequence로
   구성해 미래 observation 누출을 막았다.

#### 2. Gripper encoder

대상: `source/models/gri_encoder.py`, `source/models/gri_flow_core.py`

1. 기존 `GripperObservationEncoder`의 current-state MLP branch를 유지했다.
2. `nn.GRU(input_size=2, hidden_size=32, num_layers=1)` history branch를 추가했다.
3. `[current_feature, history_hidden]`을 concatenate한다.
4. `Linear + LayerNorm + Mish` fusion으로 다시 64차원 gripper feature를 만든다.
5. `use_gripper_history=false`이면 history module 자체를 생성하지 않아 기존
   MLP-only checkpoint 구조를 그대로 재구성한다.
6. History가 활성화됐는데 tensor가 누락된 경우 current 한 점으로 조용히
   fallback하지 말고 오류를 내어 train/inference wiring 문제를 드러낸다.

#### 3. 학습 entrypoint와 checkpoint metadata

대상: `scripts/flow/train_flow_gripper.py`

다음 CLI/config 항목을 추가했다.

```text
--use_gripper_history / --no_gripper_history
--gripper_history_sec
--gripper_history_len
--gripper_history_hidden_dim
--gripper_history_num_layers
--gripper_history_dropout
```

학습의 `gripper_history_len`은 별도 Hz를 중복 정의하지 않고 기존 `dataset_hz`와
`gripper_history_sec`로 계산한다. Policy config와 dataset stats에는 활성화 여부,
최종 length, channel schema 및 encoder dimension을 모두 저장한다.

기존 checkpoint는 history GRU weight가 없으므로 새 구조에 강제로 load하지 않는다.
Eval/inference는 checkpoint config를 읽어 old MLP-only 모델과 new history 모델을
구분하고 gripper policy를 `strict=True`로 load한다. 구조가 다르면 즉시 오류가
발생한다.

#### 4. Inference observation buffer

대상: `behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py`

1. Header가 없는 `present_position`과 `present_current_mA` callback의 monotonic 수신
   시간을 기준으로 approximate-time pair를 만든다.
2. 기본 `sync_slop=20 ms` 안에 들어온 pair만 30 Hz causal ring buffer에 넣고,
   매칭되지 않은 오래된 message는 drop counter에 기록한다.
3. Buffer가 아직 차지 않은 시작 구간은 첫 valid pair를 반복해 채운다.
4. 매 inference 요청에서 최신 시점까지의 정확히 `L=15`개 sample만 모델에 전달한다.
5. Checkpoint의 `dataset_hz`, history length, channel 순서와 runtime 구성이 다르면
   시작 단계에서 명확한 오류를 낸다.
6. 기본 `max_age=0.20 s`보다 최신 pair가 오래되면 inference plan 생성을 보류한다.
7. Pair count, timestamp skew, drop count, buffer fill ratio를
   `[GRIPPER-HISTORY]` log로 출력한다.

Recorder에서 이미 동기화된 30 Hz row를 만드는 것과 별개로, online inference도
position/current pair의 timestamp 차이가 과도하지 않은지 확인해야 한다. 단순히
callback 도착 순서대로 서로 다른 시점의 최신값을 묶으면 training history와 다른
입력 분포가 생길 수 있다.

### 실행 방법

새 history 모델은 Gripper 학습 기본값이므로 별도 옵션 없이 실행한다.

```bash
cd ~/nrs_imitation/scripts/flow
python3 train_flow_gripper_single_cam.py
```

MLP-only ablation을 새 normalization 조건에서 다시 만들 때만 다음을 사용한다.

```bash
python3 train_flow_gripper_single_cam.py --no_gripper_history
```

새 history checkpoint inference launch의 기본값은 학습과 같은 30 Hz, 15 row다.

```text
use_gripper_history:=true
gripper_history_hz:=30.0
gripper_history_len:=15
gripper_history_sync_slop_sec:=0.020
gripper_history_max_age_sec:=0.20
```

20260721 이전 MLP-only checkpoint를 재현할 때는 반드시
`use_gripper_history:=false`를 전달한다. Checkpoint와 runtime의 history 활성화,
길이, Hz 또는 channel schema가 다르면 inference node는 시작을 거부한다.

### 평가 방법

Validation loss만으로 파지 개선 여부를 결정하지 않는다. 대부분의 episode가
접근·이동 구간이면 전체 loss가 낮아져도 짧은 파지 transition은 개선되지 않을 수
있다. 동일 dataset split, seed 및 나머지 hyperparameter를 고정하고 다음 지표를
따로 기록한다.

- 물체 근처까지 정상 접근한 비율
- Gripper closing을 시작한 비율
- 손가락이 물체에 걸린 실제 파지 성공률
- 파지 후 물체를 작업면에서 들어 올린 성공률
- 파지 중 position/current 유지와 slip 발생 횟수
- 빈손 완전 닫힘과 정상 파지를 구분한 비율
- Closing 시작부터 current 상승, 파지, lift 시작까지 걸린 시간

각 설정당 동일한 초기 조건에서 최소 10회, 가능하면 20회 real rollout을 수행한다.
동시에 아래 시계열을 저장하면 실패 원인을 구분하기 쉽다.

```text
present_position
present_current
executed goal position/current
predicted gripper position/current action chunk
robot TCP Z
```

### 예상 위험과 중단 기준

- 50 episode에서 파지 transition sample이 적다면 GRU가 일반 이동 구간만 학습할
  수 있다. 이 경우 encoder 크기보다 event 주변 sample coverage를 먼저 확인한다.
- Current는 tactile force가 아니므로 물체 종류, 마찰 및 motor 온도 변화에 따라
  같은 힘에서도 값이 달라질 수 있다.
- History가 길수록 좋은 것이 아니며, 오래된 closing 정보가 현재 상태 판단을
  흐릴 수 있다.
- Joint GRU를 추가해도 실제 command와 measured current 관계가 없으면 접촉과
  높은 goal current를 구분하기 어려울 수 있다.
- B 대비 C/D의 real grasp/lift 성공률이 개선되지 않으면 hidden dimension이나
  layer를 바로 키우지 말고 먼저 history 정렬, normalization, transition sample
  수와 previous command 필요성을 점검한다.

20260722 패치의 범위는 **정규화 + causal history observation + small Joint GRU**로
제한했다. 명시적인 grasp phase, contact rule 또는 저수준 force controller는 이
실험에 섞지 않는다. 그래야 성능 변화가 history encoder 때문인지 분리해서 판단할
수 있다. 이후 필요하다면 Feel the Force처럼 policy가 desired force를 예측하고
별도 closed-loop controller가 이를 추종하는 구조를 독립적인 후속 연구로 검토한다.
