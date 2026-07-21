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
| `norm_mode` | `minmax_m11` | qpos, action, force history, marker 및 gripper current를 dataset min/max 기준으로 정규화한다. `minmax_m11`은 대략 `[-1,1]`, `minmax_01`은 `[0,1]` 범위다. 학습과 inference가 반드시 같은 stats와 mode를 사용해야 한다. |

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

현재 gripper position과 current는 각각 독립 MLP로 encoding한 뒤 다시 합쳐 하나의
gripper feature로 만든다. 이 값들은 `state_dim=9`에 포함되지 않으며, action의
마지막 두 차원을 예측하기 위한 현재 gripper 상태 condition 역할을 한다.

| 파라미터 | 기본값 | 기능 |
|---|---:|---|
| `gripper_encoder_hidden_dim` | 32 | position/current 각각의 scalar encoder 내부 hidden 크기다. |
| `gripper_feature_dim` | 64 | position feature와 current feature 각각의 출력 크기이자, 둘을 fusion한 최종 gripper feature 크기다. |

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
| `num_epochs` | 500 | 전체 train dataset을 최대 몇 번 반복할지 정한다. Early stopping으로 더 일찍 끝날 수 있다. |
| `lr` | `1e-4` | AdamW의 기준 learning rate다. 너무 크면 pretrained image encoder와 GRU가 불안정해지고, 너무 작으면 학습이 느리거나 underfit될 수 있다. |
| `weight_decay` | `1e-5` | AdamW가 큰 weight를 억제하는 정규화 강도다. 소규모 imitation dataset의 과적합 완화에 사용한다. |
| `beta1` | 0.95 | AdamW의 gradient 1차 moment 평균 계수다. 높을수록 update가 부드럽지만 최근 변화 반영은 느려진다. |
| `beta2` | 0.999 | Gradient 제곱의 2차 moment 평균 계수로 adaptive step 크기의 안정성에 관여한다. |
| `lr_scheduler` | `cosine` | Warmup 뒤 learning rate를 cosine 곡선으로 낮춘다. `none`이면 고정 LR을 사용한다. |
| `warmup_epochs` | 10 | 시작 LR을 10 epoch 동안 기준 LR까지 올린다. Random head와 pretrained backbone이 함께 업데이트되는 초반 충격을 줄인다. |
| `min_lr` | `1e-6` | Cosine decay 마지막의 최저 learning rate다. |
| `grad_clip_norm` | 1.0 | 모든 encoder와 U-Net gradient의 전체 norm이 기준을 넘으면 축소한다. Force transition/outlier batch의 폭발적인 update를 막는다. 0이면 비활성화된다. |
| `early_stopping_patience` | 30 | Validation loss가 연속 30 epoch 동안 한 번도 더 낮아지지 않으면 종료한다. 0이면 비활성화된다. |

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
| `early_stopping_patience` | 30 | validation loss가 30 epoch 연속 개선되지 않으면 불필요한 추가 학습을 종료 |
| `resample_each_epoch` | true | 매 epoch episode 내 시작 index를 다시 선택해 고정된 50개 crop만 반복 학습하는 현상을 방지 |

Warmup과 cosine scheduler 상태는 checkpoint에도 함께 저장한다. Validation은
각 epoch의 train update 이후 실행하며, 그 결과로 `policy_best.ckpt`와 early
stopping을 판단한다. Validation dataset의 시작 index는 비교 일관성을 위해
고정하고 train dataset만 epoch별로 재샘플링한다.

각 기능은 명령행에서 `--lr_scheduler none`, `--grad_clip_norm 0`,
`--early_stopping_patience 0`, `--no_resample_each_epoch`,
`--no_pin_memory`, `--no_persistent_workers`로 개별 비활성화할 수 있다.
