# OPSD Fork Suppression Research Baseline

연구팀 4명이 Qwen3-1.7B의 privileged on-policy self-distillation(OPSD)을 공통 출발점으로 사용하기 위한 저장소입니다. **기본 학습은 Vanilla OPSD**이며 진단과 mitigation은 명시적으로 켜야 작동합니다. Fork Mask의 효과를 입증한 저장소가 아니라, 효과를 검증하기 위한 연구 기반입니다.

## 연구 배경과 근거

[공식 논문, arXiv:2607.05184v1](https://arxiv.org/html/2607.05184v1)의 Table 1에서 Qwen3-1.7B의 AIME24/AIME25/HMMT25 평균 avg@16은 Base 0.372에서 OPSD 0.308로 감소합니다. 4,096 completion-token으로 학습하고 38,912-token으로 평가한 결과입니다. 짧은 budget의 개선을 긴 reasoning 성능 개선으로 해석하면 안 됩니다. 논문 §4.1은 privileged context가 thinking 모델의 fork 비율을 줄이고 lock 비율을 높이는 현상을 보고합니다. §4.2의 self-correction 토큰에 대한 teacher–student log-ratio 분석과 §4.3의 어휘 표지는 이 현상을 뒷받침하는 진단이며, 인과적 증명이나 같은 metric은 아닙니다.

[공식 upstream](https://github.com/princeton-pli/rethinking-opsd-for-thinking-models) 기준 커밋:
`46ccfbddeb92c3bb8b9602a1e6644eb9fa8f7c04`.
원본 사용법은 [README.upstream.md](README.upstream.md)에 보존합니다. Apache-2.0 LICENSE와 기존 data/eval/SLURM 기능을 유지합니다.

## 설정과 비교 실험

| 항목 | 공식 dense OpenThoughts 설정 / 본 저장소의 기준 |
|---|---|
| Model | Qwen/Qwen3-1.7B |
| Student | Thinking ON, problem만 입력, student on-policy rollout |
| Teacher | 같은 **초기** checkpoint의 frozen copy, EMA OFF (`sync_ref_model=false`) |
| Privileged context | `solution` 전체를 user message 안의 예시 풀이로 추가 |
| Loss | generalized JSD α=0.5, β=0, full vocabulary |
| Data | cleaned OpenThoughts의 첫 15,000행; `problem` / `solution` |
| Batch / LR | 8 GPU × microbatch 1 × accumulation 8 = 64; 5e-6 |
| Schedule | 1 epoch, cosine, warmup 0.03, seed 42, bf16, FSDP |
| Training sampling | T=1, top-p=1, top-k=50, 1 sample/prompt |
| Official cap | prompt 25,000, completion 4,096 |
| Official evaluation | 16 samples/problem, T=.6, top-p=.95, 38,912 tokens |

근거는 `submit_experiment.sh`의 `qwen3_1p7b_ot15k_dense`, `slurm/train.sh`, `slurm/config.sh`, `train.py`, `opsd/trainer.py`입니다. `train.py` 단독 기본값은 논문 recipe와 다릅니다(예: α=0, teacher EMA ON). **팀 실험은 반드시 config launcher를 사용하세요.** `enable_thinking=true`는 upstream trainer에서 student/teacher 양쪽 chat template에 적용됩니다. Teacher는 별도로 정답을 생성해 대체하는 방식이 아니라 student가 생성한 동일 prefix의 next-token 분포를 채점합니다. Teacher template는 probe config에도 동일하게 기록했습니다.

| 실험 | config / 모델 | 역할 |
|---|---|---|
| Base | `models/Qwen3-1.7B` | 추가 학습 없이 같은 평가 |
| Vanilla OPSD | `configs/research/vanilla_4096.json` | 주 비교 학습, masking OFF |
| Fork Mask | `configs/research/fork_4096.json` | 팀 가설: student pmax<.25 위치의 loss 제외 |
| Random Mask | `configs/research/random_4096.json` | 같은 rollout 내 같은 수의 eligible 토큰 무작위 제외 |
| Short diagnostic training | `configs/research/vanilla_1024.json` | 선택 사항; 공식 recipe와 구별 |
| Paired probe | `configs/research/probe_1024.json` | 신규 기본 진단, 1,024-token rollout |

기존 **Student OFF / Teacher ON** 결과는 `legacy-off-on`으로 별도 분류하세요. 본 ON/ON recipe와 합산하지 않습니다. 공식 upstream의 `enable_thinking=false`는 양쪽에 적용되므로 그것만으로 OFF/ON 결과를 재현했다고 주장할 수 없습니다. 기존 4,096-token checkpoint와 결과는 재학습하지 않고 probe/evaluation 입력으로 재사용합니다. 기존 결과 파일은 아직 제공되지 않았고 이 저장소에 포함하지 않습니다.

## 저장소 구조

```text
train.py                       upstream 진입점 + opt-in research CLI
opsd/{trainer,config}.py        원래 trainer + loss 직전의 opt-in hook
configs/fsdp_config_*.json      upstream FSDP
configs/research/              팀 학습 및 probe JSON configs
research/
  run.py                       config 검증, torchrun 실행, manifest
  probe.py                     동일 trajectory의 base/sparse/dense teacher 채점
  signals.py                   위치별 확률·entropy·JSD·log-ratio 저장
  masking.py                   attention을 바꾸지 않는 loss-only masking
  metrics.py                   fork/lock, avg/pass@k, 길이·EOS·어휘 proxy
  analyze.py                   위치 통계, rollout macro 평균, 결과 집계
data_tools/                    공식 dataset 전처리
evaluation/                    공식 task registry와 정답 채점
run_generate.py                공식 vLLM 평가 + opt-in seed/response metadata
run_eval.py, run_budget_sweep.py 공식 채점·budget 분석
envs/train/{pyproject.toml,uv.lock} upstream 고정 환경
envs/setup.sh, requirements.txt upstream 설치 경로
submit_experiment.sh, slurm/    공식 SLURM 실행 경로 유지
tests/                         CPU 단위·실제 loss 호환성 검사
.github/workflows/             CPU CI
docs/                          출처·검증 기록
runs/, models/, data/          로컬 자산, Git 제외
```

## Python / CUDA 환경

학습은 **Linux x86_64 + NVIDIA GPU**에서 실행합니다. Windows는 CPU 테스트만 확인했습니다. 팀 전원 같은 Git commit과 upstream lock을 사용합니다. Python 3.10, torch 2.9.0, transformers 4.57.1, TRL .24.0, vLLM .12.0, accelerate 1.11.0이 공식 기준입니다. torch lock에는 CUDA 12.8 runtime 계열이 포함됩니다. 호스트 NVIDIA driver의 해당 runtime 지원을 확인하세요. CUDA toolkit을 임의로 바꿔 lock과 다른 torch를 설치하지 마세요. 현재 환경에서 Linux/CUDA 설치 자체는 실행 검증하지 않았습니다.

```bash
git clone https://github.com/msjk27/opsd-fork-suppression-baseline.git
cd opsd-fork-suppression-baseline
# uv를 먼저 설치한 Linux 환경
bash envs/setup.sh
source envs/train/.venv/bin/activate
python --version
python -m pip check
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() == 8'
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
```

모델 최초 다운로드 후 Hugging Face revision을 따로 기록하고 모든 팀원이 같은 snapshot을 사용하세요. lock은 Python 패키지 환경을 고정하지만 원격 모델·dataset의 향후 변경까지 고정하지는 않습니다. 인증 토큰은 환경/credential manager에서만 관리합니다.

## Dataset 준비

```bash
python data_tools/build_openthoughts_30k.py
python data_tools/make_openthoughts_15k_subset.py
python data_tools/build_eval_sets.py
```

첫 명령은 `Ashkchamp/Openthoughts_math_filtered_30K`를 공식 코드대로 정리합니다. 두 번째는 shuffle 전 첫 15,000행을 선택하고 trainer가 seed 42로 shuffle합니다. 산출물은 `data/raw/openthoughts_math_filtered_15k.parquet`, 평가 자료는 `data/eval/{aime24,aime25,hmmt25}.parquet`입니다. Dataset 원본 revision과 전처리 commit을 팀 결과에 기록하세요. launcher는 실제 학습 parquet SHA256도 기록합니다. 평가 문제에 정답 풀이를 넣어 student 정확도를 측정하면 안 됩니다. Paired probe의 gold context는 teacher 측 진단 전용입니다.

## 8× RTX 3090 실행과 메모리 제약

공식 학습 하드웨어는 8×H100입니다. **8×3090 24GB에서 성공했다고 검증하지 않았습니다.** 각 GPU의 24GB는 합쳐진 단일 메모리가 아닙니다. FSDP student/optimizer 이외에도 frozen teacher와 colocated vLLM, KV cache, dense 25k prompt activation, full-vocabulary JSD 텐서가 필요합니다. 1.7B라도 최대 길이에서 OOM이 가능합니다.

팀 config는 batch/LR/loss/4k cap을 유지하고 vLLM memory utilization을 **.15**로 낮춘 실행 후보입니다. 이 값, 명시적 ON, 체크포인트 저장 주기는 팀 설정입니다. 원래 max prompt 25k를 그대로 유지하므로 먼저 1k probe와 소규모 실행으로 peak VRAM을 측정하세요. OOM이면 config를 복사해 prompt/data 길이 제한 또는 generation backend를 조정하고 새로운 실험 ID로 기록합니다. `use_vllm=false`는 upstream이 지원하지만 generation backend 변경이므로 별도 ablation입니다. Prompt 길이를 낮추면 gold 풀이가 잘릴 수 있어 공식 dense 설정과 같다고 쓸 수 없습니다. 1,024 학습은 4,096 학습의 대체 근거가 아닙니다.

```bash
# 실행 명령만 확인: GPU와 torch import가 필요하지 않음
python -m research.run --config configs/research/vanilla_4096.json --run-id vanilla-s42 --dry-run
# 실제 8 GPU 학습: 출력 디렉터리가 존재하면 덮어쓰지 않고 중단
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m research.run --config configs/research/vanilla_4096.json --run-id vanilla-s42
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m research.run --config configs/research/fork_4096.json --run-id fork-s42
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m research.run --config configs/research/random_4096.json --run-id random-s42
```

추가 seed는 `--seed 43`와 새로운 `--run-id`를 사용합니다. 자동 resume은 제공하지 않습니다. 최종 모델과 중간 HF checkpoint는 `runs/<run-id>/checkpoint/` 아래에 저장됩니다.

## 1,024-token paired probe

```bash
python -m research.probe --config configs/research/probe_1024.json
python -m research.analyze --positions runs/probe-base-1024-s42/positions.jsonl --responses runs/probe-base-1024-s42/responses.jsonl --output runs/probe-base-1024-s42/metrics.json
```

이 probe는 HF sampling T=1/top-p=1/top-k=0, 첫 16문제, 문제당 1 sample로 시작하는 **팀 진단 설정**입니다. 공식 vLLM 정확도 평가와 분리합니다. CUDA 0에 student, CUDA 1에 teacher를 올립니다. 같은 student trajectory를 unprivileged(base), answer-only(sparse), full solution(dense) teacher로 채점합니다. Teacher는 원래 초기 모델로 유지하고 config의 student만 `runs/vanilla-s42/checkpoint` 등으로 바꾸면 기존 checkpoint를 분석할 수 있습니다. 각 config의 `output_dir`는 새 경로로 바꾸세요. Prompt cap 초과는 조용히 자르지 않고 실패합니다. CPU로 옮긴 full-vocabulary log-probability도 크므로 충분한 RAM이 필요합니다. 긴 sequence와 checkpoint probe는 GPU 미검증입니다.

## Metric 정의

- 논문 fork: teacher next-token full-softmax의 `pmax < .25`; lock: `pmax > .65`. 경계값은 둘 다 아닙니다. Entropy threshold로 대체하지 않습니다. 각 유효 completion token(EOS 포함, padding 제외)을 분모로 rollout별 비율을 구한 뒤 rollout macro 평균을 보고합니다.
- Fork suppression: 동일 trajectory·position에서 base 대비 sparse/dense의 fork-rate 차이를 확인합니다. 낮은 fork-rate만으로 성능 저하나 causal mechanism을 확정하지 않습니다.
- `positions.jsonl`: 0-based completion 위치, sampled token ID, student/teacher pmax와 entropy(nats), argmax disagreement, sampled `log p_teacher - log p_student`, 대칭 JSD(α=.5), fork/lock을 저장합니다. 양의 log-ratio는 teacher가 그 sampled token을 상대적으로 지지한다는 뜻이며 JSD gradient와 동일하지 않습니다. 학습 hook은 loss eligibility/kept도 기록합니다.
- `research.analyze`는 위치별 통계와 rollout별 통계, teacher 조건별 macro fork/lock rate를 출력합니다. 학습 진단은 rank별 파일과 step/call_id/batch_row로 식별하며 epoch가 진행되면 분포가 바뀝니다. paired probe와 직접 합산하지 마세요.
- Response length는 첫 EOS까지 포함한 completion token 수입니다. Reasoning length는 `</think>` 앞부분을 재토큰화한 길이입니다. 닫는 태그가 없으면 censored reasoning으로 표시합니다. 정확한 내부 사고량의 측정은 아닙니다.
- EOS rate는 실제 EOS token 출현 비율, cap-hit은 EOS 없이 token cap에 도달한 비율입니다. finish_reason도 별도로 저장합니다. stop-string 종료와 EOS는 같지 않습니다.
- `wait/hmm/maybe/however/verify/check/reconsider/backtrack`는 대소문자를 무시한 단어 경계 기반 self-correction **proxy**입니다. raw count와 response 1k tokens당 count를 함께 저장합니다. 논문 어휘 목록 전체의 재현이 아닙니다.
- avg@k는 n≥k인 각 문제의 n개 표본 정답률을 평균한 추정량입니다. n=k이면 논문 avg@16 정의와 같습니다. pass@k는 문제별 `1-C(n-c,k)/C(n,k)`의 unbiased estimator입니다. n<k 문제는 제외하고 eligible 수를 공개합니다. 표본 수를 맞춰 비교하세요. 정답 채점은 upstream grader가 담당합니다.

Fork Mask는 **student** pmax<.25인 loss-eligible 위치를 제외합니다. Privileged teacher의 fork label과 다른 팀 가설입니다. Random Mask는 각 rollout의 동일 eligible 위치 집합에서 같은 수를 seed로 추출합니다. 둘 다 attention과 rollout을 건드리지 않고, upstream의 처음 3 token 제외 이후 loss mask만 바꿉니다. 남은 토큰 수로 rollout loss를 다시 정규화하며 전부 제외하면 그 rollout loss는 0입니다. 학습 stochastic trajectory가 달라지면 실제 mask 수도 달라질 수 있으므로 같은 trajectory상의 control과 독립 학습 간 control을 구분하세요.

## 정확도 evaluation 실제 명령

아래는 single-GPU 짧은 진단 평가입니다. 모델 경로만 바꿔 Base/Vanilla/Fork/Random을 같은 문제·seed·decoding으로 비교합니다. 매번 새 output_dir를 사용하세요. 3090에서는 batch 1부터 시작합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python run_generate.py --model_name models/Qwen3-1.7B --dataset aime24 --output_dir runs/base-eval-aime24-1024-s42 --batch_size 1 --max_tokens 1024 --num_gpus 1 --temperature 0.6 --top_p 0.95 --num_generation 16 --enable_thinking true --seed 42 --research_metrics
python run_eval.py --file_path runs/base-eval-aime24-1024-s42/aime24-temp_0.6-top_p_0.95.jsonl
python -m research.analyze --graded runs/base-eval-aime24-1024-s42/aime24-temp_0.6-top_p_0.95.jsonl --responses runs/base-eval-aime24-1024-s42/aime24-temp_0.6-top_p_0.95.jsonl --output runs/base-eval-aime24-1024-s42/research_metrics.json
```

`--dataset`는 `aime25` / `hmmt25`로 변경할 수 있습니다. 4,096 검증은 `--max_tokens 4096`와 새 output_dir, 공식 장문 비교는 38,912와 새 output_dir를 사용합니다. 38,912 설정이 3090에서 실행 가능하다는 보장은 없습니다. 긴 budget은 tensor parallel, RAM/VRAM 등을 별도 검증하고 기록하세요. upstream SLURM H100 설정도 보존되어 있습니다.

기존 4k graded JSONL은 `research.analyze --graded` 입력으로 그대로 사용할 수 있습니다. token IDs/termination metadata가 없던 과거 파일은 EOS/cap-hit을 정확히 복원할 수 없습니다. 긴 rollout의 prefix를 잘라 채점하는 `run_budget_sweep.py`는 cap별 독립 generation과 구별해야 합니다.

## 팀원 4명의 확장 규칙

공유 main은 baseline으로 유지하고 개인 branch에서 `research/` 모듈과 `configs/research/<owner>_<method>.json`을 추가합니다. 예: 1명은 paired fork 진단, 1명은 loss mitigation, 1명은 decoding/길이 분석, 1명은 evaluation/재현성 검증을 맡을 수 있습니다. 새 methodology는 config의 명시적 opt-in과 CPU test를 함께 추가하고 Vanilla loss/gradient 호환성 test를 통과해야 합니다. 비교마다 Base/Vanilla/Fork/동일 비율 Random을 포함하고 기존 결과를 덮어쓰지 않습니다. GitHub collaborator 초대는 팀원 계정을 받은 뒤 별도로 진행합니다.

`runs/<id>/manifest.json`에는 resolved config, 실제 command, seed, Git commit/dirty 상태, dataset SHA256, pip freeze가 저장됩니다. `hardware.txt`, `status.json`, `checkpoint/`, 선택적 `checkpoint/diagnostics/rank*.jsonl`도 관리합니다. Probe는 config+dataset hash와 response/position JSONL을 저장합니다. Probe/eval에도 모델 HF revision, Git commit, GPU/driver, 입력 데이터 hash를 팀 결과 기록에 보충하세요. 시스템 전체 환경변수나 인증 토큰을 manifest에 저장하지 않습니다. 결과 공유는 별도 팀 저장소/스토리지에서 하고 공개 GitHub에는 작은 익명 summary만 검토 후 올립니다.

## 테스트·현재 상태·TODO

```bash
python -m unittest discover -s tests -v
python -m research.run --config configs/research/vanilla_1024.json --run-id diagnostic-plan --dry-run
```

구현: paper-threshold fork/lock, 위치별 supervision 통계, paired teacher probe, loss-only masking/control, avg/pass@k 집계, 길이/termination/proxy, config launcher/manifest, CPU CI.

검증 기록은 [docs/VALIDATION.md](docs/VALIDATION.md), 설정 근거는 [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md)를 봅니다. TODO: Linux lock 설치, 실제 Qwen tokenizer/generation 통합, 8×3090 peak VRAM과 속도 측정, 실제 1k paired probe, 기존 4k 결과의 provenance 확인, 장문 평가, seed 반복·신뢰구간, 논문 lexical taxonomy 전체 재현, 자동 resume 및 모델/dataset revision 강제 고정. 실제 학습 성능과 mitigation 효과는 아직 검증하지 않았습니다.
