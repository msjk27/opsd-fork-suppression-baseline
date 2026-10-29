# 공식 구현 검토

기준: https://github.com/princeton-pli/rethinking-opsd-for-thinking-models
커밋 `46ccfbddeb92c3bb8b9602a1e6644eb9fa8f7c04` (검토일 2026-10-29).
논문: https://arxiv.org/html/2607.05184v1

| 확인 대상 | 코드 / 논문 근거 | 판단 |
|---|---|---|
| Qwen3-1.7B recipe | submit_experiment.sh / slurm/train.sh / slurm/config.sh | OT15k dense, LR5e-6, batch64, 4k cap |
| 데이터 | data_tools/build_openthoughts_30k.py, make_openthoughts_15k_subset.py | 정리된 30k 중 첫 15k, train에서 seed shuffle |
| Teacher | train.py model/teacher 로딩, trainer ref_model | 동일 초기 checkpoint, recipe sync_ref_model=False |
| Teacher 문맥 | train.py prepare_distil_dataset | full solution을 example-response template에 삽입 |
| Thinking | trainer.py chat_template_kwargs 적용 | enable_thinking=true가 student/teacher 모두에 적용 |
| loss | trainer.py _compute_loss | full-vocabulary generalized JSD, rollout별 valid-token 평균 |
| token skip | train.py 기본값 결정 | forced prefix 없으면 첫 3 token loss 제외, attention 유지 |
| temperature | trainer.py _get_per_token_logps_and_entropies | logit/temperature 후 softmax; team train/probe T=1 |
| fork/lock | 논문 §4.1 | teacher pmax<.25 / >.65, strict inequalities |
| supervision | 논문 §4.2 | sampled log pT-log pS, 어휘 proxy와 probability label 구별 |
| 주요 성능 | 논문 Table1 Qwen3-1.7B | avg@16 평균 .372→.308, 38,912 evaluation |
| 환경 | envs/train/pyproject.toml, uv.lock | Python3.10, H100 검증 스택 |

공식 repo에 별도 fork/lock 측정 스크립트는 확인되지 않았습니다. 본 연구 모듈은 논문 정의를 구현한 추가 코드입니다. GPU 실행으로 공식 결과를 재현했다는 의미는 아닙니다.

공식 direct train.py 기본 fsdp_config 경로는 존재하지 않는 `configs/fsdp_config_qwen.json`을 가리킵니다. upstream SLURM과 팀 launcher는 존재하는 `configs/fsdp_config_qwen3_opsd.json`을 명시적으로 넘깁니다. direct 기본값은 upstream 호환성을 위해 바꾸지 않았습니다.

팀 변경: research CLI/config 세 필드, loss-mask opt-in hook, run_generate의 선택적 seed/metadata, CPU 테스트/CI, docs와 ignore 규칙. 원래 loss 수식과 generation/data/grading 경로는 유지합니다. Random control seed는 training microstep 및 rank에서 결정되며 global RNG를 소비하지 않습니다.

Fork Mask의 student pmax 기준은 논문 teacher fork 측정과 다릅니다. 논문이 제안·검증한 mitigation이라고 서술하면 안 됩니다. 3090 vLLM utilization .15, 1k probe와 HF decoding 역시 팀 설정이며 메모리와 성능 검증이 필요합니다.
