# 검증 기록 — 2026-10-29

## 실제 실행

- Windows, Python 3.12.14, CPU torch 2.9.0 환경에서 단위 테스트 9개 통과.
- 논문 fork/lock strict 경계 .25/.65 확인.
- Fork/Random의 rollout별 mask 수 일치, padding 보호, deterministic seed, 입력 불변 확인.
- 모든 토큰 제외 시 loss/gradient 0 확인.
- avg/pass@k의 조합론적 정답, 표본 부족 null, 중복·잘못된 label 거부 확인.
- EOS가 마지막 토큰일 때 cap-hit과 구별, padding 이후 token 제외 확인.
- token log-ratio 방향, JSD, position JSONL 및 조건별 집계 확인.
- 실제 upstream `_compute_loss`와 수정본을 AST로 추출해 같은 tensor를 입력: α=0/.5/1에서 loss와 student gradient가 bitwise 동일. Fork가 없는 fixture의 세 방법도 동일. TRL/FSDP 전체 trainer integration을 실행한 것은 아님.
- 실제 train.py/run_generate.py parser 함수를 독립 실행하여 새 CLI 인자 수용 확인.
- 모든 팀 학습 config의 flag 이름, batch64, ON/ON, frozen teacher, FSDP 경로 확인.
- launcher dry-run 및 probe/analyze CLI help 정상.
- 수정/추가 Python 소스 AST parsing 및 `git diff --check` 통과.
- **download-free tiny Qwen3 integration smoke 통과**: transformers 4.57.1 Qwen3ForCausalLM의 무작위 1-layer/hidden32 모델과 synthetic tokenizer/parquet를 생성. CPU BF16 생성 → 두 sample의 같은 trajectory를 세 teacher context로 scoring → position/response JSONL → 6 rollout 집계 성공. `python -m tests.smoke_probe`로 재실행 가능. 실제 Qwen3-1.7B weights/tokenizer, 수학 성능 검증과는 무관.
- 실제 GitHub fork의 `fork=true` 및 upstream 이력 확인. Git push 인증 연결 확인.

## 실행하지 않은 항목

- Linux Python3.10 uv frozen 환경 설치, CUDA12.8 runtime/driver 호환성.
- 8×3090 distributed FSDP + frozen teacher + colocated vLLM 학습 및 메모리 측정.
- 실제 Qwen3-1.7B tokenizer/weights, OpenThoughts 다운로드와 15k 전처리.
- 실제 1,024-token paired probe, 기존 4,096-token 결과 분석(입력 미제공).
- AIME/HMMT 실제 vLLM generation/grading, 38,912-token 논문 수치 재현.
- Masking의 성능 효과, 장문 self-correction 변화, 반복 seed 신뢰구간.
- GitHub Actions의 원격 CI 성공 여부는 push 이후 workflow 실행 상태로 확인해야 함.

테스트용 모델/데이터/라이브러리는 `.test-deps/`와 임시 디렉터리에만 존재하며 Git에서 제외됩니다. 학습용 requirements.txt와 uv.lock은 원본을 유지했습니다. 테스트 CPU dependencies는 requirements-research-test.txt로 분리했습니다.
