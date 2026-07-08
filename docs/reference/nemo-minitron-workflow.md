# NeMo Minitron 공식 워크플로 (논문 방법) — 명령·데이터 형식 + 한국어 매핑

출처: NeMo 25.04 컨테이너 `/opt/NeMo/tutorials/llm/llama/pruning-distillation/` (01~04 노트북 + README).
논문: [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796).
공식 문서: [pruning](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/pruning/pruning.html) · [distillation](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/distillation/distillation.html) · [블로그](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/).

## 전체 순서 (논문 방법)
**01 데이터준비 → 02 teacher 파인튜닝(=한국어 주입) → 03 pruning(depth/width) → 04 distillation → HF export**

---

## 📦 데이터 형식 (핵심)
1. 각 split을 **jsonl, 한 줄당 `{"text": "..."}`** 로 저장 (train/validation/test).
2. **Megatron 메모리맵으로 전처리** (이게 prune·teacher FT·distill 모두가 먹는 형식):
```bash
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
  --input="{DATA_PATH}/ko-train.jsonl" \
  --tokenizer-library=huggingface \
  --tokenizer-type="<HF_MODEL_ID>" \
  --output-prefix="{DATA_PATH}/ko_tokenized_train" \
  --append-eod --workers=32
# → ko_tokenized_train_text_document.{idx,bin}
```
→ **한국어는 그냥 `{"text": 한국어문서}` jsonl 만들어 위로 전처리하면 됨** (WikiText 자리에 한국어 코퍼스).

## 모델 변환
```python
# HF → NeMo2 (우리 Nano-8B는 이미 함)
llm.import_ckpt(llm.LlamaModel(llm.Llama31Config8B()), source="hf://<MODEL>", output_path="<NEMO_PATH>")
# NeMo2 → HF (distill 후 평가/서빙용)
llm.export_ckpt(path="<NEMO_MODEL>", target="hf", output_path="<HF_OUT>")
```

---

## 02. Teacher 파인튜닝 (= 한국어 주입 단계)
> 논문: distillation 전에 teacher를 **타깃 데이터로 보정 FT** 안 하면 distillation 가이드가 suboptimal. **여기가 한국어를 넣는 자리.**
```python
import nemo_run as run
from nemo.collections import llm
recipe = llm.llama31_8b.finetune_recipe(num_nodes=1, num_gpus_per_node=DEVICES, peft_scheme=None, seq_length=8192)  # peft_scheme=None = full FT
recipe.resume.restore_config.path = MODEL_PATH      # NeMo2 체크포인트
recipe.data = run.Config(llm.PreTrainingDataModule, paths=DATA_PATHS, index_mapping_dir=..., seq_length=8192,
                         micro_batch_size=MICRO_BATCH_SIZE, global_batch_size=GLOBAL_BATCH_SIZE)
# DATA_PATHS = {"train":[1.0, ".../ko_tokenized_train_text_document"], "validation":[...], "test":[...]}
# STEPS, LR=1e-4, MIN_LR=1e-5, WARMUP=2 ...
executor = run.LocalExecutor(ntasks_per_node=recipe.trainer.devices, launcher="torchrun", env_vars={...})
run.run(recipe, executor=executor, name="ko-teacher-ft")
# → checkpoints/best (가장 낮은 val_loss 체크포인트를 best로 rename)
```

## 03. Pruning (depth 또는 width)
```python
from nemo.collections.llm.modelopt import PruningConfig
from nemo.collections.llm.modelopt.recipes import prune_recipe
# 공통: TENSOR_PARALLEL_SIZE = 1 (prune은 TP=1만 지원), PP=DEVICES, NUM_TRAIN_SAMPLES=1024(보정)
recipe = prune_recipe(nemo_checkpoint=TEACHER_FT_BEST, save_path=SAVE)
recipe.data = run.Config(llm.PreTrainingDataModule, paths=DATA_PATHS, ...)   # 한국어 보정 데이터
recipe.tp_size=1; recipe.pp_size=DEVICES; recipe.num_train_samples=1024; recipe.legacy_ckpt=True

# (a) depth pruning — 논문 권장: 연속 레이어 16~31 제거(두번째 끝블록)
PruningConfig(drop_layers=[16,17,...,31])           # 또는 target_num_layers=16 (코사인유사도 자동)
# (b) width pruning — ffn 9216, hidden 3072 (heads/query_groups도 선택)
PruningConfig(target_ffn_hidden_size=9216, target_hidden_size=3072,
              target_num_attention_heads=None, target_num_query_groups=None)
run.run(recipe, executor=run.LocalExecutor(..., launcher="torchrun"), name="ko_pruning")
```

## 04. Distillation (pruned student ← teacher)
```python
from nemo.collections.llm.modelopt.recipes import distillation_recipe
recipe = distillation_recipe(student_model_path=PRUNED, teacher_model_path=TEACHER_FT_BEST,
                             name=EXP, num_nodes=1, num_gpus_per_node=DEVICES)
recipe.resume.restore_config = run.Config(RestoreConfig, path=PRUNED)
recipe.data = run.Config(llm.PreTrainingDataModule, paths=DATA_PATHS, ...)   # 한국어
recipe.trainer.max_steps=STEPS; recipe.optim.config.lr=1e-4 ...
run.run(recipe, executor=run.LocalExecutor(..., launcher="torchrun"), name=EXP)
# → 최종 distilled 모델 (checkpoints/...). export_ckpt 로 HF 변환해 KMMLU/서빙.
```

---

## ⚠️ 리소스 현실 (중요)
- 튜토리얼 기본값 = **8×80GB GPU(H100/A100)**. TP/PP/micro-batch 줄이면 적은 자원에서도 가능하다고 명시.
- 우리 **단일 Blackwell 96GB** 에서:
  - **Pruning: OK** (TP=1, PP=1 — mock으로 검증됨, importance forward만).
  - **Teacher FT(8B full-FT): 빡빡/OOM 위험** (8B full Adam ≈ 128GB 필요 → 단일 96GB 초과 가능). LoRA(peft_scheme) 또는 메모리 최적화/시퀀스↓ 필요할 수 있음.
  - **Distillation(8B teacher freeze + 4B student train): 빡빡** — gradient checkpointing/distributed optimizer로 들어갈 수도. micro-batch·seq 조절 필요.
- NAS(Puzzle)·일부 기능은 이 노트북에서 미지원(향후).

## 한국어 적용 계획 (요약)
1. 한국어 코퍼스 → `{"text":...}` jsonl → `preprocess_data_for_megatron.py` (Nano-8B 토크나이저)
2. teacher(Nano-8B) **한국어 full-FT**(또는 메모리상 LoRA) → best 체크포인트
3. depth(16~31) 또는 width(ffn9216/hid3072) prune → 4B student
4. 한국어 데이터로 distillation
5. `export_ckpt` → HF → KMMLU/채팅 평가

**모델 주의:** Nano-8B(Llama arch)는 위 흐름 OK. 단 base 한국어가 Phi-4보다 −10pt 약하므로, teacher 한국어 FT를 충분히 하거나 base 재고려 필요. (Nemotron-3-Nano-4B 하이브리드는 Blackwell 커널 비호환 → 제외)
