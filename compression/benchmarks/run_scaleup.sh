#!/usr/bin/env bash
set -euo pipefail
cd /home/mungsik/LLM-VLM-in-Jetson/compression

TEACHER_OUT=data/distill/scaleup_teacher.jsonl
LORA_OUT=artifacts/phi4-pruned-depth-distill-lora-scaleup
MERGED_OUT=artifacts/phi4-pruned-depth-distill-merged-scaleup

echo "[runner] STEP1 vLLM generate FULL pool  $(date)"
HF_HUB_OFFLINE=1 .venv-vllm/bin/python scripts/gen_teacher_vllm.py \
  --prompts data/distill/pilot_prompts.jsonl \
  --out "$TEACHER_OUT"
echo "[runner] teacher rows: $(wc -l < "$TEACHER_OUT")"

echo "[runner] STEP2 LoRA SFT  $(date)"
.venv/bin/python scripts/train_lora_sft.py \
  --student artifacts/phi4-pruned-depth \
  --train-file "$TEACHER_OUT" \
  --out "$LORA_OUT"

echo "[runner] STEP3 merge adapter  $(date)"
BASE=artifacts/phi4-pruned-depth ADAPTER="$LORA_OUT" OUT="$MERGED_OUT" \
.venv/bin/python - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE=os.environ["BASE"]; AD=os.environ["ADAPTER"]; OUT=os.environ["OUT"]
b=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,trust_remote_code=True)
m=PeftModel.from_pretrained(b,AD).merge_and_unload()
m.save_pretrained(OUT,safe_serialization=True)
AutoTokenizer.from_pretrained(AD,trust_remote_code=True).save_pretrained(OUT)
print("[merge] saved", OUT, flush=True)
PY

echo "[runner] STEP4 KMMLU eval  $(date)"
.venv/bin/python scripts/eval_compare.py \
  --models distill_scaleup="$MERGED_OUT"

echo "[runner] ALL DONE  $(date)"
