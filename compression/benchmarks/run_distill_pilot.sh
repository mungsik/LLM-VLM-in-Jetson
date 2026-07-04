#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$HOME/LLM-VLM-in-Jetson}
cd "$ROOT/compression"

PY=${PY:-.venv/bin/python}
SCALE=${SCALE:-smoke}
TEACHER=${TEACHER:-microsoft/phi-4}
STUDENT=${STUDENT:-artifacts/phi4-pruned-depth}
LIMIT=${LIMIT:-200}

PROMPTS=${PROMPTS:-data/distill/${SCALE}_prompts.jsonl}
TEACHER_OUT=${TEACHER_OUT:-data/distill/${SCALE}_teacher.jsonl}
OUT=${OUT:-artifacts/phi4-pruned-depth-distill-lora-${SCALE}}

"$PY" scripts/prepare_distill_prompts.py --scale "$SCALE" --out "$PROMPTS"
"$PY" scripts/generate_teacher_responses.py \
  --teacher "$TEACHER" \
  --prompts "$PROMPTS" \
  --out "$TEACHER_OUT" \
  --limit "$LIMIT"
"$PY" scripts/train_lora_sft.py \
  --student "$STUDENT" \
  --train-file "$TEACHER_OUT" \
  --out "$OUT"

echo "DISTILL_DONE $OUT"
