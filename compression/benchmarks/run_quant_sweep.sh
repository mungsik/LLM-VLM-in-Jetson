#!/usr/bin/env bash
# Phi-4-mini-instruct-GGUF 전체 quant를 KMMLU(limit=20)로 순차 평가.
# 각 quant 결과를 ~/sweep/<name>.jsonl 로 저장. (인스턴스에서 실행)
set -u
cd ~/LLM-VLM-in-Jetson/compression
PY=.venv/bin/python
REPO=unsloth/Phi-4-mini-instruct-GGUF
mkdir -p ~/sweep

# name:filename (BF16/Q8_0은 점 구분, 나머지는 하이픈)
declare -a Q=(
  "BF16:Phi-4-mini-instruct.BF16.gguf"
  "Q8_0:Phi-4-mini-instruct.Q8_0.gguf"
  "Q6_K:Phi-4-mini-instruct-Q6_K.gguf"
  "Q5_K_M:Phi-4-mini-instruct-Q5_K_M.gguf"
  "Q4_K_M:Phi-4-mini-instruct-Q4_K_M.gguf"
  "Q3_K_M:Phi-4-mini-instruct-Q3_K_M.gguf"
  "Q2_K_L:Phi-4-mini-instruct-Q2_K_L.gguf"
  "Q2_K:Phi-4-mini-instruct-Q2_K.gguf"
)

for item in "${Q[@]}"; do
  name="${item%%:*}"; file="${item#*:}"
  out=~/sweep/${name}.jsonl
  if [ -s "$out" ]; then echo "[skip] $name (이미 있음)"; continue; fi
  echo "===== [$name] $file ====="
  $PY /tmp/eval_gguf.py --repo "$REPO" --gguf-file "$file" \
      --name "phi4mini-${name}" --limit 20 --out "$out" \
      2>&1 | grep -avE "Loading|it/s|Converting|Downloading" | tail -4
done
echo "ALL_DONE"
