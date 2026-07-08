#!/usr/bin/env bash
# 8종 quant 한국어 perplexity 측정 (인스턴스 실행). 결과 ~/ppl.jsonl 누적.
set -u
cd ~/LLM-VLM-in-Jetson/compression
PY=.venv/bin/python
REPO=unsloth/Phi-4-mini-instruct-GGUF
TOK=unsloth/Phi-4-mini-instruct
rm -f ~/ppl.jsonl
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
  echo "===== PPL [$name] ====="
  $PY /tmp/eval_ppl.py --repo "$REPO" --gguf-file "$file" --name "phi4mini-${name}" \
      --tokenizer "$TOK" --out ~/ppl.jsonl \
      2>&1 | grep -avE "Loading|it/s|Converting|Downloading" | tail -3
done
echo "PPL_ALL_DONE"
