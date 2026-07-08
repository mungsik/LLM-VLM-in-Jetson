#!/usr/bin/env bash
# llama.cpp llama-perplexity로 8종 quant 한국어 PPL 측정 (GPU). 결과 ~/llamappl.txt.
set -u
BIN=~/llama.cpp/build/bin/llama-perplexity
CORPUS=~/ko_corpus.txt
GD=~/.cache/huggingface/hub/models--unsloth--Phi-4-mini-instruct-GGUF/snapshots/78eb92a46fc37e6b524df991ed9aca9bc6aa7b80
CTX=512
CHUNKS=200
: > ~/llamappl.txt
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
  echo "===== PPL [$name] =====" | tee -a ~/llamappl.txt
  $BIN -m "$GD/$file" -f "$CORPUS" -ngl 99 -c $CTX --chunks $CHUNKS 2>&1 \
     | grep -E "Final estimate|PPL" | tail -2 | tee -a ~/llamappl.txt
  echo "" >> ~/llamappl.txt
done
echo "LLAMA_PPL_DONE"
