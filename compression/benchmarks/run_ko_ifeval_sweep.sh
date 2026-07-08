#!/usr/bin/env bash
# Ko-IFEval/IFEval류 생성형 지시 준수 평가.
# llama.cpp 서버(OpenAI-compatible /v1/completions)를 모델별로 띄운 뒤 lm-eval GGUF 모델로 측정한다.
#
# 기본 TASK는 ifeval_ko이다. 현재 lm-eval 기본 설치에 한국어 태스크가 없으면
# TASK=ifeval 로 영문 IFEval smoke/대체 평가를 돌리거나, custom task를 등록한 뒤 TASK를 맞춘다.
set -euo pipefail

ROOT=${ROOT:-$HOME/LLM-VLM-in-Jetson}
cd "$ROOT/compression"

PY=${PY:-.venv/bin/python}
LLAMA_SERVER=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
GD=${GD:-$HOME/.cache/huggingface/hub/models--unsloth--Phi-4-mini-instruct-GGUF/snapshots/78eb92a46fc37e6b524df991ed9aca9bc6aa7b80}
TASK=${TASK:-ifeval_ko}
LIMIT=${LIMIT:-}
PORT=${PORT:-18080}
HOST=${HOST:-127.0.0.1}
OUT=${OUT:-docs/results/ko_ifeval.jsonl}
SERVER_LOG=${SERVER_LOG:-/tmp/ko_ifeval_llama_server.log}

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

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

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_server() {
  for _ in $(seq 1 120); do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "server did not become healthy; tail follows:" >&2
  tail -80 "$SERVER_LOG" >&2 || true
  return 1
}

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

for item in "${Q[@]}"; do
  cleanup
  name="${item%%:*}"
  file="${item#*:}"
  if [[ -n "${QUANT_FILTER:-}" ]]; then
    case ",$QUANT_FILTER," in
      *",$name,"*) ;;
      *) continue ;;
    esac
  fi
  model_path="$GD/$file"
  echo "===== Ko-IFEval [$name] $file ====="
  run_dir="/tmp/ko_ifeval_lm_eval_${name}_$(date +%s)"

  "$LLAMA_SERVER" -m "$model_path" --host "$HOST" --port "$PORT" -ngl 99 -c 4096 > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_server

  "$PY" -m lm_eval \
    --model gguf \
    --model_args "base_url=http://$HOST:$PORT" \
    --tasks "$TASK" \
    "${limit_args[@]}" \
    --batch_size 1 \
    --output_path "$run_dir" \
    --log_samples

  "$PY" benchmarks/summarize_lm_eval_json.py \
    --name "phi4mini-$name" \
    --task "$TASK" \
    --results-dir "$run_dir" \
    --out "$OUT"
done

echo "KO_IFEVAL_DONE $OUT"
