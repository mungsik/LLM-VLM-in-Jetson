#!/bin/bash
# Track 2 (minitron-blackwell): mock prune 완료 대기(파이프라인 검증) →
#   기본 모델 한국어(KMMLU) 측정: Nano-8B + Nemotron-3-Nano-4B → 인스턴스 종료
# systemd-run(root)로 실행. 모든 단계 best-effort, 무슨 일이 있어도 마지막에 shutdown.
WS=/opt/minitron_ws
R=$WS/RESULTS.md
NEMO=nvcr.io/nvidia/nemo:25.04
HFTOK=$(grep -oE 'hf_[A-Za-z0-9]+' /tmp/hf_env.sh 2>/dev/null | head -1)
mkdir -p $WS
{
echo "# Minitron Track2 결과 (start $(date))"
echo ""
echo "용도: NVIDIA Minitron 파이프라인 검증 + 기본 모델 한국어(KMMLU) 베이스라인."
echo "주의: vanilla prune은 현재 MOCK calibration(파이프라인 검증용)이라 pruned 품질은 무의미."
echo "      실 calibration+distillation 기반 pruned 한국어 비교는 다음 supervised 세션."
echo ""
} > $R

DRUN="docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e HF_HOME=/workspace/hf_cache -e HF_TOKEN=$HFTOK -v $WS:/workspace $NEMO"

# [0] 진행 중인 mock prune 컨테이너 종료까지 대기 (파이프라인 검증 완결)
echo "## [0] in-progress prune(mock) 대기 — 파이프라인 검증 $(date)" >> $R
while docker ps --format '{{.Image}}' 2>/dev/null | grep -q nemo; do sleep 120; done
echo "prune 컨테이너 종료 $(date). 산출물:" >> $R
ls -la $WS/nano8b-width-4b >> $R 2>&1 || echo "(nano8b-width-4b 없음 — prune 로그 확인)" >> $R
echo "--- prune.log tail ---" >> $R
tail -8 $WS/prune.log 2>&1 | sed 's/\x1b\[[0-9;]*m//g' >> $R

# [1] Nano-8B (Llama-3.1-Nemotron-Nano-8B) 기본 KMMLU
echo "" >> $R; echo "## [1] Nano-8B base KMMLU (한국어) $(date)" >> $R
$DRUN bash -c "pip install -q lm-eval 2>/dev/null; HF_TOKEN=$HFTOK lm_eval --model hf --model_args pretrained=nvidia/Llama-3.1-Nemotron-Nano-8B-v1,trust_remote_code=True,dtype=bfloat16 --tasks kmmlu --limit 30 --device cuda:0 --batch_size 8 2>&1 | tail -40" >> $R 2>&1

# [2] Nemotron-3-Nano-4B (하이브리드 Mamba-Transformer) 기본 KMMLU
echo "" >> $R; echo "## [2] Nemotron-3-Nano-4B base KMMLU (한국어) $(date)" >> $R
$DRUN bash -c "pip install -q lm-eval 2>/dev/null; HF_TOKEN=$HFTOK lm_eval --model hf --model_args pretrained=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16,trust_remote_code=True,dtype=bfloat16 --tasks kmmlu --limit 30 --device cuda:0 --batch_size 8 2>&1 | tail -40" >> $R 2>&1

echo "" >> $R; echo "## DONE $(date) — 인스턴스 종료" >> $R
sync
/sbin/shutdown -h now
