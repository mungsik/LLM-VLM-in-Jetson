#!/bin/bash
# Track 1 (phi4-blackwell): SFT 본런 완료 대기 → chat-v1 KMMLU → 인스턴스 종료
# systemd-run(root)로 실행되므로 shutdown 직접 가능, venv는 sudo -u mungsik
LOG=/opt/track1_RESULTS.log
echo "=== track1 orchestrator start $(date) ===" > $LOG

# 1) SFT 학습 프로세스가 끝날 때까지 대기
while pgrep -f "train_sft_multiturn.py" >/dev/null 2>&1; do sleep 120; done
echo "=== SFT process ended $(date) ===" >> $LOG
sudo -u mungsik bash -lc 'ls -la /home/mungsik/LLM-VLM-in-Jetson/compression/artifacts/phi4-pruned-depth-chat-v1' >> $LOG 2>&1
echo "--- sft_main.log tail ---" >> $LOG
tail -15 /tmp/sft_main.log >> $LOG 2>&1

# 2) chat-v1 KMMLU 가드 (best effort, 90분 타임아웃)
echo "=== KMMLU eval start $(date) ===" >> $LOG
timeout 5400 sudo -u mungsik bash -lc 'cd /home/mungsik/LLM-VLM-in-Jetson/compression && source /tmp/hf_env.sh 2>/dev/null; .venv/bin/python -c "from src.common.eval_kmmlu import run_kmmlu; print(\"chatv1_KMMLU_limit500 =\", run_kmmlu(\"artifacts/phi4-pruned-depth-chat-v1\", limit=500))"' >> $LOG 2>&1
echo "=== KMMLU done $(date) ===" >> $LOG

# 3) 인스턴스 종료 (GCE: shutdown → TERMINATED, 컴퓨트 과금 중단)
echo "=== shutting down $(date) ===" >> $LOG
sync
/sbin/shutdown -h now
