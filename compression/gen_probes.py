"""프로브셋(단일15+멀티턴10)에 대해 모델 출력을 vLLM 오프라인으로 생성.
사용: python gen_probes.py <model_path> <out.jsonl>
"""
import json, sys
from pathlib import Path
from vllm import LLM, SamplingParams

model_path, out_path = sys.argv[1], sys.argv[2]
PD = "/home/mungsik/LLM-VLM-in-Jetson/docs/results/probes"
tpl = (Path(model_path) / "chat_template.jinja").read_text(encoding="utf-8")

llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=4096, enforce_eager=True, trust_remote_code=True)
sp = SamplingParams(temperature=0.7, repetition_penalty=1.15, max_tokens=512)
SYS = {"role": "system", "content": "당신은 한국어로 답하는 친절한 AI 비서입니다."}

def chat(messages):
    return llm.chat(messages, sp, chat_template=tpl)[0].outputs[0].text.strip()

results = []
# 단일턴
for line in open(f"{PD}/chat_probes_ko.jsonl", encoding="utf-8"):
    p = json.loads(line)
    ans = chat([SYS, {"role": "user", "content": p["prompt"]}])
    results.append({"id": p["id"], "type": "single", "category": p.get("category", ""),
                    "prompt": p["prompt"], "answer": ans})
# 멀티턴 (대화 누적)
for line in open(f"{PD}/chat_multiturn_probes_ko.jsonl", encoding="utf-8"):
    p = json.loads(line)
    msgs = [SYS]
    outs = []
    for t in p["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        a = chat(msgs)
        msgs.append({"role": "assistant", "content": a})
        outs.append(a)
    results.append({"id": p["id"], "type": "multiturn", "turns": [t["user"] for t in p["turns"]],
                    "answers": outs, "checks": p["checks"], "final": outs[-1]})

with open(out_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"SAVED {out_path} n={len(results)}")
