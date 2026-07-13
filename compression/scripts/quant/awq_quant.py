import json, sys
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

mp = "artifacts/phi4-sft-v2"
qp = "artifacts/awq/phi4-sft-v2-awq"
qcfg = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

calib = []
for line in open("artifacts/sft_data.jsonl", encoding="utf-8"):
    msgs = json.loads(line)["messages"]
    txt = "\n".join(m["content"] for m in msgs if m["role"] != "system")
    if len(txt) > 200:
        calib.append(txt)
    if len(calib) >= 128:
        break
print(f"[awq] calib {len(calib)} samples", flush=True)

model = AutoAWQForCausalLM.from_pretrained(mp, device_map="cuda", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
model.quantize(tok, quant_config=qcfg, calib_data=calib)
model.save_quantized(qp)
tok.save_pretrained(qp)
print("AWQ_DONE", flush=True)
