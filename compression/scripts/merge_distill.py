import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "artifacts/phi4-pruned-depth"
ADAPTER = "artifacts/phi4-pruned-depth-distill-lora-pilot2k"
OUT = "artifacts/phi4-pruned-depth-distill-merged-pilot2k"

print("[merge] loading base:", BASE, flush=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
print("[merge] applying adapter:", ADAPTER, flush=True)
m = PeftModel.from_pretrained(base, ADAPTER)
print("[merge] merge_and_unload ...", flush=True)
m = m.merge_and_unload()
print("[merge] saving ->", OUT, flush=True)
m.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
tok.save_pretrained(OUT)
print("[merge] DONE", flush=True)
