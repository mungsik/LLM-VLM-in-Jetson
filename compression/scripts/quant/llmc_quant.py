import os, sys, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

METHOD = sys.argv[1]           # gptq | awq
MODEL = "artifacts/phi4-sft-v2"
OUT = f"artifacts/llmc/phi4-sft-v2-{METHOD}-w4a16"
CALIB = "artifacts/sft_data.jsonl"
NS, ML = 512, 1024

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             trust_remote_code=True, low_cpu_mem_usage=True)
print(f"[llmc] method={METHOD} layers={model.config.num_hidden_layers}", flush=True)

ds = load_dataset("json", data_files=CALIB, split="train").shuffle(seed=42)
ds = ds.select(range(min(len(ds), NS)))
def render(s): return {"text": tok.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=False)}
ds = ds.map(render)
def tokz(s): return tok(s["text"], truncation=True, max_length=ML, add_special_tokens=False)
ds = ds.map(tokz, remove_columns=ds.column_names)

if METHOD == "gptq":
    from llmcompressor.modifiers.quantization import GPTQModifier
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
elif METHOD == "awq":
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = [AWQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]
else:
    raise SystemExit("METHOD must be gptq|awq")

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=ML, num_calibration_samples=NS)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print(f"LLMC_{METHOD.upper()}_DONE", flush=True)
