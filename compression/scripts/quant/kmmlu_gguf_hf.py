# Codex 처방: transformers gguf 버전탐지 버그 우회
import importlib.metadata, gguf
if not hasattr(gguf, "__version__"):
    gguf.__version__ = importlib.metadata.version("gguf")
from transformers.utils import import_utils
try:
    import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
except Exception:
    pass
import sys, os
sys.path.insert(0, os.getcwd())
from lm_eval import simple_evaluate
bundle, gguf_file, name = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 50
margs = (f"pretrained={bundle},gguf_file={gguf_file},"
         f"tokenizer=artifacts/phi4-sft-v2,dtype=float16,trust_remote_code=True")
r = simple_evaluate(model="hf", model_args=margs, tasks=["kmmlu"], limit=limit, device="cuda", batch_size=4)
print(f"KMMLU_RESULT {name} {r['results']['kmmlu']['acc,none']:.4f}", flush=True)
