# --- Codex 처방: transformers 5.12.1 gguf 버전탐지 버그 우회 (프로세스 한정) ---
import importlib.metadata, gguf
if not hasattr(gguf, "__version__"):
    gguf.__version__ = importlib.metadata.version("gguf")
from transformers.utils import import_utils
try:
    import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
except Exception:
    pass
# ----------------------------------------------------------------------------
import sys
from lm_eval import simple_evaluate
path, name = sys.argv[1], sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else ""
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 50
is_gguf = path.endswith(".gguf")
dtype = "half" if is_gguf else "bfloat16"   # Blackwell GGUF는 half 필수(Codex)
margs = f"pretrained={path},gpu_memory_utilization=0.6,max_model_len=2048,dtype={dtype}"
if extra:
    margs += "," + extra
r = simple_evaluate(model="vllm", model_args=margs, tasks=["kmmlu"], limit=limit, batch_size="auto")
print(f"KMMLU_RESULT {name} {r['results']['kmmlu']['acc,none']:.4f}", flush=True)
