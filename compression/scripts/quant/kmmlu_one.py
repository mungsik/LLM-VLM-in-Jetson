import sys, os
sys.path.insert(0, os.getcwd())
from src.common.eval_kmmlu import run_kmmlu
path, name = sys.argv[1], sys.argv[2]
acc = run_kmmlu(path, limit=50)
print(f"KMMLU_RESULT {name} {acc:.4f}", flush=True)
