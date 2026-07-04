"""lm-eval output_path JSON을 docs/results/*.jsonl 요약 형식으로 변환."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_result(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        [p for p in path.rglob("*.json") if "samples" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no lm-eval result JSON under {path}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result_path = find_result(Path(args.results_dir))
    with result_path.open(encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    if args.task not in results:
        raise KeyError(f"task {args.task!r} not found in {result_path}: {list(results)}")

    metrics = {k: v for k, v in results[args.task].items() if not k.startswith("alias")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "name": args.name,
                    "task": args.task,
                    "metrics": metrics,
                    "source": str(result_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"{args.name} / {args.task}: {metrics}")


if __name__ == "__main__":
    main()
