"""Full-FT student with offline top-k logit-level KD."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.kd_data import TopKKDDataset, KDCollator
from src.distill.kd_loss import kd_topk_loss

_KD_KEYS = ("pos", "pos_mask", "topk_idx", "topk_logit", "hard_labels")


class KDTrainer(Trainer):
    def __init__(self, *a, temperature=2.0, alpha=0.9, **kw):
        super().__init__(*a, **kw)
        self._T, self._alpha = temperature, alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        kd = {key: inputs.pop(key) for key in _KD_KEYS}
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        res = kd_topk_loss(
            out.logits, kd["pos"], kd["pos_mask"], kd["topk_idx"],
            kd["topk_logit"], kd["hard_labels"],
            temperature=self._T, alpha=self._alpha,
        )
        return (res["loss"], out) if return_outputs else res["loss"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="artifacts/phi4-pruned-depth-masked")
    ap.add_argument("--manifest", default="data/distill/teacher_kd/manifest.jsonl")
    ap.add_argument("--out", default="artifacts/phi4-pruned-depth-distill-kd-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--max-steps", type=int, default=-1)   # smoke용
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.student, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = TopKKDDataset(args.manifest)
    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        save_only_model=True,   # 8bit paged optimizer state는 torch.save가 깨짐 → 모델만 저장
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = KDTrainer(
        model=model, args=train_args, train_dataset=ds,
        data_collator=KDCollator(pad_id=tok.pad_token_id),
        temperature=args.temperature, alpha=args.alpha,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # distill 서빙은 chat-template 전용 → 명시 동봉
    src_tpl = Path(args.student) / "chat_template.jinja"
    if src_tpl.exists():
        (Path(args.out) / "chat_template.jinja").write_text(
            src_tpl.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
