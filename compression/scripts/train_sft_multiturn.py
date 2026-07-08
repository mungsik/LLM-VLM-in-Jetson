# compression/scripts/train_sft_multiturn.py
"""Full-FT or LoRA multi-turn plain-CE SFT for the Korean chatbot.

--lora: 어댑터만 학습(빠름·디스크 안전), 끝에 merge_and_unload 로 표준 full 모델 저장
        → KMMLU/서빙이 일반 모델 경로처럼 동작. 학습 중 체크포인트는 어댑터(~100MB)라 디스크풀 회피.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.sft_data import MultiturnSFTDataset, SFTCollator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="artifacts/phi4-pruned-depth-masked")
    ap.add_argument("--corpus", default="data/distill/chat_corpus.jsonl")
    ap.add_argument("--out", default="artifacts/phi4-pruned-depth-chat-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    full = MultiturnSFTDataset(args.corpus, tok, args.max_length)
    n_val = max(1, int(len(full) * args.val_frac))
    train_ds = torch.utils.data.Subset(full, range(n_val, len(full)))
    val_ds = torch.utils.data.Subset(full, range(n_val))

    model = AutoModelForCausalLM.from_pretrained(
        args.student, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        optim, grad_ckpt = "adamw_torch", False   # LoRA: 옵티마이저 작음 → bnb 불필요, 체크포인트 끔(속도)
    else:
        model.gradient_checkpointing_enable()
        optim, grad_ckpt = "paged_adamw_8bit", True

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, bf16=True, optim=optim,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500,
        save_total_limit=2, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], remove_unused_columns=False, gradient_checkpointing=grad_ckpt)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=SFTCollator(pad_id=tok.pad_token_id))
    trainer.train()

    out = Path(args.out)
    if args.lora:
        merged = trainer.model.merge_and_unload()   # 어댑터 병합 → 표준 모델
        merged.save_pretrained(out)
    else:
        trainer.save_model(out)
    tok.save_pretrained(out)
    src_tpl = Path(args.student) / "chat_template.jinja"
    if src_tpl.exists():
        (out / "chat_template.jinja").write_text(src_tpl.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
