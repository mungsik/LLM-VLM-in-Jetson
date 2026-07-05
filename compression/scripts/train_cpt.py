"""pruned Phi-4에 한국어 CPT(full-FT, HF Trainer + paged_adamw_8bit).

Plan 1 CPT 데이터(input_ids/labels, 4096 패킹) 소비. 프루닝 복구 겸 한국어 주입.
Unsloth 대신 기존 train_sft_multiturn.py와 동일한 검증된 패턴(HF Trainer + bnb 8bit
paged optim + gradient checkpointing) 사용 — bleeding-edge Blackwell에서 env 안전.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt_train.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    tok = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = load_from_disk(cfg["data_path"])   # input_ids/labels, 고정 4096 → 패딩 불필요

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg.get("max_steps", -1),
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=1,
        bf16=True,
        optim="paged_adamw_8bit",          # bnb 8bit paged: full-FT 옵티마이저 메모리 절감
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=default_data_collator
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"[cpt] saved -> {cfg['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
