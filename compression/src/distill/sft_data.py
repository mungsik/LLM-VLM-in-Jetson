"""Multi-turn SFT dataset: assistant-only CE labels (plain SFT, no KD)."""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from src.distill.multiturn_labels import build_multiturn_labels


class MultiturnSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 4096):
        # init 에서 전량 토크나이즈해 RAM 적재(eager). 150k 대화 ≈ 수 GB —
        # VM 96GB RAM 에서 허용, 매 step 재토크나이즈 비용을 피해 학습 throughput 우선(codex 지적: 대안은 lazy).
        self.tok = tokenizer
        self.max_length = max_length
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                lab = build_multiturn_labels(tokenizer, msgs, max_length)
                if lab["ok"] and lab["pos"]:
                    self.rows.append((lab["input_ids"], lab["pos"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, pos = self.rows[i]
        labels = [-100] * len(ids)
        posset = set(pos)
        for j in range(len(ids)):
            if j in posset:
                labels[j] = ids[j]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


class SFTCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        B = len(features)
        L = max(f["input_ids"].size(0) for f in features)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for b, f in enumerate(features):
            n = f["input_ids"].size(0)
            input_ids[b, :n] = f["input_ids"]
            labels[b, :n] = f["labels"]
            attn[b, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
