"""Dataset + collator over precomputed teacher top-k logits."""
from __future__ import annotations

import json

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class TopKKDDataset(Dataset):
    def __init__(self, manifest_path: str):
        with open(manifest_path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = load_file(self.rows[i]["file"])
        input_ids = t["input_ids"].long()
        pos = t["pos"].long()
        return {
            "input_ids": input_ids,
            "pos": pos,
            "topk_idx": t["topk_idx"].long(),
            "topk_logit": t["topk_logit"].float(),
            "hard_labels": input_ids[pos],
        }


class KDCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        B = len(features)
        L = max(f["input_ids"].size(0) for f in features)
        P = max(f["pos"].size(0) for f in features)
        k = features[0]["topk_idx"].size(1)

        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        pos = torch.zeros((B, P), dtype=torch.long)
        pos_mask = torch.zeros((B, P), dtype=torch.float)
        topk_idx = torch.zeros((B, P, k), dtype=torch.long)
        topk_logit = torch.zeros((B, P, k), dtype=torch.float)
        hard_labels = torch.full((B, P), -100, dtype=torch.long)

        for b, f in enumerate(features):
            li, pi = f["input_ids"].size(0), f["pos"].size(0)
            input_ids[b, :li] = f["input_ids"]
            attn[b, :li] = 1
            pos[b, :pi] = f["pos"]
            pos_mask[b, :pi] = 1.0
            topk_idx[b, :pi] = f["topk_idx"]
            topk_logit[b, :pi] = f["topk_logit"]
            hard_labels[b, :pi] = f["hard_labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pos": pos,
            "pos_mask": pos_mask,
            "topk_idx": topk_idx,
            "topk_logit": topk_logit,
            "hard_labels": hard_labels,
        }
