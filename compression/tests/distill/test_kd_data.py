import json
import torch
from safetensors.torch import save_file
from src.distill.kd_data import TopKKDDataset, KDCollator


def _make_example(tmp, name, L, P, k):
    input_ids = torch.arange(L, dtype=torch.int32)
    pos = torch.arange(L - P, L, dtype=torch.int32)         # 마지막 P개가 assistant
    topk_idx = torch.zeros(P, k, dtype=torch.int32)
    topk_logit = torch.ones(P, k, dtype=torch.float16)
    f = tmp / f"{name}.safetensors"
    save_file(
        {"input_ids": input_ids, "pos": pos, "topk_idx": topk_idx, "topk_logit": topk_logit},
        str(f),
    )
    return {"id": name, "file": str(f), "n_pos": P}


def test_dataset_and_collator_shapes(tmp_path):
    rows = [_make_example(tmp_path, "a", L=6, P=2, k=3),
            _make_example(tmp_path, "b", L=8, P=4, k=3)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    ds = TopKKDDataset(str(manifest))
    assert len(ds) == 2
    batch = KDCollator(pad_id=0)([ds[0], ds[1]])

    assert batch["input_ids"].shape == (2, 8)               # max L
    assert batch["pos"].shape == (2, 4)                     # max P
    assert batch["topk_idx"].shape == (2, 4, 3)
    assert batch["attention_mask"].sum().item() == 6 + 8    # 실토큰 수
    # 예제 a는 P=2라 뒤 2자리 padding
    assert batch["pos_mask"][0].tolist() == [1, 1, 0, 0]
    # hard_labels = input_ids[pos]; padding 위치는 -100
    assert batch["hard_labels"][0, 2].item() == -100
    assert batch["hard_labels"][0, 0].item() == ds[0]["input_ids"][ds[0]["pos"][0]]
