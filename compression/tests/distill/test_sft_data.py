import json
import torch
from src.distill.sft_data import MultiturnSFTDataset, SFTCollator


class FakeTok:
    pad_token_id = 0
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]
        for m in messages:
            head = {"user": 10, "assistant": 20, "system": 30}[m["role"]]
            ids += [head] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [20]
        return ids


def _write(tmp, convos):
    p = tmp / "c.jsonl"
    p.write_text("\n".join(json.dumps({"messages": c}, ensure_ascii=False) for c in convos), encoding="utf-8")
    return str(p)


def test_labels_mask_user_supervise_assistant(tmp_path):
    tok = FakeTok()
    convos = [[
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "xy"},
    ]]
    ds = MultiturnSFTDataset(_write(tmp_path, convos), tok, max_length=4096)
    item = ds[0]
    ids, labels = item["input_ids"], item["labels"]
    assert ids.tolist() == tok.apply_chat_template(convos[0])
    from src.distill.multiturn_labels import build_multiturn_labels
    pos = build_multiturn_labels(tok, convos[0])["pos"]
    sup = [labels[j].item() for j in range(len(labels)) if labels[j].item() != -100]
    assert sup == [ids[j].item() for j in pos]
    assert all(labels[j].item() == -100 for j in range(len(labels)) if j not in set(pos))


def test_labels_concrete_non_circular(tmp_path):
    # build_multiturn_labels 에 의존하지 않는 직접 검증:
    # [user 'ab', assistant 'xy'] → 마지막 2토큰(x,y)만 supervise, 나머지 -100.
    tok = FakeTok()
    convos = [[{"role": "user", "content": "ab"}, {"role": "assistant", "content": "xy"}]]
    ds = MultiturnSFTDataset(_write(tmp_path, convos), tok)
    ids = ds[0]["input_ids"].tolist()
    labels = ds[0]["labels"].tolist()
    assert labels[:-2] == [-100] * (len(labels) - 2)
    assert labels[-2:] == ids[-2:]


def test_collator_pads(tmp_path):
    tok = FakeTok()
    convos = [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bb"}],
        [{"role": "user", "content": "ccc"}, {"role": "assistant", "content": "d"}],
    ]
    ds = MultiturnSFTDataset(_write(tmp_path, convos), tok)
    batch = SFTCollator(pad_id=0)([ds[0], ds[1]])
    B, L = batch["input_ids"].shape
    assert B == 2
    assert batch["labels"].shape == (B, L)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
