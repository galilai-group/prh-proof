"""Shared fixtures. Tests run on CPU with tiny tensors; ``slow`` marks end-to-end training runs."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from prh import data
from prh.models import TOKENIZER


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Fake Flickr30k (Karpathy CSV and images), COCO Karpathy captions and Recap-DataComp text parquet."""
    flickr = tmp_path / "flickr"
    (flickr / "flickr30k-images").mkdir(parents=True)
    rows = []
    for i in range(12):
        Image.new("RGB", (64, 48), (i * 20, 0, 0)).save(flickr / "flickr30k-images" / f"{i}.jpg")
        split = "train" if i < 8 else "val" if i < 10 else "test"
        rows.append(
            {
                "filename": f"{i}.jpg",
                "img_id": 100 + i,
                "split": split,
                "raw": json.dumps([f"photo {i} caption {j}" for j in range(5)]),
            }
        )
    pd.DataFrame(rows).to_csv(flickr / "flickr_annotations_30k.csv", index=False)
    coco = tmp_path / "coco"
    coco.mkdir()
    for s, n in (("train", 3), ("restval", 2)):
        pd.DataFrame({"sentences": [[f"coco {s} {i} {j}" for j in range(5)] for i in range(n)]}).to_parquet(
            coco / f"coco_{s}.parquet"
        )
    recap = tmp_path / "recap" / "data" / "train_data"
    recap.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "org_caption": ["alt text", None],
                "re_caption": ["a long recaption", "only this"],
                "re_caption_condition_diverse_topk": ["another one", ""],
            }
        ),
        recap / "train-00000.parquet",
    )
    msrvtt, frames = tmp_path / "msrvtt", tmp_path / "frames"  # 12 clips: 6 train, 3 val, 3 test; frame k is grey level 20 k
    (msrvtt / "annotations").mkdir(parents=True)
    ids = [f"video{i}" for i in range(12)]
    pd.DataFrame({"video_id": ids[:6]}).to_csv(msrvtt / "annotations" / "MSRVTT_train.7k.csv", index=False)
    pd.DataFrame({"video_id": ids[:9]}).to_csv(msrvtt / "annotations" / "MSRVTT_train.9k.csv", index=False)
    pd.DataFrame({"video_id": ids[9:], "sentence": [f"test sentence {v}" for v in ids[9:]]}).to_csv(
        msrvtt / "annotations" / "MSRVTT_JSFUSION_test.csv", index=False
    )
    sentences = [{"video_id": v, "caption": f"clip {v} caption {j}"} for v in ids for j in range(20)]
    json.dump({"sentences": sentences}, open(msrvtt / "annotations" / "MSRVTT_data.json", "w"))
    for v in ids:
        (frames / v).mkdir(parents=True)
        for k in range(data.N_FRAMES):
            Image.new("RGB", (64, 48), (20 * k,) * 3).save(frames / v / f"{k}.jpg")
    for name, path in (("FLICKR", flickr), ("COCO", coco), ("RECAP", tmp_path / "recap"), ("MSRVTT", msrvtt), ("FRAMES", frames)):
        monkeypatch.setattr(data, name, path)
    return tmp_path
