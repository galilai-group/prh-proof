"""Zero-pair alignment (prh.pairfree) and the MSR-VTT evaluation, on synthetic worlds.

Tests that run the method of Schnaus et al. need their package (``pip install -e ".[pairfree]"``) and skip without it.
"""

import json

import pandas as pd
import pytest
import torch

import evaluate
from prh import pairfree
from prh.align import Rotation


def _world(n=600, d=12, k=6, seed=0):
    """Images and captions of one clustered world: the captions are the images rotated, plus a little noise."""
    g = torch.Generator().manual_seed(seed)
    z = (3 * torch.randn(k, d, generator=g))[torch.randint(k, (n,), generator=g)] + 0.3 * torch.randn(n, d, generator=g)
    q = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    return z, z @ q + 0.05 * torch.randn(n, d, generator=g), q


def test_procrustes_on_all_pairs_finds_the_rotation_and_a_map_round_trips(tmp_path):
    x, y, q = _world()
    m = pairfree.procrustes(x, y, torch.arange(len(x)))
    assert torch.allclose(m.W, q, atol=0.02)
    assert (m.similarity(x[:100], y[:100]).argmax(1) == torch.arange(100)).float().mean() > 0.9
    m.save(tmp_path / "m.pt", seed=3)
    assert torch.equal(pairfree.Map.load(tmp_path / "m.pt").W, m.W) and torch.load(tmp_path / "m.pt")["seed"] == 3


def test_prompts_are_read_as_data_and_embed_writes_their_layout(tmp_path, monkeypatch):
    lists = {"CIFAR_TEMPLATES": ["a {}."] * 18, "CIFAR10_CLASSES": ["cat"] * 10, "CIFAR100_CLASSES": ["dog"] * 100}
    lists["IMAGENET100_PROMPTS"] = ["x - y"] * 100
    (tmp_path / "datasets.py").write_text('raise SystemExit("never run")\n' + "".join(f"{k} = {v!r}\n" for k, v in lists.items()))
    ann = tmp_path / "raw/coco2014/annotations"
    ann.mkdir(parents=True)
    coco = {
        "images": [{"id": 2}, {"id": 1}],
        "annotations": [{"image_id": i, "caption": c} for i, c in ((1, "a"), (2, "b"), (1, "c"))],
    }
    for split in ("train", "val"):
        (ann / f"captions_{split}2014.json").write_text(json.dumps(coco))
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.0, 2.0]}
    monkeypatch.setattr(pairfree, "ROSETTA", tmp_path)
    monkeypatch.setattr(
        pairfree.features, "embed_texts", lambda m, texts, batch_size: torch.tensor([vectors.get(t, [1.0, 1.0]) for t in texts])
    )
    pairfree.embed("toy", "some-model")
    captions = torch.load(tmp_path / "embeddings/coco_val2014/language/toy.pt").float()
    assert torch.allclose(captions, torch.tensor([[0.7071, 0.7071], [0.0, 1.0]]), atol=1e-2)  # images sorted by id
    assert torch.load(tmp_path / f"embeddings/CIFAR-100/{pairfree.ZERO_SHOT['CIFAR-100'][0]}/toy.pt").shape == (100, 18, 2)
    assert torch.load(tmp_path / "embeddings/ImageNet-100/language/toy.pt").shape == (100, 2)


def test_their_method_aligns_a_world_without_pairs_and_the_shuffled_control_does_not():
    pytest.importorskip("unpaired_rosetta")
    x, y, _ = _world()
    rnd, ix, iy, known = pairfree.split(len(x), seed=0, pairs=10)
    assert set(ix.tolist()).isdisjoint(iy.tolist()) and len(ix) + len(iy) == len(x) and len(known) == 10
    assert torch.equal(pairfree.split(len(x), seed=0)[1], ix)  # the halves do not depend on the known pairs
    small = {"num_clusters": 6, "num_restarts": 3, "batch_size": 200, "num_iterations": 10}
    m, control = (pairfree.fit(x, y, 0, shuffle=s, **small) for s in (False, True))
    good, bad = (pairfree.scores(a, x[:200], y[:200], 0)["foscttm"] for a in (m, control))
    assert good < 0.15 and bad > 0.25
    assert pairfree.objective(m, x, y, 0, b=100) > pairfree.objective(control, x, y, 0, b=100)


def test_video_align_scores_clips_against_captions_through_each_map(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(0)
    n, d = 120, 6
    raw = torch.randn(n, d, generator=g, dtype=torch.float64)
    q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    ids = [f"video{i}" for i in range(n)]
    feats = {"ids": ids, "raw_mean": raw, "raw_mid": raw + 0.5 * torch.randn(n, d, generator=g, dtype=torch.float64)}
    feats |= {"text_mean": raw @ q, "js_ids": ids[100:], "js_text": raw[100:] @ q}  # the captions: a rotation of the clips
    torch.save(feats, tmp_path / "feats.pt")
    eye = Rotation(torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64))
    eye.save(tmp_path / "eye.pt")
    torch.save({"rotation": {"R": q, "mu_a": torch.zeros(d), "mu_b": torch.zeros(d)}}, tmp_path / "bundle.pt")
    (tmp_path / "annotations").mkdir()
    pd.DataFrame({"video_id": ids[:100]}).to_csv(tmp_path / "annotations/MSRVTT_train.9k.csv", index=False)
    monkeypatch.setattr(evaluate.data, "MSRVTT", tmp_path)
    evaluate.main(["video-align", str(tmp_path / "bundle.pt"), "--rotation", str(tmp_path / "eye.pt"), "--out", str(tmp_path)])
    r = pd.read_csv(tmp_path / "alignment.csv").set_index(["clip", "map"])["t2v R@1"]
    assert r["mean of 8 frames", "captioner map"] == 1.0 and r["mean of 8 frames", "refitted on the 9k training clips"] == 1.0
    assert r["mean of 8 frames", "--rotation"] < 0.5 and r["middle frame", "captioner map"] < 1.0
