"""End to end: train.py trains two arms with stable-pretraining, and evaluate.py aligns them (CPU, a few steps)."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import train
from prh.features import load_embeddings
from prh.models import load_decoder

REPO = Path(__file__).resolve().parents[1]


def _train(roots, *args):
    env = os.environ | {"PRH_FLICKR": str(roots / "flickr"), "PRH_COCO": str(roots / "coco"), "CUDA_VISIBLE_DEVICES": ""}
    cmd = [
        sys.executable,
        "train.py",
        "--arch",
        "tiny",
        "--epochs",
        "2",
        "--val-every",
        "2",
        "--batch-size",
        "4",
        "--workers",
        "1",
        "--views",
        "2",
        "2",
        "--cache-dir",
        str(roots / "runs"),
        *args,
    ]
    subprocess.run(cmd, cwd=REPO, env=env, check=True, capture_output=True, timeout=900)
    run = max((roots / "runs").rglob("last.ckpt"), key=os.path.getmtime).parents[1]
    return run


@pytest.mark.slow
def test_two_arms_train_separately_and_align(roots):
    image = _train(roots, "--modality", "image")
    text = _train(roots, "--modality", "text", "--caption-views", "--data", "flickr+coco")
    for run in (image, text):
        dumps = load_embeddings(run)
        assert sorted(dumps) == [0, 1]  # random init (sanity pass) and the epoch-1 validation
        x, pid, is_test = dumps[1]
        assert len(x) == 4 and is_test.sum() == 2
    out = subprocess.run(
        [sys.executable, "evaluate.py", "align", str(image), str(text)], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    assert "R@1" in out and len(out.strip().splitlines()) == 3


@pytest.mark.slow
def test_resume_continues_a_run_in_a_new_run_dir(roots):
    first = _train(roots, "--modality", "text")
    second = _train(roots, "--modality", "text", "--epochs", "4", "--resume", str(first / "checkpoints" / "last.ckpt"))
    before, after = load_embeddings(first), load_embeddings(second)
    assert second != first and sorted(after) == [1, 3]  # the sanity pass sees the restored epoch-1 weights, no epoch 0
    assert torch.allclose(after[1][0], before[1][0], atol=1e-5) and not torch.allclose(after[3][0], before[1][0])


@pytest.mark.slow
def test_supervised_decoder_trains_on_precomputed_features_and_keeps_every_epoch(roots):
    pd.DataFrame({"captions": [[f"photo {i} caption {j}" for j in range(5)] for i in range(10)]}).to_parquet(roots / "p.parquet")
    torch.save(torch.randn(10, 24), roots / "p.pt")
    features = ["--features", str(roots / "p.pt"), "--data", f"{roots / 'p.parquet'}:6"]
    run = _train(roots, "--modality", "text", "--objective", "decode", *features, "--noise", "0", "--keep-epochs")
    kept = sorted((run / "checkpoints").glob("epoch=*.ckpt"))
    assert [p.name.split("-")[0] for p in kept] == ["epoch=0", "epoch=1"]  # 30 captions, batches of 4: 7 steps each
    assert "optimizer_states" not in torch.load(kept[0], map_location="cpu", weights_only=False)  # weights only
    dec = load_decoder(kept[1])
    assert dec.mu.shape == (24,) and dec.mu.abs().sum() > 0
    assert not list(run.rglob("validation_epoch=*.pt"))  # nothing validates in the loop


def test_features_replace_the_encoder_of_a_decoder():
    args = train.parse_args(["--modality", "text", "--objective", "decode", "--features", "x.pt", "--data", "p.parquet:3"])
    assert args.encoder is None and args.features == Path("x.pt") and not args.keep_epochs


@pytest.mark.parametrize(
    "argv",
    [
        ["--modality", "text", "--val-every", "1"],
        ["--modality", "text", "--caption-views", "1.5"],
        ["--modality", "image", "--objective", "ntp"],
        ["--modality", "text", "--objective", "decode"],
        ["--modality", "text", "--encoder", "x.ckpt"],
        ["--modality", "text", "--lm", "Qwen/Qwen2.5-0.5B"],
        ["--modality", "text", "--objective", "decode", "--encoder", "x.ckpt", "--features", "x.pt"],
        ["--modality", "text", "--features", "x.pt"],
    ],
)
def test_inconsistent_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        train.parse_args(argv)


def test_init_decoder_carries_the_language_model_over_to_another_conditioning_width():
    from prh.models import CaptionDecoder

    text, image = CaptionDecoder(0, 32, arch="tiny"), CaptionDecoder(0, 48, arch="tiny")
    with pytest.raises(RuntimeError):  # without partial, a width mismatch is an error, as before
        train.init_decoder(image, text.state_dict())
    loaded = train.init_decoder(image, text.state_dict(), partial=True)
    assert "prefix.weight" not in loaded and "mu" not in loaded  # the input map and the mean have the new width
    assert all(torch.equal(image.state_dict()[k], text.state_dict()[k]) for k in loaded)
    assert any(k.startswith("decoder.") for k in loaded)  # the decoder itself carries over
