"""The command-line scripts' own logic, on synthetic embeddings (training is covered by test_train.py)."""

import pandas as pd
import pytest
import torch
from PIL import Image

import evaluate
from prh import captioner as bundle
from prh.align import Rotation
from prh.features import save_embeddings
from prh.models import CaptionDecoder, TextEncoder, tokenizer


def _run_dumps(folder, epochs, x, pid, is_test):
    """A training run's validation dumps: the same pairs at several epochs, ``x[e]`` at epoch ``e``."""
    (folder / "embeddings").mkdir(parents=True)
    for e in epochs:
        dump = {"embedding": x[e], "proj": x[e], "pair_id": pid, "is_test": is_test}
        torch.save(dump, folder / "embeddings" / f"validation_epoch={e}_batch=0_device=0.pt")


def test_align_pairs_a_frozen_side_with_every_epoch_and_finds_the_rotation(tmp_path):
    g = torch.Generator().manual_seed(0)
    n, d = 400, 8
    pid, is_test = torch.arange(n), torch.arange(n) >= 200
    a = torch.randn(n, d, generator=g)
    q = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    save_embeddings(tmp_path / "frozen", a, pid, is_test)
    # epoch 0: unrelated; epoch 9: exactly a rotation of the frozen side
    _run_dumps(tmp_path / "run", (0, 9), {0: torch.randn(n, d, generator=g), 9: a @ q}, pid, is_test)
    evaluate.main(["align", str(tmp_path / "frozen"), str(tmp_path / "run"), "--csv", str(tmp_path / "out.csv")])
    df = pd.read_csv(tmp_path / "out.csv").set_index("epoch")
    assert list(df.index) == [0, 9] and df.loc[9, "R@1"] == 1.0 and df.loc[0, "R@1"] < 0.1
    evaluate.main(["align", str(tmp_path / "frozen"), str(tmp_path / "run"), "--last", "--csv", str(tmp_path / "last.csv")])
    assert pd.read_csv(tmp_path / "last.csv").epoch.tolist() == [9]


def test_align_val_only_never_touches_test_pairs(tmp_path):
    g = torch.Generator().manual_seed(0)
    n, d = 400, 8
    pid, is_test = torch.arange(n), torch.arange(n) >= 200
    a = torch.randn(n, d, generator=g)
    q = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    b = torch.where(is_test[:, None], torch.randn(n, d, generator=g), a @ q)  # test pairs carry no signal
    save_embeddings(tmp_path / "frozen", a, pid, is_test)
    _run_dumps(tmp_path / "run", (9,), {9: b}, pid, is_test)
    run = ["align", str(tmp_path / "frozen"), str(tmp_path / "run"), "--split", "100"]
    evaluate.main(run + ["--val-only", "--csv", str(tmp_path / "val.csv")])
    evaluate.main(run + ["--csv", str(tmp_path / "all.csv")])
    assert pd.read_csv(tmp_path / "val.csv")["R@1"].item() == 1.0 > pd.read_csv(tmp_path / "all.csv")["R@1"].item()


def test_wit_embed_requires_a_name_for_checkpoints():
    with pytest.raises(SystemExit, match="--name"):
        evaluate.main(["wit-embed", "some/run/last.ckpt"])


def test_captioner_chains_encoder_rotation_and_decoder(tmp_path, monkeypatch):
    dec = CaptionDecoder(0, cond_dim=6, arch="tiny").eval()
    torch.save({"state_dict": {"decoder." + k: v for k, v in dec.state_dict().items()}}, tmp_path / "decoder.ckpt")
    Rotation(torch.eye(4, 6, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)).save(
        tmp_path / "rotation.pt"
    )
    seen = []

    def fake_encoder(model, images, dtype=torch.float32):
        seen.append((model, len(images)))
        return torch.randn(len(images), 4)

    monkeypatch.setattr(bundle, "embed_images", fake_encoder)
    captioner = bundle.Captioner.from_runs(tmp_path / "decoder.ckpt", tmp_path / "rotation.pt", image_model="dinov2-s", beam=2)
    captions = captioner([Image.new("RGB", (32, 32))] * 3)
    assert seen == [("dinov2-s", 3)] and len(captions) == 3 and all(isinstance(c, str) for c in captions)
    # the bundle reloads with the same rotation, settings and (bfloat16-rounded) decoder
    captioner.save(tmp_path / "captioner.pt")
    loaded = bundle.Captioner.load(tmp_path / "captioner.pt")
    assert (loaded.image_model, loaded.beam, loaded.samples, loaded.dec.config) == ("dinov2-s", 2, 0, captioner.dec.config)
    assert torch.equal(loaded.rot.R, captioner.rot.R) and torch.equal(loaded.dec.mu, captioner.dec.mu)
    w, v = loaded.dec.prefix.weight, captioner.dec.prefix.weight
    assert torch.equal(w, v.to(torch.bfloat16).float())


def test_captioner_reads_descriptions_with_its_decoders_text_jepa(tmp_path):
    pad = tokenizer().pad_token_id
    enc = TextEncoder(pad, arch="tiny").eval()
    torch.save({"state_dict": {"model.backbone." + k: v for k, v in enc.state_dict().items()}}, tmp_path / "jepa.ckpt")
    dec = CaptionDecoder(pad, cond_dim=192, arch="tiny").eval()
    sd = {"decoder." + k: v for k, v in dec.state_dict().items()}
    torch.save({"state_dict": sd, "hyper_parameters": {"encoder": str(tmp_path / "jepa.ckpt")}}, tmp_path / "decoder.ckpt")
    Rotation(
        torch.eye(4, 192, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), torch.ones(192, dtype=torch.float64)
    ).save(tmp_path / "rotation.pt")
    captioner = bundle.Captioner.from_runs(tmp_path / "decoder.ckpt", tmp_path / "rotation.pt")
    q = captioner.read(["a dog on a beach", "a bus"])
    centred = torch.nn.functional.normalize(enc.encode(["a dog on a beach", "a bus"]).double() - 1, dim=1)
    assert q.shape == (2, 192) and torch.allclose(q, centred, atol=1e-6)
    captioner.save(tmp_path / "captioner.pt")  # the bundle keeps the text JEPA exactly
    assert torch.equal(bundle.Captioner.load(tmp_path / "captioner.pt").read(["a dog on a beach", "a bus"]), q)


def test_encode_images_matches_embed_images():
    from prh.features import embed_images, encode_images
    from prh.views import image_eval_view

    ims = [Image.new("RGB", (300, 260), c) for c in [(10, 200, 30), (240, 240, 240)]]
    x = torch.stack([image_eval_view()({"image": im})["image"] for im in ims])
    assert torch.allclose(encode_images("random-vit-s", x), embed_images("random-vit-s", ims), atol=1e-5)


def test_photos_are_upright_and_opaque(tmp_path):
    im = Image.new("RGB", (40, 20))
    exif = im.getexif()
    exif[0x0112] = 6  # a phone held upright: stored sideways, rotated on display
    im.save(tmp_path / "phone.jpg", exif=exif)
    assert bundle.photo(tmp_path / "phone.jpg").size == (20, 40)
    assert bundle.photo(Image.new("RGBA", (4, 4), (255, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
    assert bundle.photo(Image.new("L", (4, 4), 7)).getpixel((0, 0)) == (7, 7, 7)


def test_supervised_baselines_fit_a_non_orthogonal_map():
    import caption

    g = torch.Generator().manual_seed(0)
    a = torch.randn(3000, 16, generator=g, dtype=torch.float64)
    M = torch.randn(16, 8, generator=g, dtype=torch.float64)  # not a rotation: Procrustes cannot recover it
    b = a @ M + 3.0
    maps = caption.supervised_maps(a, b, a.mean(0), b.mean(0), steps=1500)
    err = {name: ((f(a) - b).norm() / (b - b.mean(0)).norm()).item() for name, f in maps.items()}
    assert err["linear map, any matrix"] < 1e-2 and err["two-layer MLP"] < 0.3


def test_wit_scores_the_cached_features_of_the_chosen_benchmark(tmp_path, monkeypatch, capsys):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, 8, generator=g)
    monkeypatch.setitem(evaluate.BENCH, "coco", tmp_path)
    torch.save([torch.randn(50, 8, generator=g), x], tmp_path / "a.pt")
    torch.save([x.clone()], tmp_path / "b.pt")  # the PRH clipping is per coordinate: not rotation invariant
    evaluate.main(["wit", "a:b", "--bench", "coco"])
    assert "PRH COCO-5k mutual-kNN a x b: 1.000 (layers 1/1, 0/0)" in capsys.readouterr().out


def test_report_writes_what_tables_reads(tmp_path, monkeypatch):
    import tables

    calls = []
    monkeypatch.setattr(evaluate, "embed", lambda a: calls.append(("embed", a.pairs, a.last_token)))
    monkeypatch.setattr(evaluate, "wit_embed", lambda a: calls.append(("wit-embed", a.name, a.bench)))
    table = pd.DataFrame({"epoch": [0], "mknn": [0.21], "R@1": [0.344], "R@5": [0.658]})
    monkeypatch.setattr(evaluate, "align", lambda a: calls.append(("align", a.split, a.val_only)) or table)
    monkeypatch.setattr(
        evaluate, "wit", lambda a: [f"PRH WIT-1024 mutual-kNN {p.replace(':', ' x ')}: 0.09{i}" for i, p in enumerate(a.pairs)]
    )
    evaluate.main(["report", "run/last.ckpt", "--name", "t", "--results", str(tmp_path)])
    assert calls == [("embed", "flickr-all", False), ("align", 0, False), ("wit-embed", "text-t", "wit")]
    assert tables.scores(tmp_path / "t.txt") == (0.091, 0.344)  # DINOv2 ViT-B, the second of S/B/L/g


def test_tables_gather_the_grid_and_fit_the_law(tmp_path):
    import numpy as np

    import tables

    for tag, r1 in (("phase-n50-q0.1", 0.278), ("phase-s80-q1", 0.409)):
        (tmp_path / f"{tag}.txt").write_text(
            f"DINOv2 x {tag}, 30k-pair rotation\n     0 0.2 {r1} 0.6\nPRH WIT-1024 mutual-kNN dinov2 x text-{tag}: 0.062\n"
        )
    grid = tables.phase_grid(tmp_path)
    assert grid[["images_M", "samples_M", "q", "r1"]].values.tolist() == [[10.2, 20, 0.1, 0.278], [10.2, 80, 1.0, 0.409]]
    q, m, n = np.meshgrid([0, 0.01, 0.1, 0.3, 1], [5, 20, 80], [0.4, 10, 50], indexing="ij")
    truth = (0.19, 0.02, 0.02, 0.3, 1.4, 0.8, 0.3)
    cells = pd.DataFrame({"q": q.ravel(), "samples_M": m.ravel(), "images_M": n.ravel()})
    cells["r1"] = tables.law((cells.q, cells.samples_M, cells.images_M), *truth)
    theta, _, rmse = tables.fit_law(cells)
    assert rmse < 1e-4 and np.allclose(tables.law((1, 160, 10), *theta), tables.law((1, 160, 10), *truth), atol=1e-3)
