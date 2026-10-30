"""Embedding dumps: OnlineWriter's layout, DDP padding, epoch order."""

import pytest
import torch
from PIL import Image

from prh.features import last_embeddings, load_embeddings, save_embeddings


def _dump(folder, epoch, device, rows, emb):
    """One OnlineWriter validation batch (the file names stable-pretraining writes)."""
    folder.mkdir(parents=True, exist_ok=True)
    dump = {"embedding": emb, "proj": -emb, "pair_id": torch.tensor(rows), "is_test": torch.tensor(rows) % 2 == 0}
    torch.save(dump, folder / f"validation_epoch={epoch}_batch=0_device={device}.pt")


def test_save_and_load_roundtrip(tmp_path):
    x, pid, test = torch.randn(5, 3), torch.tensor([4, 2, 0, 3, 1]), torch.tensor([1, 0, 1, 0, 1]).bool()
    save_embeddings(tmp_path, x, pid, test)
    (y, qid, qtest), *rest = load_embeddings(tmp_path).values()
    assert not rest and torch.equal(qid, torch.arange(5))  # rows sorted by pair id
    assert torch.allclose(y, x[pid.argsort()].double()) and torch.equal(qtest, test[pid.argsort()])


def test_ddp_ranks_are_merged_and_padding_duplicates_dropped(tmp_path):
    emb = torch.arange(8.0).view(4, 2)
    _dump(tmp_path / "embeddings", 9, 0, [0, 2, 4, 0], emb)  # DistributedSampler pads rank 0 with row 0 again
    _dump(tmp_path / "embeddings", 9, 1, [1, 3, 5, 1], emb + 10)
    x, pid, test = last_embeddings(tmp_path)
    assert torch.equal(pid, torch.arange(6)) and torch.equal(test, pid % 2 == 0)
    assert x[pid == 3].tolist() == [[12.0, 13.0]]
    assert torch.equal(last_embeddings(tmp_path, key="proj")[0], -x)


def test_epochs_come_in_order_and_last_is_the_final_validation(tmp_path):
    for epoch in (19, 0, 9):
        _dump(tmp_path / "embeddings", epoch, 0, [0, 1], torch.full((2, 2), float(epoch)))
    epochs = load_embeddings(tmp_path)
    assert list(epochs) == [0, 9, 19] and last_embeddings(tmp_path)[0].unique().item() == 19


def test_several_runs_are_never_silently_merged(tmp_path):
    for run in ("a", "b"):
        _dump(tmp_path / run / "embeddings", 0, 0, [0, 1], torch.randn(2, 2))
    with pytest.raises(ValueError, match="exactly one run"):
        load_embeddings(tmp_path)
    with pytest.raises(ValueError):
        load_embeddings(tmp_path / "missing")


def test_cls_avg_features_concatenate_the_cls_and_mean_patch_tokens():
    import timm

    from prh.features import _ClsAvg

    net = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out, tokens = _ClsAvg(net)(x), net.forward_features(x)
    assert out.shape == (2, 2 * net.embed_dim)
    assert torch.allclose(out[:, : net.embed_dim], net(x), atol=1e-5)  # the CLS half is the usual embedding
    assert torch.allclose(out[:, net.embed_dim :], tokens[:, net.num_prefix_tokens :].mean(1), atol=1e-5)


def test_image_encoder_combinations_concatenate_unit_rms_blocks(monkeypatch):
    import timm

    from prh import features

    nets = {m: timm.create_model(f"vit_{m}_patch16_224", pretrained=False, num_classes=0).eval() for m in ("tiny", "small")}
    monkeypatch.setattr(features, "_image_net", lambda m, layers: (nets[m], lambda im: torch.zeros(3, 224, 224)))
    monkeypatch.setattr(features, "IMAGE_MODELS", {"tiny": "", "small": ""})
    images = [Image.new("RGB", (32, 32))] * 2
    x = features.embed_images("tiny,small", images)
    tiny, small = features.embed_images("tiny", images), features.embed_images("small", images)
    assert x.shape == (2, 192 + 384) and features.is_image_model_name("tiny,small")
    assert torch.allclose(x[:, :192], tiny / tiny.norm(dim=1, keepdim=True) * 192**0.5, atol=1e-5)
    assert torch.allclose(x[:, 192:].pow(2).mean(1), torch.ones(2), atol=1e-5)  # unit root-mean-square per coordinate
    assert torch.allclose(x[:, 192:], small / small.norm(dim=1, keepdim=True) * 384**0.5, atol=1e-5)


def test_per_layer_features_mean_pool_encoders_without_a_cls_token(monkeypatch):
    import timm

    from prh import features

    gap = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, class_token=False, global_pool="avg")
    cls = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).eval()
    images = [Image.new("RGB", (32, 32))] * 2
    x = torch.zeros(2, 3, 224, 224)
    for net, pool in ((gap.eval(), lambda t: t.mean(1)), (cls, lambda t: t[:, 0])):
        monkeypatch.setattr(features, "_image_net", lambda m, layers: (net, lambda im: torch.zeros(3, 224, 224)))
        layers = features.embed_images("any", images, layers=True)
        with torch.no_grad():
            tokens = net.blocks(net.norm_pre(net._pos_embed(net.patch_embed(x))))
        assert len(layers) == len(net.blocks) and torch.allclose(layers[-1], pool(tokens), atol=1e-5)
