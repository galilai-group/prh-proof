"""Forward functions of the three objectives, on tiny networks inside stable-pretraining's Module."""

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from stable_pretraining.backbone.utils import EvalOnly
from stable_pretraining.methods.lejepa import LeJEPA

from prh import forward
from prh.models import CaptionDecoder, TextEncoder
from prh.views import view_transforms


def _captions(n):
    return [[torch.randint(1000, 5000, (int(torch.randint(5, 20, ()).item()),)) for _ in range(5)] for _ in range(n)]


def _val_batch(tok, n=6):
    ids = torch.stack([view_transforms("text", tok)[1]({"text": c[0]})["text"] for c in _captions(n)])
    return {"text": ids, "pair_id": torch.arange(n), "is_test": torch.zeros(n, dtype=torch.bool)}


def test_lejepa_text_arm(tok):
    model = LeJEPA("vit_tiny_patch16_224")
    model.backbone = TextEncoder(tok.pad_token_id, arch="tiny")
    module = spt.Module(model=model, key="text", n_global=2, forward=forward.lejepa, hparams={})
    train_tf = view_transforms("text", tok, views=(2, 2))[0]
    items = [train_tf({"text": c})["views"] for c in _captions(8)]
    batch = {"views": [{"text": torch.stack([it[v]["text"] for it in items])} for v in range(4)]}
    out = module(batch, "fit")
    out["loss"].backward()
    assert out["loss"].isfinite() and model.backbone.bert.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    val = module.eval()(_val_batch(tok), "validate")
    assert val["embedding"].shape == (6, 192) and val["proj"].shape[0] == 6 and torch.equal(val["pair_id"], torch.arange(6))


def test_next_token_is_cross_entropy_of_the_tied_head(tok):
    enc = TextEncoder(tok.pad_token_id, causal=True, arch="tiny")
    module = spt.Module(backbone=enc, key="text", forward=forward.next_token, hparams={}).eval()
    batch = _val_batch(tok)
    out = module(batch, "validate")
    tokens = batch["text"]
    logits = enc.hidden(tokens)[:, :-1] @ enc.bert.embeddings.word_embeddings.weight.T
    mask = tokens[:, 1:].ne(tok.pad_token_id)
    manual = -logits.log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)[mask].mean()
    assert torch.allclose(out["loss"], manual, atol=1e-5) and torch.allclose(out["embedding"], enc(tokens))


def test_decoder_trains_on_noised_frozen_embeddings(tok):
    enc = TextEncoder(tok.pad_token_id, arch="tiny")
    dec = CaptionDecoder(tok.pad_token_id, enc.embed_dim, arch="tiny")
    module = spt.Module(decoder=dec, encoder=EvalOnly(enc), noise=3.0, key="text", forward=forward.decode, hparams={})
    batch = _val_batch(tok)
    a, b = module(batch, "fit"), module(batch, "fit")
    assert not torch.allclose(a["embedding"], b["embedding"])  # noise in training
    assert dec.mu.abs().sum() > 0  # the running mean moved
    a["loss"].backward()
    assert dec.prefix.weight.grad.abs().sum() > 0 and all(p.grad is None for p in enc.parameters())
    module.eval()
    v1, v2 = module(batch, "validate"), module(batch, "validate")
    assert torch.equal(v1["embedding"], v2["embedding"]) and torch.allclose(v1["embedding"], dec.condition(enc(batch["text"])))
    assert (
        torch.allclose(v1["embedding"].norm(dim=1), torch.ones(6))
        and F.cosine_similarity(a["embedding"], v1["embedding"]).mean() < 0.5
    )
