"""Embeddings: read those dumped during training, or compute them with frozen encoders.

Two protocols are supported, and they differ on purpose:

* ``layers=False``: the paper's Flickr/COCO protocol. One embedding per input: the CLS token at 224 px with
  :func:`prh.views.image_eval_view` for images; for texts, our encoders' own pooling and the mean of the last
  hidden state of pretrained language models (truncated at 77 tokens).
* ``layers=True``: the protocol of the PRH paper (Huh et al., 2024) on WIT-1024. One feature per layer: the CLS
  token after every block with timm's preprocessing at native resolution for images, the masked mean of every
  hidden state for texts (truncated at 512 tokens).

Models are named by a key of :data:`IMAGE_MODELS` / :data:`TEXT_MODELS`, or by the path of a ``train.py``
checkpoint (``.ckpt``). Several image encoders joined by commas (``"dinov2-g+avg,dinov3-l+avg"``) give the
concatenation of their embeddings, each rescaled to unit root-mean-square per coordinate so that none dominates by
the scale of its features.
"""

import functools
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .models import TOKENIZER, TextEncoder, arch_of_width
from .views import TextView, image_eval_view

#: frozen self-supervised image encoders (timm names), embedded by their CLS token (I-JEPA: mean patch token); MAE
#: reconstructs masked pixels, the generative counterpart
IMAGE_MODELS = {
    "dinov2-s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "dinov2-l": "vit_large_patch14_dinov2.lvd142m",
    "dinov2-g": "vit_giant_patch14_dinov2.lvd142m",
    "dinov3-l": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3-h": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "dinov3-7b": "vit_7b_patch16_dinov3.lvd1689m",
    "ijepa-h": "vit_huge_patch14_gap_224.in22k_ijepa",
    "ijepa-g": "vit_giant_patch16_gap_224.in22k_ijepa",
    "mae-b": "vit_base_patch16_224.mae",
    "mae-l": "vit_large_patch16_224.mae",
    "mae-h": "vit_huge_patch14_224.mae",
    # CLIP's image towers, trained on LAION-2B image-text pairs: the "CLIP" panel of Huh et al. (2024, Fig. 3)
    "clip-b": "vit_base_patch16_clip_224.laion2b",
    "clip-l": "vit_large_patch14_clip_224.laion2b",
    "clip-h": "vit_huge_patch14_clip_224.laion2b",
}
#: ``<name>+avg``: the CLS token concatenated with the mean patch token, DINOv2's linear-evaluation features
IMAGE_MODELS |= {f"{k}+avg": v for k, v in IMAGE_MODELS.items() if k.startswith("dino")}
TEXT_MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen3b": "Qwen/Qwen2.5-3B",
    "qwen7b": "Qwen/Qwen2.5-7B",
    "mistral7b": "mistralai/Mistral-7B-v0.1",
    "qwen14b": "Qwen/Qwen2.5-14B",
    "qwen32b": "Qwen/Qwen2.5-32B",
    "qwen72b": "Qwen/Qwen2.5-72B",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",  # the language model of the LLaVA captioner behind Recap-DataComp
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    "bert": "bert-base-uncased",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",  # contrastive sentence encoder, 1B sentence pairs
    "clip": "openai/clip-vit-base-patch32",  # text tower, trained jointly with images
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_embeddings(out_dir: str | Path, embedding: torch.Tensor, pair_id: torch.Tensor, is_test: torch.Tensor) -> None:
    """Write frozen embeddings in ``OnlineWriter``'s layout (one validation epoch, 0), readable by :func:`load_embeddings`."""
    (Path(out_dir) / "embeddings").mkdir(parents=True, exist_ok=True)
    dump = {"embedding": embedding, "proj": embedding, "pair_id": torch.as_tensor(pair_id), "is_test": torch.as_tensor(is_test)}
    torch.save(dump, Path(out_dir) / "embeddings" / "validation_epoch=0_batch=0_device=0.pt")


def load_embeddings(run_dir: str | Path, key: str = "embedding") -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``{epoch: (features, pair_id, is_test)}`` from one run's ``OnlineWriter`` dumps, in order of epochs.

    Rows are sorted by pair id. Epoch 0 is the random initialization (the sanity pass). ``key`` is ``"embedding"`` (backbone) or ``"proj"``.
    """
    files = list(Path(run_dir).rglob("validation_epoch=*_batch=*.pt"))
    if len({f.parent for f in files}) != 1:
        raise ValueError(f"{run_dir} must hold the embeddings of exactly one run, found {len({f.parent for f in files})}")
    parts = defaultdict(list)
    for f in files:
        parts[int(re.search(r"epoch=(\d+)", f.name)[1])].append(torch.load(f))
    out = {}
    for epoch, ps in parts.items():
        cat = {k: torch.cat([p[k] for p in ps]) for k in (key, "pair_id", "is_test")}
        order = cat["pair_id"].argsort()
        keep = torch.ones(len(order), dtype=torch.bool)  # DDP ranks pad the validation set with duplicates
        keep[1:] = cat["pair_id"][order][1:] != cat["pair_id"][order][:-1]
        order = order[keep]
        out[epoch] = (cat[key][order].double(), cat["pair_id"][order], cat["is_test"][order].bool())
    return dict(sorted(out.items()))


def last_embeddings(run_dir: str | Path, key: str = "embedding") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(features, pair_id, is_test)`` of the last validation of a run (the only one of a frozen model)."""
    epochs = load_embeddings(run_dir, key)
    return epochs[max(epochs)]


def _batches(xs, size):
    return (xs[i : i + size] for i in range(0, len(xs), size))


class _ClsAvg(torch.nn.Module):
    """A timm ViT returning its CLS token and its mean patch token, concatenated."""

    def __init__(self, net):
        super().__init__()
        self.net, self.blocks = net, net.blocks

    def forward(self, x):
        tokens = self.net.forward_features(x)
        return torch.cat([tokens[:, 0], tokens[:, self.net.num_prefix_tokens :].mean(1)], 1)


@functools.lru_cache(maxsize=4)  # an app embeds one image at a time; load the networks once
def _image_net(model, layers):
    """``(net, transform)`` for a DINOv2 name or one of our image-JEPA checkpoints."""
    import timm

    if model in IMAGE_MODELS:
        name = IMAGE_MODELS[model]
        if layers:
            if model.endswith("+avg"):
                raise ValueError("per-layer features are CLS tokens; use the name without +avg")
            net = timm.create_model(name, pretrained=True, num_classes=0)
            return net, timm.data.create_transform(**timm.data.resolve_data_config(net.pretrained_cfg, model=net))
        net = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
        net = _ClsAvg(net) if model.endswith("+avg") else net
    else:
        from stable_pretraining.methods.lejepa import LeJEPA

        if model == "random-vit-s":  # untrained (seed 0): the floor of the unimodal checks
            torch.manual_seed(0)
            net = LeJEPA("vit_small_patch16_224").backbone
        else:
            sd = torch.load(model, map_location="cpu", weights_only=False)["state_dict"]
            jepa = LeJEPA(f"vit_{arch_of_width(sd['model.backbone.cls_token'].shape[-1])}_patch16_224")
            jepa.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
            net = jepa.backbone
    view = image_eval_view()
    return net, lambda im: view({"image": im})["image"]


def is_image_model_name(model: str) -> bool:
    """Whether ``model`` names frozen image encoders: a key of :data:`IMAGE_MODELS`, or several joined by commas."""
    return model == "random-vit-s" or all(m in IMAGE_MODELS for m in model.split(","))


@torch.no_grad()
def embed_images(model: str, images, layers: bool = False, batch_size: int = 64, dtype: torch.dtype = torch.float32):
    """Embeddings ``[N, d]`` (or one per block if ``layers``) of images given as paths or PIL images.

    ``dtype``: the encoder's weights and inputs (``torch.bfloat16`` halves the memory of a demo's encoders).
    """
    if "," in model:
        if layers:
            raise ValueError("per-layer features take one encoder")
        parts = [embed_images(m, images, batch_size=batch_size, dtype=dtype) for m in model.split(",")]
        return torch.cat([torch.nn.functional.normalize(x, dim=1) * x.shape[1] ** 0.5 for x in parts], 1)
    net, tf = _image_net(model, layers)
    net = net.eval().to(DEVICE, dtype)
    # per-layer features: the CLS token after each block, or the mean token for encoders without one (I-JEPA)
    pool = (lambda o: o[:, 0]) if getattr(net, "num_prefix_tokens", 1) else (lambda o: o.mean(1))
    feats, hooks = (
        [],
        [blk.register_forward_hook(lambda m, i, o: feats.append(pool(o).float().cpu())) for blk in net.blocks] if layers else [],
    )
    out = []
    for chunk in _batches(list(images), batch_size):
        x = torch.stack([tf((Image.open(im) if isinstance(im, (str, Path)) else im).convert("RGB")) for im in chunk])
        x = x.to(DEVICE, dtype)
        out.append(net(x).float().cpu())
    for h in hooks:
        h.remove()
    n = len(net.blocks)
    return [torch.cat(feats[i::n]) for i in range(n)] if layers else torch.cat(out)


@torch.no_grad()
def encode_images(model: str, x: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """:func:`embed_images` of a batch already in the encoders' input format (:func:`prh.views.image_eval_view`).

    For large image sets, whose decoding and resizing belong in data-loader workers.
    """
    parts = [_image_net(m, False)[0].eval().to(DEVICE, dtype)(x.to(DEVICE, dtype)).float().cpu() for m in model.split(",")]
    return (
        parts[0]
        if len(parts) == 1
        else torch.cat([torch.nn.functional.normalize(p, dim=1) * p.shape[1] ** 0.5 for p in parts], 1)
    )


@functools.lru_cache(maxsize=1)  # STS embeds two lists of sentences with the same model
def _text_net(model, layers, last_token=False):
    """Function mapping a list of strings to ``(hidden states, attention mask, final embedding)``."""
    if model not in TEXT_MODELS:  # one of our checkpoints: our own tokenizer view and pooling
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        torch.manual_seed(0)  # random-bert-s: untrained, the floor of the unimodal checks
        enc = TextEncoder(tok.pad_token_id) if model == "random-bert-s" else TextEncoder.from_checkpoint(model, tok.pad_token_id)
        enc, view = enc.eval().to(DEVICE), TextView(tok)

        def ours(texts):
            ids = torch.stack(
                [view({"text": torch.tensor(tok(t, add_special_tokens=False)["input_ids"])})["text"] for t in texts]
            ).to(DEVICE)
            mask = ids.ne(enc.pad_id)
            hs = enc.bert(input_ids=ids, attention_mask=mask, output_hidden_states=layers).hidden_states if layers else None
            if last_token:  # the last token's hidden state, the other usual embedding of a causal model
                return hs, mask, enc.hidden(ids)[torch.arange(len(ids)), mask.sum(1) - 1]
            return hs, mask, None if layers else enc(ids)

        return ours
    name = TEXT_MODELS[model]
    dtype = torch.bfloat16 if re.search(r"\d+b$", model) and not model.endswith(("0.5b", "1.5b")) else torch.float32
    big = re.search(r"(1[4-9]|[2-9]\d)b$", model) is not None  # 14B and up: split across the visible GPUs
    tok = AutoTokenizer.from_pretrained(name)
    net = AutoModel.from_pretrained(name, dtype=dtype, **({"device_map": "auto"} if big else {})).eval()
    net = net if big else net.to(DEVICE)
    tok.pad_token = tok.pad_token or tok.eos_token  # Llama/Mistral have none; padding is masked out of the mean

    def pretrained(texts):
        length = 512 if layers and model != "clip" else 77  # CLIP's positions stop at 77
        t = tok(texts, padding=True, truncation=True, max_length=length, return_tensors="pt").to(DEVICE)
        if model == "clip" and layers:  # the text tower's hidden states, mean-pooled like every other model's
            return net.text_model(**t, output_hidden_states=True).hidden_states, t["attention_mask"], None
        if model == "clip":
            o = net.get_text_features(**t)  # recent transformers wrap the projected embedding in pooler_output
            return None, t["attention_mask"], o if torch.is_tensor(o) else o.pooler_output
        out = net(**t, output_hidden_states=layers)
        return (
            (out.hidden_states, t["attention_mask"], None)
            if layers
            else (None, t["attention_mask"], _mean(out.last_hidden_state, t["attention_mask"]))
        )

    return pretrained


def _mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h.float() * m).sum(1) / m.sum(1)


@torch.no_grad()
def embed_texts(model: str, texts: list[str], layers: bool = False, batch_size: int = 64, last_token: bool = False):
    """Embeddings ``[N, d]`` (or the masked mean of every hidden state if ``layers``) of a list of strings.

    ``last_token``: for our checkpoints, the last token's final hidden state instead of the model's own pooling.
    """
    net, out = _text_net(model, layers, last_token), []
    for chunk in _batches(texts, batch_size):
        hs, mask, emb = net(chunk)
        out.append([_mean(h, mask).float().cpu() for h in hs] if layers else emb.float().cpu())
    return [torch.cat(layer) for layer in zip(*out)] if layers else torch.cat(out)
