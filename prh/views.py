"""Positive views. Image views are stable-pretraining multi-crop; text views are the text analogue (span crops).

Whether a JEPA is *good* or *bad* (paper, Sec. 2) is decided by what the views are drawn from: different
observations of one world state (good) or corruptions of a single observation (bad). Both modalities follow one
rule: given a list (the captions of an image, the frames of a clip) each view first draws its own element, given
a single caption or image every view corrupts it. :class:`TextView` does this for text, :func:`_pick` for images.
"""

import stable_pretraining as spt
import torch
from stable_pretraining.data import gpu_transforms as gt
from stable_pretraining.data import transforms
from torch import nn


class TextView(transforms.Transform):
    """Random contiguous span of a caption with some tokens masked, framed by ``[CLS]``/``[SEP]`` and padded.

    The text counterpart of ``RandomResizedCrop`` plus photometric noise. With the default ``scale=(1, 1)`` and
    no masking it is the deterministic evaluation view (truncation and padding only).

    Args:
        tok: tokenizer providing the ``cls``/``sep``/``mask``/``pad`` token ids.
        scale: range of the kept fraction of the caption.
        mask_ratio: probability of replacing each kept token by ``[MASK]``.
        max_len: output length, special tokens included.
        pair: next-token control: concatenate two different captions ``[CLS] a [SEP] b [SEP]``, uncropped (one
            caption if there is only one).
    """

    def __init__(self, tok, scale=(1.0, 1.0), mask_ratio=0.0, max_len=64, pair=False):
        super().__init__()
        self.scale, self.mask_ratio, self.max_len, self.pair = scale, mask_ratio, max_len, pair
        self.cls, self.sep, self.mask, self.pad = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id, tok.pad_token_id

    def _frame(self, *parts):
        out = torch.full((self.max_len,), self.pad, dtype=torch.long)
        seq = torch.cat([torch.tensor([self.cls]), *[t for p in parts for t in (p, torch.tensor([self.sep]))]])
        out[: len(seq)] = seq
        return out

    def __call__(self, x):
        ids = x["text"]
        if isinstance(ids, list) and self.pair and len(ids) > 1:
            i, j = torch.randperm(len(ids))[:2].tolist()
            h = (self.max_len - 3) // 2
            x["text"] = self._frame(ids[i][:h], ids[j][:h])
            return x
        if isinstance(ids, list):  # good views: each view draws its own caption of the image
            ids = ids[torch.randint(len(ids), ()).item()]
        ids = ids[: self.max_len - 2]
        if len(ids):  # an empty caption stays [CLS] [SEP]
            lo, hi = self.scale
            keep = max(1, round(len(ids) * (lo + (hi - lo) * torch.rand(()).item())))
            start = torch.randint(len(ids) - keep + 1, ()).item()
            ids = ids[start : start + keep].clone()
            ids[torch.rand(keep) < self.mask_ratio] = self.mask
        x["text"] = self._frame(ids)
        return x


class CaptionView(transforms.Transform):
    """Caption-decoder training item: the caption as written and its evaluation view.

    The language model's target keeps the caption's case; the frozen encoder reads its token ids.
    """

    def __init__(self, tok, max_len=64):
        super().__init__()
        self.tok, self.view = tok, TextView(tok, max_len=max_len)

    def __call__(self, x):
        ids = torch.tensor(self.tok(x["caption"], add_special_tokens=False)["input_ids"], dtype=torch.long)
        x["text"] = self.view({"text": ids})["text"]
        return x


class Photometric(nn.Module):
    """GPU half of the image views: stable-pretraining's kornia colour jitter, grayscale, blur, solarize.

    Crops stay on the CPU workers (cheap with PIL); the photometric operations, costly on CPU, run here with
    one call per resolution on the stacked views.

    Args:
        n_global: number of global views (the first views of the batch); the rest are local.
    """

    def __init__(self, n_global=2):
        super().__init__()
        self.n_global = n_global
        self.chain = gt.GPUCompose(
            [
                gt.GPURandomHorizontalFlip(p=0.5),
                gt.GPUColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                gt.GPURandomGrayscale(p=0.2),
                gt.GPUGaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
                gt.GPURandomSolarize(p=0.2),
                gt.GPUNormalize(**spt.data.static.ImageNet),
            ]
        )

    def forward(self, batch):
        for views in (batch["views"][: self.n_global], batch["views"][self.n_global :]):
            if not views:
                continue
            out = self.chain({"image": torch.cat([v["image"] for v in views]).float() / 255})["image"]
            for v, o in zip(views, out.chunk(len(views))):
                v["image"] = o
        return batch


def _pick(x):
    """Good image views: a view of a clip (a list of frames) draws its own frame."""
    if isinstance(x["image"], list):
        x["image"] = x["image"][torch.randint(len(x["image"]), ()).item()]
    return x


def _rgb(x):
    """``ToImage(rgb=True)`` only expands grayscale; a few ImageNet JPEGs are CMYK (4 channels)."""
    x["image"] = x["image"].convert("RGB")
    return x


def image_eval_view():
    """Resize 256, centre crop 224, ImageNet normalization: the view every image embedding is computed on."""
    return transforms.Compose(
        _rgb, transforms.Resize(256), transforms.CenterCrop((224, 224)), transforms.ToImage(rgb=True, **spt.data.static.ImageNet)
    )


def view_transforms(
    modality,
    tok,
    objective="lejepa",
    caption_views=False,
    views=(2, 6),
    gscale=(0.5, 1.0),
    lscale=(0.15, 0.5),
    mask=0.15,
    max_len=64,
):
    """Views of one training arm.

    Args:
        modality: ``"image"`` or ``"text"``.
        tok: tokenizer (text only).
        objective: ``"lejepa"`` (multi-view), ``"ntp"`` or ``"decode"`` (whole captions, no views).
        caption_views: next-token arm only: train on pairs of captions of one image (the paper's control).
        views: numbers of global and local views.
        gscale: kept caption fraction of global text views.
        lscale: kept caption fraction of local text views.
        mask: masked fraction of text-view tokens.
        max_len: tokens per text view.

    Returns:
        ``(train_cpu, eval, train_gpu)`` transforms; ``train_gpu`` is ``None`` for text.
    """
    if objective == "decode":  # decoders train on captions as written (prh.data.Captions)
        return CaptionView(tok, max_len), TextView(tok), None
    if objective == "ntp":
        return TextView(tok, max_len=128 if caption_views else 64, pair=caption_views), TextView(tok), None
    if modality == "image":

        def crop(size, scale):
            return transforms.Compose(
                _pick,
                _rgb,
                transforms.RandomResizedCrop((size, size), scale=scale),
                transforms.ToImage(torch.uint8, scale=False, rgb=True),
            )

        crops = [crop(224, (0.3, 1.0)) for _ in range(views[0])] + [crop(96, (0.05, 0.3)) for _ in range(views[1])]
        return transforms.MultiViewTransform(crops), image_eval_view(), Photometric(views[0])
    text = [TextView(tok, gscale, mask, max_len) for _ in range(views[0])] + [
        TextView(tok, lscale, mask, max_len) for _ in range(views[1])
    ]
    return transforms.MultiViewTransform(text), TextView(tok), None
