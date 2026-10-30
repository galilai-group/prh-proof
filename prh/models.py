"""Text-side networks: the BERT-style text encoder of every text arm, and the two text-only caption decoders.

Image encoders come from stable-pretraining (``LeJEPA``'s ViT) or timm (frozen DINOv2); see :mod:`prh.features`.
"""

import functools
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from stable_pretraining.backbone.utils import from_huggingface
from torch import nn
from transformers import AutoTokenizer

TOKENIZER = "bert-base-uncased"
#: width, attention heads and layers of the text encoder; widths match ViT-{T,S,B,L}/16.
ARCH = {"tiny": (192, 3, 12), "small": (384, 6, 12), "base": (768, 12, 12), "large": (1024, 16, 24)}


def arch_of_width(width: int) -> str:
    """Name of the architecture with that embedding width (checkpoints only store weights)."""
    return {w: a for a, (w, _, _) in ARCH.items()}[width]


@functools.cache
def tokenizer(name: str = TOKENIZER):
    """A HuggingFace tokenizer, loaded once."""
    return AutoTokenizer.from_pretrained(name)


def _checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _state(path: str | Path, prefix: str, ckpt: dict | None = None) -> dict[str, torch.Tensor]:
    """Sub-state-dict under ``prefix`` of a stable-pretraining (Lightning) checkpoint."""
    sd = (ckpt or _checkpoint(path))["state_dict"]
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


class TextEncoder(nn.Module):
    """Randomly initialized BERT, sized like the ViT of the same ``arch``.

    Bidirectional encoders (JEPA arms) pool the ``[CLS]`` token, the role of the ViT's CLS token. Causal ones
    (next-token prediction, caption decoder) mean-pool over non-padding tokens, since a causal ``[CLS]`` token
    only sees itself.

    Args:
        pad_id: padding token id; padding is masked out of attention and pooling.
        causal: causal (decoder) attention.
        arch: key of :data:`ARCH`.
    """

    def __init__(self, pad_id: int, causal: bool = False, arch: str = "small"):
        super().__init__()
        width, heads, layers = ARCH[arch]
        self.bert = from_huggingface(
            TOKENIZER,
            pretrained=False,
            hidden_size=width,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=4 * width,
            is_decoder=causal,
        )
        self.bert.pooler = None
        self.pad_id, self.embed_dim, self.causal = pad_id, width, causal

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Last hidden states ``[B, L, width]``."""
        return self.bert(input_ids=tokens, attention_mask=tokens.ne(self.pad_id)).last_hidden_state

    def pool(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Embedding ``[B, width]`` from the hidden states ``h`` of ``tokens``."""
        if not self.causal:
            return h[:, 0]
        valid = tokens.ne(self.pad_id).unsqueeze(-1).to(h.dtype)
        return (h * valid).sum(1) / valid.sum(1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pooled embedding ``[B, width]``."""
        return self.pool(self.hidden(tokens), tokens)

    @torch.no_grad()
    def encode(self, texts: list[str], max_len: int = 64) -> torch.Tensor:
        """Embeddings ``[N, width]`` of raw strings, read like the evaluation view (``[CLS]`` text ``[SEP]``, padded)."""
        ids = tokenizer()(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt").input_ids
        return self(ids.to(self.bert.device))

    @classmethod
    def from_checkpoint(cls, path: str | Path, pad_id: int) -> "TextEncoder":
        """Text encoder of a ``train.py`` run: a JEPA arm (``model.backbone.*``) or a next-token arm (``backbone.*``)."""
        sd = _state(path, "model.backbone.")
        causal = not sd
        sd = sd or _state(path, "backbone.")
        enc = cls(pad_id, causal=causal, arch=arch_of_width(sd["bert.embeddings.word_embeddings.weight"].shape[1]))
        enc.load_state_dict(sd)
        return enc.eval()


class _Conditioned(nn.Module):
    """Conditioning shared by the caption decoders: the frozen text-JEPA embedding, centred and unit-norm."""

    def __init__(self, cond_dim: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(cond_dim))

    @torch.no_grad()
    def update_mean(self, z: torch.Tensor, momentum: float = 0.01) -> None:
        """Track the mean of the conditioning embeddings (an EMA over training batches)."""
        self.mu.lerp_(z.float().mean(0), momentum)

    def condition(self, z: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        r"""Centred unit-norm conditioning vector; ``noise`` adds isotropic noise of that relative scale.

        A clean unit vector and its noised version have cosine :math:`(1+\sigma^2)^{-1/2}`; see
        :func:`prh.metrics.matched_noise` for the value that matches a given cross-modal alignment.
        """
        c = F.normalize(z.float() - self.mu, dim=-1)
        if noise:
            c = F.normalize(c + noise * torch.randn_like(c) / c.shape[1] ** 0.5, dim=-1)
        return c


class CaptionDecoder(_Conditioned):
    """Causal transformer, trained from scratch, that rebuilds a caption from a (frozen) text-JEPA embedding.

    The conditioning vector (:meth:`condition`) enters the decoder as a prefix token. Trained on text only; at
    test time it decodes rotated image embeddings.

    Args:
        pad_id: padding token id.
        cond_dim: width of the conditioning embedding (the text JEPA's).
        arch: decoder size, a key of :data:`ARCH`.
    """

    def __init__(self, pad_id: int, cond_dim: int, arch: str = "small"):
        super().__init__(cond_dim)
        self.decoder = TextEncoder(pad_id, causal=True, arch=arch)
        self.prefix = nn.Linear(cond_dim, self.decoder.embed_dim)
        self.config = {"pad_id": pad_id, "cond_dim": cond_dim, "arch": arch}

    def logits(self, c: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Next-token logits ``[B, L, V]`` given the conditioning ``c`` and the tokens so far (teacher forcing)."""
        words = self.decoder.bert.embeddings.word_embeddings
        x = torch.cat([self.prefix(c)[:, None], words(tokens)], 1)
        mask = torch.cat([torch.ones_like(tokens[:, :1]), tokens.ne(self.decoder.pad_id)], 1)
        return self.decoder.bert(inputs_embeds=x, attention_mask=mask).last_hidden_state @ words.weight.T

    def loss(self, c: torch.Tensor, tokens: torch.Tensor, captions: list[str] | None = None) -> torch.Tensor:
        """Cross-entropy of every caption token after ``[CLS]``, given ``c`` (``captions`` are not needed)."""
        logits = self.logits(c, tokens[:, :-1])[:, 1:]
        return F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten(), ignore_index=self.decoder.pad_id)

    @torch.no_grad()
    def generate(self, c: torch.Tensor, max_len: int = 32, beam: int = 1, nbest: int = 1) -> list[str]:
        """Beam search (greedy for ``beam=1``) with length-normalized log-probabilities.

        Args:
            c: conditioning vectors ``[N, cond_dim]`` from :meth:`condition`.
            max_len: maximum number of generated tokens.
            beam: beam width (at least ``nbest``).
            nbest: captions returned per row, best first.

        Returns:
            ``N * nbest`` captions, row by row.
        """
        beam = max(beam, nbest)
        tok, words = tokenizer(), self.decoder.bert.embeddings.word_embeddings
        n, pad = len(c), tok.pad_token_id
        pre = self.prefix(c).repeat_interleave(beam, 0)[:, None]
        ids = torch.full((n * beam, 1), tok.cls_token_id, device=c.device)
        score = torch.full((n, beam), -float("inf"), device=c.device)
        score[:, 0] = 0  # all beams start identical; keep one
        done = torch.zeros(n * beam, dtype=torch.bool, device=c.device)
        special = torch.tensor([pad, tok.cls_token_id, tok.mask_token_id, tok.unk_token_id], device=c.device)
        for _ in range(max_len):
            h = self.decoder.bert(inputs_embeds=torch.cat([pre, words(ids)], 1)).last_hidden_state[:, -1]
            logp = (h @ words.weight.T).log_softmax(-1)
            logp[:, special] = -float("inf")  # live hypotheses never emit special tokens other than [SEP]
            logp[done] = -float("inf")
            logp[done, pad] = 0  # finished hypotheses only extend with padding, at no cost
            score, idx = (score.view(-1, 1) + logp).view(n, -1).topk(beam, -1)
            src = (idx // logp.shape[1] + beam * torch.arange(n, device=c.device)[:, None]).view(-1)
            nxt = (idx % logp.shape[1]).view(-1)
            ids, done = torch.cat([ids[src], nxt[:, None]], 1), done[src] | (nxt == tok.sep_token_id)
            if done.all():
                break
        length = ids.ne(pad).sum(1).view(n, beam)
        best = (score / length).argsort(1, descending=True)[:, :nbest] + beam * torch.arange(n, device=c.device)[:, None]
        return [tok.decode(i, skip_special_tokens=True) for i in ids[best.flatten()]]

    @classmethod
    def from_checkpoint(cls, path: str | Path, ckpt: dict | None = None) -> "CaptionDecoder":
        """Decoder of a ``train.py --objective decode`` run (``decoder.*`` in its state dict)."""
        sd = _state(path, "decoder.", ckpt)
        out, inp = sd["prefix.weight"].shape  # decoder width, conditioning width
        return build_decoder({"pad_id": tokenizer().pad_token_id, "cond_dim": inp, "arch": arch_of_width(out)}, sd)


def _tidy(text: str) -> str:
    """Undo the lower-casing and spacing of the encoder's tokenizer: sentence case, ``t - shirt`` -> ``t-shirt``."""
    text = re.sub(r"(?<=\w) - (?=\w)", "-", text.strip())
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), text)


class LMDecoder(_Conditioned):
    """Pretrained causal language model, fine-tuned on text only to write the caption of a text-JEPA embedding.

    The conditioning vector becomes ``n_prefix`` input embeddings of the language model through a two-layer MLP
    (a soft prompt, as in ClipCap, Mokady et al. 2021). The target is the caption as written (training with
    :class:`prh.data.Captions`) or as the frozen encoder read it; :meth:`generate` tidies the casing and hyphens of
    the latter.

    Args:
        lm: HuggingFace name of a causal language model, e.g. ``"Qwen/Qwen2.5-1.5B"``.
        cond_dim: width of the conditioning embedding (the text JEPA's).
        n_prefix: number of prefix embeddings.
        pretrained: start from the language model's weights (``False``: its architecture only, to load a checkpoint).
    """

    def __init__(self, lm: str, cond_dim: int, n_prefix: int = 10, pretrained: bool = True):
        super().__init__(cond_dim)
        from transformers import AutoConfig, AutoModelForCausalLM

        if pretrained:
            self.lm = AutoModelForCausalLM.from_pretrained(lm, dtype=torch.float32)
        else:
            self.lm = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(lm), dtype=torch.float32)
        self.tok, width = tokenizer(lm), self.lm.config.hidden_size
        self.prefix = nn.Sequential(nn.Linear(cond_dim, width), nn.GELU(), nn.Linear(width, n_prefix * width))
        self.n_prefix, self.config = n_prefix, {"lm": lm, "cond_dim": cond_dim, "n_prefix": n_prefix}

    def _soft_prompt(self, c: torch.Tensor) -> torch.Tensor:
        embed = self.lm.get_input_embeddings().weight
        return self.prefix(c.to(embed.dtype)).view(len(c), self.n_prefix, -1).to(embed.dtype)

    def loss(self, c: torch.Tensor, tokens: torch.Tensor, captions: list[str] | None = None) -> torch.Tensor:
        """Cross-entropy of the caption and of its end, given ``c``.

        The target is ``captions`` as written if given, else the text of the encoder's token ids (lower-cased).
        """
        texts = list(captions) if captions is not None else tokenizer().batch_decode(tokens, skip_special_tokens=True)
        t = self.tok([s + self.tok.eos_token for s in texts], return_tensors="pt", padding=True, truncation=True, max_length=64)
        t = t.to(c.device)  # a few captions are paragraphs: their target stops after 64 tokens, like the encoder's view
        x = torch.cat([self._soft_prompt(c), self.lm.get_input_embeddings()(t.input_ids)], 1)
        mask = torch.cat([t.attention_mask.new_ones(len(c), self.n_prefix), t.attention_mask], 1)
        labels = torch.cat(
            [torch.full_like(mask[:, : self.n_prefix], -100), t.input_ids.masked_fill(t.attention_mask == 0, -100)], 1
        )
        return self.lm(inputs_embeds=x, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(
        self, c: torch.Tensor, max_len: int = 32, beam: int = 1, nbest: int = 1, samples: int = 0, temperature: float = 0.7
    ) -> list[str]:
        """Beam search (greedy for ``beam=1``) from the soft prompt of each row of ``c``: ``N * nbest`` captions, row by row.

        With ``samples``, each row's ``nbest`` beams are followed by that many nucleus samples (top-p 0.95), e.g. the
        candidates of :func:`prh.metrics.consensus`.
        """
        x = self._soft_prompt(c)
        kw = dict(
            inputs_embeds=x,
            attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=x.device),
            max_new_tokens=max_len,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        ids = self.lm.generate(**kw, num_beams=max(beam, nbest), num_return_sequences=nbest, do_sample=False)
        out = [_tidy(t) for t in self.tok.batch_decode(ids, skip_special_tokens=True)]
        if not samples:
            return out
        ids = self.lm.generate(**kw, do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=samples)
        drawn = [_tidy(t) for t in self.tok.batch_decode(ids, skip_special_tokens=True)]
        return [t for i in range(len(c)) for t in out[i * nbest : (i + 1) * nbest] + drawn[i * samples : (i + 1) * samples]]

    @classmethod
    def from_checkpoint(cls, path: str | Path, ckpt: dict | None = None) -> "LMDecoder":
        """Decoder of a ``train.py --objective decode --lm <name>`` run."""
        ckpt = ckpt or _checkpoint(path)
        sd = _state(path, "decoder.", ckpt)
        out, width = sd["prefix.2.weight"].shape
        return build_decoder({"lm": ckpt["hyper_parameters"]["lm"], "cond_dim": sd["mu"].shape[0], "n_prefix": out // width}, sd)


def build_decoder(config: dict, state: dict[str, torch.Tensor]) -> CaptionDecoder | LMDecoder:
    """A caption decoder from its ``config`` attribute and its state dict, in evaluation mode."""
    dec = LMDecoder(**config, pretrained=False) if "lm" in config else CaptionDecoder(**config)
    dec.load_state_dict({k: v.float() for k, v in state.items()})
    return dec.eval()


def rerank(candidates: list[str], z: torch.Tensor, encoder: TextEncoder, mu: torch.Tensor) -> list[str]:
    """For each row of ``z`` (text-JEPA space), the candidate whose embedding is closest to it.

    ``candidates`` holds the same number of captions for every row, row by row (``generate(..., nbest=k)``);
    cosines are taken after centring both sides by ``mu``, the text JEPA's mean.
    """
    k = len(candidates) // len(z)
    e = F.normalize(encoder.encode(candidates).double().cpu() - mu, dim=1).view(len(z), k, -1)
    best = (e * F.normalize(z.double().cpu() - mu, dim=1)[:, None]).sum(-1).argmax(1)
    return [candidates[i * k + j] for i, j in enumerate(best.tolist())]


def load_decoder(path: str | Path) -> CaptionDecoder | LMDecoder:
    """The caption decoder of a ``train.py --objective decode`` checkpoint, whichever kind it is."""
    ckpt = _checkpoint(path)
    lm = (ckpt.get("hyper_parameters") or {}).get("lm")
    return (LMDecoder if lm else CaptionDecoder).from_checkpoint(path, ckpt)
