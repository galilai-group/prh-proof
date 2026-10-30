"""Alignment, captioning and text-similarity scores."""

import math
from collections import Counter

import torch
import torch.nn.functional as F

from .align import procrustes, whitener


def mutual_knn(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    """Mean fraction of shared cosine ``k``-nearest neighbours between paired rows (Huh et al., 2024).

    Fits nothing; chance is about ``k / N``. ``k`` is capped at ``N - 1``.
    """
    k = min(k, len(a) - 1)

    def knn_mask(x):
        s = (F.normalize(x, dim=1) @ F.normalize(x, dim=1).T).fill_diagonal_(-float("inf"))
        return torch.zeros_like(s).scatter_(1, s.topk(k, dim=1).indices, 1.0)

    return ((knn_mask(a) * knn_mask(b)).sum(1) / k).mean().item()


def unbiased_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear CKA between paired rows with the unbiased HSIC estimator (Song et al., 2012), as in Huh et al. (2024).

    Fits nothing; invariant to rotations and isotropic scalings of either side, 1 when ``b`` is a rotation of ``a``.
    """

    def hsic(K, L):
        m = len(K)
        K, L = K.clone().fill_diagonal_(0), L.clone().fill_diagonal_(0)
        return ((K * L).sum() + K.sum() * L.sum() / ((m - 1) * (m - 2)) - 2 * K.sum(0) @ L.sum(1) / (m - 2)) / (m * (m - 3))

    K, L = a.double() @ a.double().T, b.double() @ b.double().T
    return (hsic(K, L) / (hsic(K, K) * hsic(L, L)).sqrt()).item()


def recall(a: torch.Tensor, b: torch.Tensor, ks=(1, 5)) -> list[float]:
    """Fraction of rows of ``a`` whose paired row of ``b`` is among their top-``k`` cosine matches in ``b``.

    Ties count against the true pair, so a collapsed encoder scores at chance rather than perfectly.
    """
    s = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T
    rank = (s >= s.diag()[:, None]).sum(1) - 1
    return [(rank < k).float().mean().item() for k in ks]


def alignment(a: torch.Tensor, b: torch.Tensor, is_test: torch.Tensor, test_means: bool = False) -> dict[str, float]:
    """The paper's protocol on paired rows: everything is fitted on the non-test rows, scored on the test rows.

    Args:
        a: embeddings ``[N, d_a]`` of one side of the pairs.
        b: embeddings ``[N, d_b]`` of the other side, row ``i`` paired with row ``i`` of ``a``.
        is_test: rows to score; the others fit the means, the rotation and the whitening.
        test_means: centre the test rows by their own means instead of the fitted ones (an unpaired statistic,
            for test pairs from another dataset).

    Returns:
        ``mknn`` (mutual 10-NN), ``R@1``/``R@5`` (A-to-B retrieval through one rotation: equality up to rotation)
        and ``wR@1``/``wR@5`` (same after whitening each side: equality up to an invertible linear map), and ``cos``,
        the mean cosine between a rotated row of ``a`` and its pair (the residual that sets the decoder's noise,
        :func:`matched_noise`).
    """
    va, vb, ta, tb = a[~is_test], b[~is_test], a[is_test], b[is_test]
    va, vb, ta, tb = (
        va - va.mean(0),
        vb - vb.mean(0),
        ta - (ta if test_means else a[~is_test]).mean(0),
        tb - (tb if test_means else b[~is_test]).mean(0),
    )
    rotated = ta @ procrustes(va, vb)
    r1, r5 = recall(rotated, tb)
    wa, wb = whitener(va), whitener(vb)
    wr1, wr5 = recall(ta @ wa @ procrustes(va @ wa, vb @ wb), tb @ wb)
    cos = F.cosine_similarity(rotated, tb, dim=1).mean().item()
    return {"mknn": mutual_knn(ta, tb), "R@1": r1, "R@5": r5, "wR@1": wr1, "wR@5": wr5, "cos": cos}


def _prh_prep(f):
    """The PRH paper's preprocessing: elementwise clip at the 0.95 quantile of absolute values, L2-normalize."""
    stride = max(1, -(-f.numel() // 10_000_000))  # torch.quantile takes at most 2^24 values
    t = torch.quantile(f.abs().flatten().float()[::stride], 0.95)
    return F.normalize(f.float().clamp(-t, t), dim=-1)


def prh_score(
    layers_a: list[torch.Tensor], layers_b: list[torch.Tensor], k: int = 10, metric: str = "mknn"
) -> tuple[float, int, int]:
    """Alignment score of the PRH paper: max over layer pairs of mutual ``k``-NN after its preprocessing.

    Each layer's features are clipped elementwise at the 0.95 quantile of their absolute values and
    L2-normalized, as in the released code of Huh et al. (2024). ``metric="cka"`` scores each layer pair by
    :func:`unbiased_cka` instead.

    Returns:
        ``(score, layer_a, layer_b)`` of the best pair.
    """
    score = {"mknn": lambda a, b: mutual_knn(a, b, k), "cka": unbiased_cka}[metric]
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # thousands of layer pairs on 5,000 rows: CPU takes hours
    fa, fb = [_prh_prep(f.to(dev)) for f in layers_a], [_prh_prep(f.to(dev)) for f in layers_b]
    return max((score(a, b), i, j) for i, a in enumerate(fa) for j, b in enumerate(fb))


def prh_subsample(layers_a, layers_b, n: int = 200, size: int = 800, k: int = 10, seed: int = 0) -> torch.Tensor:
    """:func:`prh_score` at its best layer pair on ``n`` random subsets of ``size`` rows (without replacement).

    Resampling with replacement would duplicate rows, which share their neighbours and inflate mutual k-NN.
    Subsets depend only on ``seed`` and the number of rows, so scores of different models are paired.
    """
    _, i, j = prh_score(layers_a, layers_b, k)
    a, b = _prh_prep(layers_a[i]), _prh_prep(layers_b[j])
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(len(a), generator=g)[:size] for _ in range(n)]
    return torch.tensor([mutual_knn(a[r], b[r], k) for r in rows])


def subsample_interval(full: float, subsets: torch.Tensor, n: int, size: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95% interval for a statistic of ``n`` rows, from its values on random subsets.

    On subsets of ``size`` rows drawn without replacement, an average-like statistic varies around its ``n``-row
    value with variance proportional to ``1/size - 1/n``, while the ``n``-row value varies around the truth with
    variance proportional to ``1/n``. The subsets' spread is therefore rescaled by ``sqrt(size / (n - size))``.
    """
    if not 0 < size < n:
        raise ValueError(f"subsets of {size} rows cannot estimate the spread of a statistic of {n} rows")
    se = subsets.std().item() * (size / (n - size)) ** 0.5
    return full - z * se, full + z * se


def whiten_top(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rows of ``x`` centred, projected on their top-``k`` principal directions and whitened (identity covariance)."""
    x = x - x.mean(0)
    evals, evecs = torch.linalg.eigh(x.T @ x / len(x))
    return x @ evecs[:, -k:] / evals[-k:].sqrt()


def scene_explained(f: torch.Tensor) -> float:
    """Fraction ``tr C_f^(1/2) / k`` of a whitened embedding that the world state explains (Thm. 3).

    ``f``: ``[states, observations, k]`` whitened embeddings of several observations of each state (e.g. the five
    captions of an image). ``C_f`` is the covariance of the per-state means, less the ``1/observations`` of the
    within-state covariance that averaging finitely many observations adds.
    """
    n, v, k = f.shape
    m = f.mean(1)
    within = torch.einsum("ijk,ijl->kl", f - m[:, None], f - m[:, None]) / (n * (v - 1))
    c = (m - m.mean(0)).T @ (m - m.mean(0)) / n - within / v
    return torch.linalg.eigvalsh(c).clamp(0, 1).sqrt().sum().item() / k


def rotation_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared distance ``min_R E||a R - b||^2`` over rotations, of paired whitened rows: ``2k - 2||a^T b / n||_*``."""
    return 2 * a.shape[1] - 2 * torch.linalg.svdvals(a.T @ b / len(a)).sum().item()


def matched_noise(cosine: float) -> float:
    """Decoder training noise that leaves the cosine a cross-modal map leaves: ``(1 + s^2)^(-1/2) = cosine``."""
    return (cosine**-2 - 1) ** 0.5


def _norm(s: str) -> str:
    return " ".join("".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").split())


def caption_scores(preds: list[str], refs: list[list[str]], more: bool = False) -> dict[str, float]:
    """BLEU-4 and CIDEr of ``preds`` against lists of reference captions (``pycocoevalcap``, lower-cased, no punctuation).

    ``more``: also METEOR (which runs java) and ROUGE-L, and the mean number of words of ``preds``.
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
    except ImportError as e:
        raise ImportError("caption scores need pycocoevalcap: pip install pycocoevalcap") from e

    res = {i: [_norm(p)] for i, p in enumerate(preds)}
    gts = {i: [_norm(r) for r in rs] for i, rs in enumerate(refs)}
    out = {"BLEU-4": Bleu(4).compute_score(gts, res, verbose=0)[0][3], "CIDEr": float(Cider().compute_score(gts, res)[0])}
    if more:
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge

        res = {i: [p or "a"] for i, (p,) in res.items()}  # METEOR's java reader rejects an empty caption
        out |= {"METEOR": Meteor().compute_score(gts, res)[0], "ROUGE-L": float(Rouge().compute_score(gts, res)[0])}
        out["words"] = sum(len(p.split()) for (p,) in res.values()) / len(res)
    return out


def consensus(candidates: list[str], sigma: float = 6.0) -> str:
    """Minimum-Bayes-risk choice among one image's candidate captions: the one most similar to all the others.

    The similarity is CIDEr's without document frequencies, which one image's candidates cannot estimate: the
    mean over n = 1..4 of the cosine between n-gram counts (the candidate's clipped by the other's), times a
    Gaussian penalty of width ``sigma`` words on their length difference. No reference caption is involved.
    """
    grams = []
    for c in candidates:
        w = _norm(c).split()
        g = [Counter(tuple(w[i : i + n]) for i in range(len(w) - n + 1)) for n in range(1, 5)]
        grams.append((g, [math.sqrt(sum(v * v for v in x.values())) for x in g], len(w)))

    def sim(a, b):
        (ga, na, la), (gb, nb, lb) = a, b
        cos = [sum(min(v, y[t]) * y[t] for t, v in x.items()) / (p * q) if p and q else 0.0 for x, y, p, q in zip(ga, gb, na, nb)]
        return sum(cos) / 4 * math.exp(-((la - lb) ** 2) / (2 * sigma**2))

    score = [sum(sim(a, b) for j, b in enumerate(grams) if j != i) for i, a in enumerate(grams)]
    return candidates[max(range(len(candidates)), key=score.__getitem__)]


def sts_spearman(a: torch.Tensor, b: torch.Tensor, scores) -> float:
    """Spearman correlation between the cosine of paired sentence embeddings and human similarity scores.

    Both sides are centred by their shared mean (JEPA embeddings are not zero-mean).
    """
    from scipy.stats import spearmanr

    mu = torch.cat([a, b]).mean(0)
    return float(spearmanr(F.cosine_similarity(a - mu, b - mu).numpy(), scores)[0])
