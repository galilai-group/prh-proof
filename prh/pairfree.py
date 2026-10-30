"""Alignment with no pair at all, on the benchmark of Schnaus et al. (2026, "Shared Geometry As A Rosetta Stone").

Their Wasserstein Procrustes maps image embeddings to text embeddings from two unpaired pools, the images of one half
of a paired corpus and the captions of the other half: it matches k-means centers of the two clouds by a quadratic
assignment of their kernels, averages the restarts into a first map and refines an orthogonal map by optimal
transport. We run their code unchanged with its defaults (``pip install -e ".[pairfree]"``: the commit and the QAP
solver version we ran) and their five seeds. No pair enters a fit, its means or the choice of a seed; the held-out
pairs only score the map.

Benchmarks (:func:`load`):

* ``coco2014``, their protocol: their stored DINOv2 ViT-B/14 mean-token embeddings of COCO train2014 (the corpus)
  and val2014 (40,504 scored pairs), each image's captions averaged into one text embedding; FOSCTTM, and zero-shot
  CIFAR-10 top-1, CIFAR-100 and ImageNet-100 top-5 with their class prompts. Data: ``python -m prh.data rosetta``;
  our text encoders: :func:`embed`.
* ``flickr``, ours: DINOv2 ViT-B/14 [CLS] and first-caption embeddings of Flickr30k (``evaluate.py embed <model>
  runs/ref/all_<name> --pairs flickr-all``); the 30,014 non-test pairs are the corpus, the 1,000 test pairs are
  scored (FOSCTTM, image-to-caption R@1 and R@5).
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from . import features
from .data import ROSETTA

#: their stored image embeddings: DINOv2 ViT-B/14, mean of the patch tokens, 224 px
IMAGE = "dinov2_vit-b14@224_mean"
#: their five seeds, in their order
SEEDS = [734796314, 576165995, 2197670066, 839703249, 2584932063]
#: zero-shot benchmarks: the folder of their class prompts, and the accuracy their paper reports
ZERO_SHOT = {
    "CIFAR-10": ("language_b953684b7ed3a62cbea9fe4e30935ee2", "top1"),
    "CIFAR-100": ("language_b953684b7ed3a62cbea9fe4e30935ee2", "top5"),
    "ImageNet-100": ("language", "top5"),
}
#: their Qwen3-8B text encoders: our name -> their file name. Their caption files hold each image's captions
#: ``[N, 5, d]``; :func:`load` reduces them once with their own ``reduce_texts``, so that a fit does not read 14 GB.
STORED = {"qwen3-embedding-8b-red": "qwen3-embedding-8b", "qwen3-8b-gen-red": "qwen3-8b-gen_69445574b3021b8a2733e87d8603d769"}


class Map:
    """An image-to-text map in their aligner interface: ``unit(x - mean_x) @ W`` scored against ``unit(y - mean_y)``."""

    def __init__(self, W: torch.Tensor, mean_x: torch.Tensor, mean_y: torch.Tensor):
        self.W, self.mean_x, self.mean_y = W.float(), mean_x.float(), mean_y.float()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float() - self.mean_x, dim=-1) @ self.W

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.transform(x) @ F.normalize(y.float() - self.mean_y, dim=-1).T

    def save(self, path: str | Path, **meta) -> None:
        torch.save({"W": self.W, "mean_x": self.mean_x, "mean_y": self.mean_y, **meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Map":
        m = torch.load(path)
        return cls(m["W"], m["mean_x"], m["mean_y"])


def _embeddings(dataset: str, modality: str, name: str) -> Path:
    return ROSETTA / "embeddings" / dataset / modality / f"{name}.pt"


def load(bench: str, text: str, ref: Path = Path("runs/ref")) -> tuple[torch.Tensor, ...]:
    """``(x, y, val_x, val_y)``: the paired corpus whose halves become the unpaired pools, and the scored pairs."""
    if bench == "flickr":
        (x, pid, test), (y, pid_y, test_y) = (features.last_embeddings(ref / d) for d in ("flickr_dinov2_all", f"all_{text}"))
        assert torch.equal(pid, pid_y) and torch.equal(test, test_y), "the two sides were embedded on different pairs"
        x, y = x.float(), y.float()
        return x[~test], y[~test], x[test], y[test]
    from unpaired_rosetta.embeddings import load_rows, reduce_texts

    for split in ("train", "val") if text in STORED else ():
        dst = _embeddings(f"coco_{split}2014", "language", text)
        if not dst.exists():  # row-wise, so chunk by chunk gives the same rows as their loader
            x = torch.load(dst.with_name(f"{STORED[text]}.pt"), mmap=True)
            torch.save(torch.cat([reduce_texts(x[i : i + 2048].clone()).float() for i in range(0, len(x), 2048)]), dst)
    return tuple(
        load_rows(_embeddings(f"coco_{s}2014", m, n))
        for s in ("train", "val")
        for m, n in (("vision", IMAGE), ("language", text))
    )


def zero_shot_data(text: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``{benchmark: (images, class prompts, labels)}`` of the zero-shot benchmarks whose prompts ``text`` has embedded."""
    from unpaired_rosetta.embeddings import load_rows

    out = {}
    for name, (folder, _) in ZERO_SHOT.items():
        prompts = _embeddings(name, folder, STORED.get(text, text))
        if prompts.exists():
            labels = torch.load(ROSETTA / "embeddings" / name / "labels.pt")
            out[name] = (load_rows(_embeddings(name, "vision", IMAGE)), load_rows(prompts), labels)
    return out


def split(n: int, seed: int, pairs: int = 0):
    """Their draws from one ``Randomness(seed)``: two disjoint halves of the corpus, then ``pairs`` known pairs."""
    from unpaired_rosetta.evaluation import disjoint_split, sample_pairs
    from unpaired_rosetta.randomness import Randomness

    rnd = Randomness(seed)
    ix, iy = disjoint_split(n, rnd)
    return rnd, ix, iy, sample_pairs(n, pairs, rnd) if pairs else None


def fit(x, y, seed: int, pairs: int = 0, shuffle: bool = False, workers: int = 1, **settings) -> Map:
    """Their Wasserstein Procrustes on the images of one half of the corpus and the captions of the other.

    ``pairs`` known pairs of the corpus join the fit (0: none). ``shuffle``, a negative control: each text coordinate
    permuted independently across the caption pool, which keeps every marginal and destroys the geometry.
    ``settings``: their hyperparameters (``num_clusters``, ``num_restarts``, ...), theirs by default.
    """
    from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

    rnd, ix, iy, known = split(len(x), seed, pairs)
    pool_y = y[iy]
    if shuffle:
        pool_y = pool_y.gather(0, torch.rand(pool_y.shape, generator=torch.Generator().manual_seed(12345)).argsort(0))
    known_x, known_y = (x[known], y[known]) if pairs else (None, None)
    wp = WassersteinProcrustes(num_workers=workers, **settings).fit(x[ix], pool_y, known_x, known_y, rnd)
    return Map(wp.weight, wp.mean_x, wp.mean_y)


def procrustes(x, y, rows: torch.Tensor) -> Map:
    """A paired reference: orthogonal Procrustes on the corpus pairs ``rows``, each side centred by its corpus mean."""
    mean_x, mean_y = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    u, _, vt = torch.linalg.svd((x[rows] - mean_x).double().T @ (y[rows] - mean_y).double(), full_matrices=False)
    return Map((u @ vt).float(), mean_x, mean_y)


def scores(m: Map, val_x, val_y, seed: int, zero_shot: dict | None = None) -> dict[str, float]:
    """Their FOSCTTM of the scored pairs (0 perfect, .5 chance), and image-to-caption R@1 and R@5 up to 10,000 pairs.

    With ``zero_shot`` (:func:`zero_shot_data`), also their zero-shot accuracy on each of its benchmarks.
    """
    from unpaired_rosetta.evaluation import retrieval_metrics, zero_shot_accuracy
    from unpaired_rosetta.randomness import Randomness

    out = retrieval_metrics(m, val_x, val_y, Randomness(seed))
    if len(val_x) <= 10_000:  # ties count against the true pair
        s = m.similarity(val_x, val_y).double()
        rank = (s >= s.diag()[:, None]).sum(1) - 1
        out["R@1"], out["R@5"] = ((rank < k).float().mean().item() for k in (1, 5))
    for name, (images, prompts, labels) in (zero_shot or {}).items():
        out[name] = zero_shot_accuracy(m, images, prompts, labels, Randomness(0))[ZERO_SHOT[name][1]]
    return out


def objective(m: Map, x, y, seed: int, b: int = 10_000) -> float:
    """Their objective on the unpaired pools of the fit with ``seed``, which reads no pair and no scored row.

    The mean matched score of their batched Hungarian assignment of ``unit(x) W`` to ``unit(y)`` on two blocks of
    ``b`` pooled rows drawn with a fixed generator, as in each refinement step: a model-selection score.
    """
    from unpaired_rosetta.assignment import batched_hungarian_matching
    from unpaired_rosetta.randomness import Randomness

    _, ix, iy, _ = split(len(x), seed)
    assert torch.allclose(x[ix].float().mean(0), m.mean_x.squeeze(0), atol=1e-5), "not the pools of this map's fit"
    px, py = m.transform(x[ix]), F.normalize(y[iy].float() - m.mean_y, dim=-1)
    r = Randomness(4242)
    ox, oy = r.permutation(len(px))[: 2 * b], r.permutation(len(py))[: 2 * b]
    rows, cols = batched_hungarian_matching(px[ox], py[oy], b)
    return (px[ox][rows] * py[oy][cols]).sum(1).mean().item()


def diagnose(x, y, val_x, val_y, seed: int) -> dict:
    """Where a fit goes wrong: the fit with ``seed`` replayed step by step, compared with a paired oracle at each stage.

    The replay calls their functions in their order with their random stream, so it ends on the fit's map; the
    oracle ``W*`` is Procrustes on all corpus pairs, for diagnosis only. Per restart: the fraction of k-means centers
    that the quadratic assignment matches as ``W*`` does, and the cosine of the restart's cross-covariance with ``W*``;
    then the cosine of their average (the first map) with ``W*``, and FOSCTTM on 5,000 scored pairs after the read-out
    and every 10 refinement steps.
    """
    from scipy.optimize import linear_sum_assignment
    from unpaired_rosetta.determinism import torch_threads
    from unpaired_rosetta.evaluation import retrieval_metrics
    from unpaired_rosetta.geometric_initialization import GeometricInitialization, centered_kernel, kmeans_centers
    from unpaired_rosetta.linalg import center_and_normalize, polar
    from unpaired_rosetta.qap import MPOptQAPSolver
    from unpaired_rosetta.randomness import Randomness
    from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

    rnd, ix, iy, _ = split(len(x), seed)
    wp = WassersteinProcrustes()
    c = wp.num_clusters
    cos = lambda a, b: ((a * b).sum() / (a.norm() * b.norm())).item()  # noqa: E731
    with torch_threads():  # their _fit, with the diagnostics
        mean_x, mean_y = x[ix].float().mean(0, keepdim=True), y[iy].float().mean(0, keepdim=True)
        X, Y = center_and_normalize(x[ix].float(), mean_x), center_and_normalize(y[iy].float(), mean_y)
        oracle = polar(center_and_normalize(x.float(), mean_x).T @ center_and_normalize(y.float(), mean_y))
        sub = torch.randperm(len(val_x), generator=torch.Generator().manual_seed(0))[:5000]
        fosc = lambda W: retrieval_metrics(Map(W, mean_x, mean_y), val_x[sub], val_y[sub], Randomness(0))["foscttm"]  # noqa: E731
        init, covs, restarts = GeometricInitialization(c, wp.num_restarts, wp.batch_size, 1), [], []
        for r in [init.draw_restart(len(X), len(Y), rnd) for _ in range(wp.num_restarts)]:  # their solve_restart
            A = kmeans_centers(X if r.rows_x is None else X[r.rows_x], c, r.kmeans_seed_x)
            B = kmeans_centers(Y if r.rows_y is None else Y[r.rows_y], c, r.kmeans_seed_y)
            perm = MPOptQAPSolver().solve(-centered_kernel(A), centered_kernel(B), None, r.solver_seed)
            _, truth = linear_sum_assignment(-(F.normalize(A @ oracle, dim=1) @ F.normalize(B, dim=1).T).double().numpy())
            covs.append((A * torch.full((c,), 1.0 / c)[:, None]).T @ B[perm])
            matched = (perm == torch.as_tensor(truth)).float().mean().item()
            restarts.append({"centers_matched_as_oracle": matched, "cosine_with_oracle": cos(covs[-1], oracle)})
        M = torch.stack(covs).mean(0)
        W = wp.readout(X, Y, M, None, None, rnd)
        trajectory = [fosc(W)]
        for step in range(wp.num_iterations):
            W = wp.refinement_step(X, Y, W, None, None, rnd)
            if step % 10 == 9:
                trajectory.append(fosc(W))
    final = retrieval_metrics(Map(W, mean_x, mean_y), val_x, val_y, Randomness(seed))["foscttm"]
    return {
        "seed": seed,
        "restarts": restarts,
        "first_map_cosine_with_oracle": cos(M, oracle),
        "foscttm_5k_readout_then_every_10_steps": trajectory,
        "foscttm_oracle_5k": fosc(oracle),
        "foscttm": final,
    }


def prompts() -> dict[str, list[str]]:
    """The class prompts of their zero-shot benchmarks, read as data from their ``datasets.py`` (not executed)."""
    c = {}
    for node in ast.parse((ROSETTA / "datasets.py").read_text()).body:
        for t in node.targets if isinstance(node, ast.Assign) else ():
            if getattr(t, "id", "") in ("CIFAR_TEMPLATES", "CIFAR10_CLASSES", "CIFAR100_CLASSES", "IMAGENET100_PROMPTS"):
                c[t.id] = ast.literal_eval(node.value)
    cifar = lambda classes: [t.format(k) for k in classes for t in c["CIFAR_TEMPLATES"]]  # noqa: E731
    return {
        "CIFAR-10": cifar(c["CIFAR10_CLASSES"]),
        "CIFAR-100": cifar(c["CIFAR100_CLASSES"]),
        "ImageNet-100": c["IMAGENET100_PROMPTS"],
    }


def coco_captions(split: str) -> list[list[str]]:
    """All captions of each COCO 2014 image of ``split``, images sorted by id (their order)."""
    data = json.loads((ROSETTA / "raw/coco2014/annotations" / f"captions_{split}2014.json").read_text())
    caps = defaultdict(list)
    for a in data["annotations"]:
        caps[a["image_id"]].append(a["caption"])
    return [caps[im["id"]] for im in sorted(data["images"], key=lambda im: im["id"])]


def embed(name: str, model: str, batch_size: int = 256) -> None:
    """A text encoder's embeddings in their layout under ``name``, in bfloat16.

    Each COCO 2014 image's captions, reduced as theirs (the normalized mean of the unit caption embeddings), and the
    class prompts of the zero-shot benchmarks.
    """
    for split in ("train", "val"):
        path = _embeddings(f"coco_{split}2014", "language", name)
        if not path.exists():
            caps = coco_captions(split)
            flat = F.normalize(features.embed_texts(model, [t for c in caps for t in c], batch_size=batch_size).float(), dim=1)
            image = torch.repeat_interleave(torch.arange(len(caps)), torch.tensor([len(c) for c in caps]))
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(F.normalize(torch.zeros(len(caps), flat.shape[1]).index_add_(0, image, flat), dim=1).bfloat16(), path)
    for bench, texts in prompts().items():
        path = _embeddings(bench, ZERO_SHOT[bench][0], name)
        if not path.exists():
            x = features.embed_texts(model, texts, batch_size=batch_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((x if bench == "ImageNet-100" else x.view(-1, 18, x.shape[1])).bfloat16(), path)
