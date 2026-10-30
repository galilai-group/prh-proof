"""Linear maps between two embedding spaces: the single rotation that the PRH says relates them.

When the two widths differ (768 for DINOv2, 384 for a small text JEPA), the map has orthonormal rows; we keep
calling it the rotation. Pairs are needed only to fit it (paper, Prop. 3); unpaired data can refine it.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def procrustes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Orthogonal ``R`` minimizing ``||a @ R - b||_F`` over paired rows.

    If the widths differ, ``R`` has orthonormal rows or columns and maximizes ``tr(R^T a^T b)``, which is the
    least-squares fit when ``a`` is isotropic (``a^T a`` proportional to the identity).
    """
    u, _, vt = torch.linalg.svd(a.T @ b, full_matrices=False)
    return u @ vt


def whitener(x: torch.Tensor, shrink: float = 0.1) -> torch.Tensor:
    """Inverse square root of the covariance of ``x``, shrunk toward isotropy (few samples per dimension)."""
    c = torch.cov(x.T)
    c = (1 - shrink) * c + shrink * c.diagonal().mean() * torch.eye(len(c), dtype=c.dtype, device=c.device)
    evals, evecs = torch.linalg.eigh(c)
    return evecs @ torch.diag(evals.rsqrt()) @ evecs.T


def _sinkhorn_plan(a, b, reg):
    """Entropic optimal-transport plan between the rows of ``a`` and ``b`` (uniform weights, cosine cost)."""
    import ot

    w = np.full(len(a), 1 / len(a))
    cost = torch.cdist(F.normalize(a, dim=1), F.normalize(b, dim=1)).pow(2).numpy()
    return torch.as_tensor(ot.sinkhorn(w, w, cost, reg=reg, numItermax=10000), dtype=a.dtype)


def project(z: torch.Tensor, memory: torch.Tensor, mu: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
    """Move each row of ``z`` onto the cloud of ``memory`` rows: their softmax(cosine / ``tau``)-weighted mean.

    With ``memory`` the text-JEPA embeddings of a text corpus (no images), this replaces a rotated image embedding
    by a nearby mixture of real text embeddings, which a text-only decoder knows how to decode (DeCap, Li et al.
    2023). Cosines are taken after centring by ``mu``.
    """
    m = memory - mu
    w = (F.normalize(z - mu, dim=1) @ F.normalize(m, dim=1).T / tau).softmax(1)
    return w @ m + mu


class Rotation:
    """The map ``x -> (x - mu_a) @ R + mu_b`` from space A to space B.

    Args:
        R: ``[d_a, d_b]`` matrix with orthonormal rows or columns.
        mu_a, mu_b: means of the two spaces (estimated without pairs).
    """

    def __init__(self, R: torch.Tensor, mu_a: torch.Tensor, mu_b: torch.Tensor):
        self.R, self.mu_a, self.mu_b = R, mu_a, mu_b

    @classmethod
    def fit(cls, a, b, mu_a=None, mu_b=None) -> "Rotation":
        """Procrustes on paired rows of ``a`` and ``b``; means default to those of the pairs themselves."""
        mu_a = a.mean(0) if mu_a is None else mu_a
        mu_b = b.mean(0) if mu_b is None else mu_b
        return cls(procrustes(a - mu_a, b - mu_b), mu_a, mu_b)

    @classmethod
    def fit_pooled(cls, sets) -> "Rotation":
        """One Procrustes fit on several paired sets ``[(a, b), ...]``, each centred by its own means.

        The map then centres by the means of all rows pooled, e.g. Flickr photos and web images for a demo that
        captions images of every kind.
        """
        a = torch.cat([x - x.mean(0) for x, _ in sets])
        b = torch.cat([y - y.mean(0) for _, y in sets])
        mu_a, mu_b = torch.cat([x for x, _ in sets]).mean(0), torch.cat([y for _, y in sets]).mean(0)
        return cls(procrustes(a, b), mu_a, mu_b)

    def refine(self, a, b, rounds: int = 3, reg: float = 0.02) -> "Rotation":
        """Wasserstein-Procrustes on *unpaired* rows: match the mapped ``a`` to ``b`` by optimal transport, refit.

        Refining a rotation fitted on few pairs fills in what the pairs leave undetermined (paper, Sec. 5).
        """
        a, b, R = a - self.mu_a, b - self.mu_b, self.R
        for _ in range(rounds):
            R = procrustes(a, len(a) * _sinkhorn_plan(a @ R, b, reg) @ b)
        return Rotation(R, self.mu_a, self.mu_b)

    @classmethod
    def unpaired(cls, a, b, eps: float = 5e-3, rounds: int = 10) -> "Rotation":
        """Zero pairs: match the two clouds by their internal geometry (entropic Gromov-Wasserstein), then refine.

        Near-isotropic embeddings barely constrain the rotation (paper, Prop. 3), and this stays at chance.
        """
        import ot

        mu_a, mu_b = a.mean(0), b.mean(0)
        dist = lambda x: (1 - F.normalize(x, dim=1) @ F.normalize(x, dim=1).T).numpy()  # noqa: E731
        w = np.full(len(a), 1 / len(a))
        plan = ot.gromov.entropic_gromov_wasserstein(
            dist(a - mu_a), dist(b - mu_b), w, w, "square_loss", epsilon=eps, max_iter=500
        )
        R = procrustes(a - mu_a, len(a) * torch.as_tensor(plan, dtype=a.dtype) @ (b - mu_b))
        return cls(R, mu_a, mu_b).refine(a, b, rounds)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu_a) @ self.R + self.mu_b

    def save(self, path: str | Path) -> None:
        torch.save({"R": self.R, "mu_a": self.mu_a, "mu_b": self.mu_b}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Rotation":
        return cls(**torch.load(path))
