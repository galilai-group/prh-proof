"""The rotation between two embedding spaces: fitting, unpaired refinement, persistence."""

import torch

from prh.align import Rotation, procrustes, whitener
from prh.metrics import recall


def _rotation(d_a, d_b, dtype=torch.float64):
    return torch.linalg.qr(torch.randn(d_a, d_b, dtype=dtype))[0]


def test_procrustes_recovers_a_rotation():
    a, q = torch.randn(500, 64, dtype=torch.float64), _rotation(64, 64)
    assert torch.allclose(procrustes(a, a @ q), q, atol=1e-8)


def test_procrustes_with_different_widths_returns_orthonormal_columns_and_recovers_the_map():
    a = torch.linalg.qr(torch.randn(2000, 96, dtype=torch.float64))[0]  # isotropic, as SIGReg targets
    q = _rotation(96, 32)  # 96 -> 32, orthonormal columns
    r = procrustes(a, a @ q)
    assert r.shape == (96, 32) and torch.allclose(r.T @ r, torch.eye(32, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(r, q, atol=1e-8)


def test_rotation_fit_handles_means_and_round_trips(tmp_path):
    a, q = torch.randn(300, 16, dtype=torch.float64), _rotation(16, 16)
    b = (a - 2) @ q + 7
    rot = Rotation.fit(a, b)
    assert torch.allclose(rot(a), b, atol=1e-8)
    rot.save(tmp_path / "rotation.pt")
    assert torch.allclose(Rotation.load(tmp_path / "rotation.pt")(a), b, atol=1e-8)


def test_unpaired_refinement_improves_a_few_pair_rotation():
    """Few pairs leave the map underdetermined; refining on unpaired clouds fills in the rest (paper, Sec. 5)."""
    torch.manual_seed(0)
    centers = torch.randn(200, 24, dtype=torch.float64) * 3  # clustered clouds, as real embeddings are
    a = centers[torch.randint(200, (1000,))] + 0.3 * torch.randn(1000, 24, dtype=torch.float64)
    q = _rotation(24, 24)
    b = a @ q + 0.05 * torch.randn(1000, 24, dtype=torch.float64)
    few = Rotation.fit(a[:12], b[:12], a.mean(0), b.mean(0))
    refined = few.refine(a, b[torch.randperm(1000)], rounds=5)  # unpaired: the texts are shuffled
    assert (few.R - q).norm() > 1 and (refined.R - q).norm() < 0.1  # 12 pairs < 24 dimensions: underdetermined
    assert recall(few(a), b)[0] < 0.5 and recall(refined(a), b)[0] > 0.95


def test_whitener_whitens():
    x = torch.randn(5000, 8, dtype=torch.float64) @ torch.randn(8, 8, dtype=torch.float64)
    w = whitener(x, shrink=0.0)
    assert torch.allclose(torch.cov((x @ w).T), torch.eye(8, dtype=torch.float64), atol=1e-6)


def test_project_moves_points_onto_the_memory_cloud():
    from prh.align import project

    memory = torch.eye(3, dtype=torch.float64) * 5
    mu = torch.zeros(3, dtype=torch.float64)
    near = memory[1:2] + 0.1 * torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(project(near, memory, mu, tau=1e-3), memory[1:2], atol=1e-6)
    between = project(torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64), memory, mu, tau=1e-3)
    assert torch.allclose(between, torch.tensor([[2.5, 2.5, 0.0]], dtype=torch.float64), atol=1e-6)


def test_fit_pooled_recovers_one_rotation_from_differently_centred_sets():
    q = _rotation(6, 6)
    a1 = torch.randn(200, 6, dtype=torch.float64) + 5  # two domains with different means, one shared rotation
    a2 = torch.randn(300, 6, dtype=torch.float64) - 3
    rot = Rotation.fit_pooled([(a1, (a1 - a1.mean(0)) @ q + 1), (a2, (a2 - a2.mean(0)) @ q - 2)])
    assert torch.allclose(rot.R, q, atol=1e-8)
    assert torch.allclose(rot.mu_a, torch.cat([a1, a2]).mean(0))
