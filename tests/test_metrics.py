"""Alignment, captioning and text-similarity scores."""

import importlib.util
import math

import pytest
import torch

from prh.metrics import (
    alignment,
    caption_scores,
    consensus,
    matched_noise,
    mutual_knn,
    prh_score,
    prh_subsample,
    recall,
    rotation_error,
    scene_explained,
    sts_spearman,
    subsample_interval,
    unbiased_cka,
    whiten_top,
)


def test_rotated_copy_is_perfectly_aligned_and_independent_data_is_at_chance():
    a = torch.randn(2000, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    is_test = torch.arange(2000) >= 1000
    rotated = alignment(a, a @ q + 1.0, is_test)
    assert rotated["R@1"] == 1.0 and rotated["mknn"] == 1.0
    independent = alignment(a, torch.randn(2000, 32), is_test)
    assert independent["R@1"] < 0.01 and independent["mknn"] < 0.05


def test_scores_ignore_scale():
    a = torch.randn(300, 16)
    assert recall(a, 5 * a) == [1.0, 1.0] and mutual_knn(a, 5 * a) == 1.0


def test_whitened_retrieval_recovers_a_linear_map_that_a_rotation_cannot():
    a = torch.randn(2000, 32, dtype=torch.float64)
    A = torch.randn(32, 32, dtype=torch.float64) @ torch.diag(torch.logspace(-2, 1, 32, dtype=torch.float64))
    m = alignment(a, a @ A, torch.arange(2000) >= 1000)
    assert m["wR@1"] > 0.9 and m["wR@1"] > m["R@1"]


def test_recall_counts_ties_against_the_true_pair():
    r1, r5 = recall(torch.randn(100, 8), torch.ones(100, 8))  # a collapsed encoder scores at chance
    assert r1 < 0.02 and r5 < 0.1


def test_mutual_knn_counts_shared_neighbours():
    a = torch.randn(200, 4)
    b = a.clone()
    b[100:] = torch.randn(100, 4)  # half the rows keep their neighbourhoods
    assert 0.2 < mutual_knn(a, b) < 0.75


def test_prh_score_picks_the_best_layer_pair_and_ignores_outlier_scale():
    x = torch.randn(300, 16)
    layers_a = [torch.randn(300, 16), x, torch.randn(300, 16)]
    spiky = x.clone()
    spiky[0, 0] = 1e6  # clipped at the 0.95 quantile
    score, i, j = prh_score(layers_a, [torch.randn(300, 16), spiky])
    assert (i, j) == (1, 1) and score > 0.9


def test_test_rows_from_another_dataset_are_centred_by_their_own_means():
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    fit, other = torch.randn(1000, 16), torch.randn(500, 16) + 4  # the test pairs come from a shifted dataset
    a, b = torch.cat([fit, other]), torch.cat([fit, other]) @ q + torch.cat([torch.zeros(1000, 16), torch.ones(500, 16)])
    is_test = torch.arange(1500) >= 1000
    assert alignment(a, b, is_test, test_means=True)["R@1"] == 1.0
    assert alignment(a, b, is_test)["R@1"] < 1.0


def test_prh_subsample_is_paired_and_centred_on_the_full_score():
    x = torch.randn(400, 8)
    layers_a, layers_b = [x], [x + 0.5 * torch.randn(400, 8)]
    s1, s2 = prh_subsample(layers_a, layers_b, n=20, size=300), prh_subsample(layers_a, layers_b, n=20, size=300)
    assert s1.shape == (20,) and torch.equal(s1, s2)  # same seed, same subsets: paired across models
    assert 0 < s1.mean() < 1 and s1.std() > 0


def test_matched_noise_matches_the_cosine():
    for c in (0.2, 0.33, 0.6):
        assert abs((1 + matched_noise(c) ** 2) ** -0.5 - c) < 1e-12
    assert abs(matched_noise(0.33) - 2.86) < 0.01  # the paper's value


def test_sts_spearman_is_one_for_monotone_similarity():
    a = torch.randn(50, 8)
    t = torch.linspace(0, 1, 50)[:, None]
    b = (1 - t) * a + t * torch.randn(50, 8)
    assert sts_spearman(a, b, -t.squeeze().numpy()) > 0.8


@pytest.mark.skipif(importlib.util.find_spec("pycocoevalcap") is None, reason="pycocoevalcap is not installed")
def test_caption_scores_prefer_the_reference():
    refs = [["a dog runs on the beach", "a puppy plays by the sea"], ["a man rides a red bike", "a cyclist on a road"]] * 5
    good = caption_scores(["a dog runs on the beach", "a man rides a red bike"] * 5, refs)
    bad = caption_scores(["a bowl of soup on a table", "two cats sleep inside"] * 5, refs)
    assert good["CIDEr"] > bad["CIDEr"] and good["BLEU-4"] > bad["BLEU-4"]


def test_subsample_interval_recovers_the_standard_error_of_a_mean():
    x = torch.randn(1000, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    subsets = torch.stack([x[torch.randperm(1000, generator=g)[:800]].mean() for _ in range(2000)])
    lo, hi = subsample_interval(x.mean().item(), subsets, 1000, 800)
    assert abs((hi - lo) / (2 * 1.96) - 1000**-0.5) < 0.1 * 1000**-0.5  # the full-sample standard error, 1/sqrt(n)


def test_consensus_keeps_the_caption_the_others_agree_with():
    cands = [
        "Two cats sleep on a sofa.",
        "a dog runs on the beach",
        "A dog running on a beach!",
        "a dog on the beach",
        "a red car",
    ]
    assert consensus(cands) == "a dog on the beach"
    assert consensus(["only one"]) == "only one" and consensus(["same", "same"]) == "same"


def test_unbiased_cka_matches_the_u_statistic_and_its_invariances():
    import itertools

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(7, 3, generator=g, dtype=torch.float64), torch.randn(7, 4, generator=g, dtype=torch.float64)

    def hsic_u(K, L):  # Song et al. (2012), Thm. 2: average over ordered 4-tuples of distinct rows
        m, total = len(K), 0.0
        for t in itertools.permutations(range(m), 4):
            h = 0.0
            for i, j, q, r in itertools.permutations(t):
                h += K[i, j] * L[i, j] + K[i, j] * L[q, r] - 2 * K[i, j] * L[i, q]
            total += h / 24
        return total / math.perm(m, 4)

    K, L = a @ a.T, b @ b.T
    reference = hsic_u(K, L) / (hsic_u(K, K) * hsic_u(L, L)) ** 0.5
    assert unbiased_cka(a, b) == pytest.approx(reference, abs=1e-10)
    x = torch.randn(300, 16, generator=g)
    q = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
    assert unbiased_cka(x, 3.0 * x @ q) == pytest.approx(1.0, abs=1e-6)
    assert abs(unbiased_cka(x, torch.randn(300, 16, generator=g))) < 0.05
    layers = [torch.randn(300, 16, generator=g), x]
    assert prh_score(layers, [x @ q], metric="cka")[1:] == (1, 0)


def test_scene_explained_fraction_and_its_rotation_error_bound():
    """Two whitened directions carry the scene (shared by an image's five captions), two the caption's own nuisance."""
    g = torch.Generator().manual_seed(0)
    scene = torch.randn(4000, 1, 2, generator=g, dtype=torch.float64).expand(4000, 5, 2)
    f = torch.cat([scene, torch.randn(4000, 5, 2, generator=g, dtype=torch.float64)], -1)
    assert scene_explained(f) == pytest.approx(0.5, abs=0.02)  # tr C_f^(1/2) / k = 2 / 4
    a, b = whiten_top(f[:, 0], 4), whiten_top(f[:, 1], 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.float64))
    assert rotation_error(a, a @ q) == pytest.approx(0, abs=1e-9)
    assert rotation_error(a, b) == pytest.approx(2 * 4 * (1 - 0.5), abs=0.15)  # Thm. 3's bound, attained here
