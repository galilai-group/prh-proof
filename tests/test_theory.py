"""The paper's finite-world statements (Sec. 2 and 3), checked numerically on random and adversarial worlds."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prh import theory as th


def random_rotation(k, rng):
    q, r = np.linalg.qr(rng.standard_normal((k, k)))
    return q * np.sign(np.diag(r))


def test_with_marginal_gives_a_symmetric_positive_pair_distribution():
    pi, p = th.random_world(30, np.random.default_rng(0))
    assert np.allclose(pi, pi.T) and (pi >= 0).all()
    assert np.allclose(th.marginal(pi), p, atol=1e-12) and np.isclose(pi.sum(), 1)


def test_spectrum_is_that_of_the_markov_operator_on_l2p():
    pi, p = th.random_world(25, np.random.default_rng(1))
    mu, phi = th.spectrum(pi)
    T = pi / p[:, None]  # (T h)(z) = E[h(z') | z]
    assert np.isclose(mu[0], 1) and np.allclose(phi[:, 0], phi[0, 0]) and (np.diff(mu) <= 1e-12).all()
    assert np.allclose(phi.T @ (p[:, None] * phi), np.eye(25), atol=1e-10)  # orthonormal in L2(p)
    assert np.allclose(T @ phi, phi * mu, atol=1e-10)


def test_change_of_variables_lemma():
    """L_m(f) = L^pi(f o g_m): a JEPA on rendered observations solves the world problem."""
    rng = np.random.default_rng(2)
    pi, _ = th.random_world(20, rng)
    g = rng.choice(60, 20, replace=False)
    f = rng.standard_normal((60, 4))
    assert np.isclose(th.jepa_loss(f, th.render(pi, g, 60)), th.jepa_loss(f[g], pi))


@pytest.mark.parametrize("seed", range(5))
def test_prop1_spectral_solution_is_the_minimum_and_its_rotations_too(seed):
    rng = np.random.default_rng(seed)
    pi, p = th.random_world(40, rng)
    k = 1 + seed
    mu = th.spectrum(pi)[0]
    phi = th.spectral_solution(pi, k)
    assert np.allclose(p @ phi, 0, atol=1e-10) and np.allclose(phi.T @ (p[:, None] * phi), np.eye(k), atol=1e-10)
    best = 2 * (1 - mu[1 : k + 1]).sum()
    assert np.isclose(th.jepa_loss(phi, pi), best)
    assert np.isclose(th.jepa_loss(phi @ random_rotation(k, rng).T, pi), best)
    for _ in range(50):  # every other feasible embedding does worse
        assert th.jepa_loss(th.whiten(rng.standard_normal((40, k)), p), pi) >= best - 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_thm1_two_modalities_with_the_same_positives_agree_up_to_rotation(seed):
    """Each modality solves its JEPA on its own observations; paired observations then agree up to one rotation."""
    rng = np.random.default_rng(seed)
    n, k = 30, 3
    pi, p = th.random_world(n, rng)
    g1, g2 = rng.choice(100, n, replace=False), rng.choice(500, n, replace=False)  # injective renderings
    s1, f1 = th.observed_solution(th.render(pi, g1, 100), k)
    s2, f2 = th.observed_solution(th.render(pi, g2, 500), k)
    h1, h2 = f1[np.searchsorted(s1, g1)], f2[np.searchsorted(s2, g2)]  # f_m(g_m(z)) for every world state z
    assert th.rotation_distance(h1, h2, p) < 1e-8
    assert np.allclose(h1, h2 @ th.best_rotation(h1, h2, p).T, atol=1e-8)


def test_thm1_needs_shared_positives():
    rng = np.random.default_rng(0)
    pi1, p = th.random_world(30, rng)
    pi2 = th.with_marginal(rng.random((30, 30)) + rng.random((30, 30)).T, p)
    assert th.rotation_distance(th.spectral_solution(pi1, 3), th.spectral_solution(pi2, 3), p) > 0.1


def test_thm2_davis_kahan_bound_holds():
    ratios = []
    for pi1, pi2, p, k in th.mismatched_kernels(300):
        mu = th.spectrum(pi2)[0]
        if mu[k] - mu[k + 1] < 1e-9:  # the bound needs a gap
            continue
        err = th.rotation_distance(th.spectral_solution(pi1, k), th.spectral_solution(pi2, k), p)
        bound = th.davis_kahan_bound(pi1, pi2, k)
        assert err <= bound + 1e-9, (err, bound)
        ratios.append(err / bound)
    assert len(ratios) > 250 and max(ratios) > 0.1  # covered many cases, and the bound is not vacuously loose


def test_thm2_kernel_mismatch_norms():
    rng = np.random.default_rng(3)
    pi1, p = th.random_world(15, rng)
    pi2 = th.with_marginal(rng.random((15, 15)) + rng.random((15, 15)).T, p)
    op, hs = th.kernel_mismatch(pi1, pi2)
    E = pi1 / p[:, None] - pi2 / p[:, None]  # T_pi1 - T_pi2 as a matrix acting on functions
    assert np.isclose(hs**2, ((pi1 - pi2) ** 2 / np.outer(p, p)).sum())
    h = rng.standard_normal((15, 200))
    norm = lambda x: np.sqrt(p @ x**2)  # noqa: E731
    assert (norm(E @ h) <= op * norm(h) + 1e-12).all() and op <= hs + 1e-12


def test_prop2_second_moments_do_not_identify_the_rotation():
    rng = np.random.default_rng(4)
    pi, p = th.random_world(30, rng)
    u = th.spectral_solution(pi, 4)
    Q, R = random_rotation(4, rng), random_rotation(4, rng)
    for A in (Q, R):
        assert np.allclose((u @ A.T).T @ (p[:, None] * (u @ A.T)), np.eye(4), atol=1e-10)


@pytest.mark.parametrize("k", [2, 5, 16])
def test_prop2_k_pairs_identify_the_rotation_and_fewer_do_not(k):
    rng = np.random.default_rng(k)
    Q = random_rotation(k, rng)
    u = rng.standard_normal((k, k))  # embeddings of k world states in general position
    ones = np.ones(k)
    assert np.allclose(th.best_rotation(u @ Q.T, u, ones), Q)
    v = np.linalg.svd(u[:-1])[2][-1]  # orthogonal to the first k - 1 embeddings
    R = Q @ (np.eye(k) - 2 * np.outer(v, v))  # another rotation that fits those k - 1 pairs exactly
    assert np.allclose(u[:-1] @ R.T, u[:-1] @ Q.T) and not np.allclose(R, Q)


def test_prop3_next_token_loss_is_prediction_plus_log_partition():
    torch.manual_seed(0)
    h, w, y = torch.randn(64, 16), torch.randn(100, 16), torch.randint(100, (64,))
    ce = F.cross_entropy(h @ w.T, y, reduction="none")
    assert torch.allclose(ce, -(h * w[y]).sum(1) + torch.logsumexp(h @ w.T, 1), atol=1e-5)
    tau = 1e-3  # zero temperature: the anti-collapse term becomes the max, and the loss a margin
    assert torch.allclose(tau * torch.logsumexp(h @ w.T / tau, 1), (h @ w.T).max(1).values, atol=1e-2)
    A = torch.randn(16, 16, dtype=torch.float64)  # GL(k) symmetry: h -> A h, w -> A^-T w
    h, w = h.double(), w.double()
    assert torch.allclose((h @ A.T) @ (w @ torch.linalg.inv(A)).T, h @ w.T, atol=1e-8)


def _nuisance_world(seed, n=10, m1=4, m2=3):
    rng = np.random.default_rng(seed)
    pi, p = th.random_world(n, rng)
    q1, q2 = rng.dirichlet(np.ones(m1)), rng.dirichlet(np.ones(m2))
    return rng, pi, p, q1, q2


def _lazy(pi_or_q, rng, keep):
    """Row-stochastic corruption with the given stationary distribution: keep with prob ``keep``, else move."""
    if pi_or_q.ndim == 2:  # world states: move along the world's own positive pairs
        p = th.marginal(pi_or_q)
        return keep * np.eye(len(p)) + (1 - keep) * pi_or_q / p[:, None]
    q = pi_or_q
    K = rng.random((len(q), len(q)))
    return keep * np.eye(len(q)) + (1 - keep) * th.with_marginal(K + K.T, q) / q[:, None]


@pytest.mark.parametrize("seed", range(3))
def test_thm3_nuisance_bound_holds_for_any_two_encoders(seed):
    rng, pi, p, q1, q2 = _nuisance_world(seed)
    for _ in range(20):
        f = th.whiten(rng.normal(size=(len(p) * len(q1), 3)), np.kron(p, q1))
        g = th.whiten(rng.normal(size=(len(p) * len(q2), 3)), np.kron(p, q2))
        bound = max(th.nuisance_bound(f, p, q1), th.nuisance_bound(g, p, q2))
        assert th.cross_modal_error(f, g, p, q1, q2) >= bound - 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_thm3_good_views_ignore_nuisance_and_align_exactly(seed):
    _, pi, p, q1, q2 = _nuisance_world(seed)
    f, g = (th.spectral_solution(th.good_views(pi, q), 3) for q in (q1, q2))
    assert th.nuisance_bound(f, p, q1) < 1e-8 and th.cross_modal_error(f, g, p, q1, q2) < 1e-8


@pytest.mark.parametrize("seed", range(3))
def test_thm3_bad_views_spend_dimensions_on_nuisance(seed):
    rng, pi, p, q1, q2 = _nuisance_world(seed)
    k, A = 4, _lazy(pi, rng, keep=0.2)  # corruptions move the state a lot ...
    B = _lazy(q1, rng, keep=0.9)  # ... and keep the nuisance: spans of one caption keep its word choice
    f = th.spectral_solution(th.bad_views(p, q1, A, B), k)
    r = th.nuisance_directions(A.T @ (p[:, None] * A), B.T @ (q1[:, None] * B), k)
    c = np.sort(np.linalg.eigvalsh(th.state_explained_covariance(f, p, q1)))
    assert 1 <= r < k and np.allclose(c, [0] * r + [1] * (k - r), atol=1e-8)
    g = th.spectral_solution(th.good_views(pi, q2), k)
    assert th.cross_modal_error(f, g, p, q1, q2) >= 2 * r - 1e-8
    # a corruption that forgets the nuisance (B = 1 q^T) loses no direction
    f0 = th.spectral_solution(th.bad_views(p, q1, A, np.tile(q1, (len(q1), 1))), k)
    assert th.nuisance_bound(f0, p, q1) < 1e-8


def test_thm4_recovers_prop3_for_product_corruptions():
    rng, pi, p, q1, _ = _nuisance_world(0)
    k, A, B = 3, _lazy(pi, rng, keep=0.6), _lazy(q1, rng, keep=0.2)  # the nuisance is forgotten faster than the state
    W, state = th.bad_views(p, q1, A, B), np.repeat(np.arange(len(p)), len(q1))
    eta, op, hs = th.coverage(W, state, len(p))
    alpha, beta = th.spectrum(A.T @ (p[:, None] * A))[0], th.spectrum(B.T @ (q1[:, None] * B))[0]
    assert op < 1e-12 and hs < 1e-12 and np.isclose(eta, beta[1]) and alpha[k] > max(alpha[k + 1], beta[1])
    assert (
        np.allclose(th.world_kernel(W, state, len(p)), A.T @ (p[:, None] * A)) and th.coverage_bound(W, state, len(p), k) < 1e-10
    )
    world = th.spectral_solution(th.world_kernel(W, state, len(p)), k)[state]
    assert th.rotation_distance(th.spectral_solution(W, k), world, W.sum(1)) < 1e-8


@pytest.mark.parametrize("seed", range(3))
def test_thm4_bounds_hold_and_are_informative(seed):
    rng, informative = np.random.default_rng(seed), [0, 0]
    for _ in range(2000):  # until both bounds have been informative (below the trivial ceiling) ten times
        if min(informative) >= 10:
            break
        n, k = rng.integers(6, 14), rng.integers(1, 4)
        pi, p = th.random_world(n, rng)
        mu = th.spectrum(pi)[0]
        if mu[k] - max(mu[k + 1], 0) < 0.05:
            continue
        mods = []
        for _ in range(2):  # two modalities of one world, rendered and viewed differently
            state, rho = th.random_rendering(rng, p)
            W = th.mixed_views(rng, pi, state, rho, rng.uniform(0, 0.3), 10 ** rng.uniform(-4, -1))
            f, tau = th.spectral_solution(W, k), th.coverage_bound(W, state, n, k)
            world = th.spectral_solution(th.world_kernel(W, state, n), k)[state]
            assert th.rotation_distance(f, world, rho) <= tau + 1e-9  # (i)
            P = np.eye(n)[state].T * rho / p[:, None]  # conditional expectation given the state
            assert k - np.trace((P @ f).T @ (p[:, None] * (P @ f))) <= tau**2 + 1e-9  # tr C_f >= k - tau^2
            informative[0] += tau < np.sqrt(2 * k)
            mods.append((W, state, rho, f, tau))
        (W1, s1, r1, f1, t1), (W2, s2, r2, f2, t2) = mods
        pw1, pw2 = th.world_kernel(W1, s1, n), th.world_kernel(W2, s2, n)
        rhs = t1 + t2 + th.davis_kahan_bound(pw1, pw2, k)
        assert th.paired_distance(f1, f2, s1, s2, r1, r2, n) <= rhs + 1e-9  # (ii)
        informative[1] += rhs < np.sqrt(2 * k)
    assert min(informative) >= 10


def test_thm4_gap_condition_is_prop3s_condition_for_no_nuisance_direction():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n, m, k = rng.integers(4, 9), rng.integers(2, 5), rng.integers(1, 4)
        pi, p = th.random_world(n, rng)
        q = rng.dirichlet(np.ones(m))
        A, B = _lazy(pi, rng, keep=rng.uniform(0, 1)), _lazy(q, rng, keep=rng.uniform(0, 1))
        Ap, Bp = A.T @ (p[:, None] * A), B.T @ (q[:, None] * B)
        lam = sorted((a * b for a in th.spectrum(Ap)[0] for b in th.spectrum(Bp)[0]), reverse=True)[1:] + [-np.inf]
        separated_without_nuisance = lam[k - 1] > lam[k] + 1e-9 and th.nuisance_directions(Ap, Bp, k) == 0
        tau = th.coverage_bound(th.bad_views(p, q, A, B), np.repeat(np.arange(n), m), n, k)
        assert np.isfinite(tau) == separated_without_nuisance


def test_thm4_perfect_coverage_gives_good_views():
    rng = np.random.default_rng(1)
    pi, p = th.random_world(9, rng)
    fs = []
    for _ in range(2):  # positives are fresh observations of pi-related states: no nuisance survives
        state, rho = th.random_rendering(rng, p)
        W = pi[np.ix_(state, state)] * (rho / p[state])[:, None] * (rho / p[state])[None, :]
        eta, op, _ = th.coverage(W, state, 9)
        assert eta <= 1e-12 and op < 1e-12 and np.allclose(th.world_kernel(W, state, 9), pi)
        fs.append((th.spectral_solution(W, 3), state, rho))
    (f1, s1, r1), (f2, s2, r2) = fs
    assert th.paired_distance(f1, f2, s1, s2, r1, r2, 9) < 1e-6


def test_thm4_illustration_good_views_align_and_weak_views_need_samples():
    assert th.line_alignment(1, 0.5, "good", 0) > 1 - 1e-9  # good views: exact at one sample per state
    few, many = th.line_alignment(1, 0.5, 0.005, 0), th.line_alignment(64, 0.5, 0.005, 0)
    assert few < 0.5 < 0.9 < many  # weak bad views need many samples per state (Thm. 4: coverage)
