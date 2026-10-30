"""The finite-world objects of the paper's Section 2, executable (NumPy, float64).

A world is a finite set of states ``z = 0..n-1`` with distribution ``p``. A positive-pair distribution is a
symmetric ``[n, n]`` matrix ``pi`` with both marginals ``p``. Its Markov operator ``T`` on ``L2(p)`` is unitarily
equivalent to the symmetric matrix ``S = D^-1/2 pi D^-1/2`` (``D = diag(p)``), which is what we diagonalize:
an eigenvector ``u`` of ``S`` is the eigenfunction ``phi = D^-1/2 u`` of ``T``. Functions ``h: Z -> R^k`` are
``[n, k]`` arrays, one row per world state. ``tests/test_theory.py`` checks every statement numerically.
"""

import numpy as np


def marginal(pi: np.ndarray) -> np.ndarray:
    """``p(z) = sum_z' pi(z, z')``."""
    return pi.sum(1)


def with_marginal(K: np.ndarray, p: np.ndarray, iters: int = 10_000, tol: float = 1e-13) -> np.ndarray:
    """Symmetric positive-pair distribution ``diag(d) K diag(d)`` with marginal ``p`` (symmetric Sinkhorn).

    Args:
        K: symmetric matrix with positive entries (any affinity between world states).
        p: target marginal, positive, summing to one.
        iters: maximum number of Sinkhorn iterations.
        tol: tolerance on the marginal.
    """
    d = np.sqrt(p / K.sum(1))
    for _ in range(iters):
        d = np.sqrt(d * p / (K @ d))
        if np.abs(d * (K @ d) - p).max() < tol:
            break
    return d[:, None] * K * d[None, :]


def spectrum(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues ``mu_0 = 1 >= mu_1 >= ...`` of ``T_pi`` and eigenfunctions as columns, orthonormal in ``L2(p)``."""
    p = marginal(pi)
    S = pi / np.sqrt(np.outer(p, p))
    mu, U = np.linalg.eigh(S)
    order = np.argsort(mu)[::-1]
    return mu[order], U[:, order] / np.sqrt(p)[:, None]


def spectral_solution(pi: np.ndarray, k: int) -> np.ndarray:
    """``Phi_k = (phi_1, ..., phi_k)``, the JEPA minimizer of Prop. 1 (unique up to rotation if ``mu_k > mu_k+1``).

    The top ``k`` eigenfunctions on ``1^perp``, also when the eigenvalue 1 is not simple (disconnected kernels).
    """
    p = marginal(pi)
    u = np.sqrt(p)
    B = np.linalg.svd(u[:, None], full_matrices=True)[0][:, 1:]  # orthonormal basis of the centred functions
    mu, V = np.linalg.eigh(B.T @ (pi / np.outer(u, u)) @ B)
    return B @ V[:, np.argsort(mu)[::-1][:k]] / u[:, None]


def jepa_loss(h: np.ndarray, pi: np.ndarray) -> float:
    """Prediction term ``E_pi ||h(z) - h(z')||^2`` of the squared-loss JEPA."""
    diff = h[:, None, :] - h[None, :, :]
    return float((pi * (diff**2).sum(-1)).sum())


def whiten(h: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Centre and whiten ``h`` in ``L2(p)``: the result satisfies Prop. 1's constraint ``E h = 0, E h h^T = I``."""
    h = h - p @ h
    evals, evecs = np.linalg.eigh(h.T @ (p[:, None] * h))
    return h @ evecs @ np.diag(evals**-0.5) @ evecs.T


def best_rotation(h1: np.ndarray, h2: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Orthogonal ``Q`` minimizing ``E_p ||h1(z) - Q h2(z)||^2`` (weighted Procrustes); ``h1 ~ h2 @ Q.T``."""
    u, _, vt = np.linalg.svd(h1.T @ (p[:, None] * h2))
    return u @ vt


def rotation_distance(h1: np.ndarray, h2: np.ndarray, p: np.ndarray) -> float:
    """``min_Q (E_p ||h1(z) - Q h2(z)||^2)^1/2``, the left side of Thm. 2."""
    Q = best_rotation(h1, h2, p)
    return float(np.sqrt(p @ ((h1 - h2 @ Q.T) ** 2).sum(1)))


def kernel_mismatch(pi1: np.ndarray, pi2: np.ndarray) -> tuple[float, float]:
    """Operator and Hilbert-Schmidt norms of ``E = T_pi1 - T_pi2`` on ``L2(p)`` (common marginal ``p``)."""
    p = marginal(pi1)
    E = (pi1 - pi2) / np.sqrt(np.outer(p, p))
    return float(np.linalg.norm(E, 2)), float(np.linalg.norm(E, "fro"))


def davis_kahan_bound(pi1: np.ndarray, pi2: np.ndarray, k: int) -> float:
    """Right side of Thm. 2: ``2 sqrt(2) / delta * min(sqrt(k) ||E||_op, ||E||_HS)``, ``delta`` the gap of ``T_pi2``."""
    mu = np.append(spectrum(pi2)[0], -np.inf)  # mu_|Z| = -inf: for k = |Z| - 1 the bound is 0
    op, hs = kernel_mismatch(pi1, pi2)
    return 2 * np.sqrt(2) / (mu[k] - mu[k + 1]) * min(np.sqrt(k) * op, hs)


def render(pi: np.ndarray, g: np.ndarray, size: int) -> np.ndarray:
    """Positive pairs of a modality that renders world state ``z`` as observation ``g[z]`` (``g`` injective).

    Returns the ``[size, size]`` distribution of observation pairs ``(g(z), g(z'))``; observations that no world
    state renders have probability zero.
    """
    out = np.zeros((size, size))
    out[np.ix_(g, g)] = pi
    return out


def observed_solution(pi_x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral JEPA solution of a modality from its observation pairs alone, on the observed support.

    Returns:
        ``(support, f)``: the observations with positive probability and the encoder's value on each.
    """
    support = np.flatnonzero(pi_x.sum(1) > 0)
    return support, spectral_solution(pi_x[np.ix_(support, support)], k)


def random_world(n: int, rng: np.random.Generator, blocks: int = 0, leak: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Random positive-pair distribution ``pi`` on ``n`` world states, with a random marginal ``p``.

    Args:
        n: number of world states.
        rng: NumPy random generator.
        blocks: if positive, the states form this many clusters linked with weight ``leak``.
        leak: weight of the links between clusters; small values give small spectral gaps, the hard case of Thm. 2.

    Returns:
        ``(pi, p)``.
    """
    p = rng.dirichlet(np.ones(n))
    K = rng.random((n, n))
    K = K + K.T
    if blocks:
        label = rng.integers(blocks, size=n)
        K = np.where(label[:, None] == label[None, :], K, leak * K)
    return with_marginal(K, p), p


def mismatched_kernels(cases: int, seed: int = 0):
    """Pairs of kernels with a common marginal, from a tiny perturbation to an unrelated kernel.

    Every other world is clustered (small spectral gaps). Yields ``(pi1, pi2, p, k)``.
    """
    rng = np.random.default_rng(seed)
    for case in range(cases):
        n, k = rng.integers(8, 40), rng.integers(1, 6)
        blocks, leak = (rng.integers(2, 6), 10.0 ** -rng.uniform(0, 3)) if case % 2 else (0, 1.0)
        pi2, p = random_world(n, rng, blocks, leak)
        K = rng.random((n, n))
        t = 10.0 ** -rng.uniform(0, 4)
        yield (1 - t) * pi2 + t * with_marginal(K + K.T, p), pi2, p, int(k)


# -- Thm. 3: observations are (state, nuisance) pairs -------------------------------------------------------
# A modality observes world state z together with a nuisance nu ~ q of its own (word choice, viewpoint),
# independent of z and of the other modality's nuisance. Observation (z, nu) is index z * len(q) + nu, with
# distribution p(z) q(nu); the rendering (z, nu) -> observation is injective, so a state has many observations.


def good_views(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Positive pairs of two observations of related states, each with its own nuisance: ``pi (x) q q^T``."""
    return np.kron(pi, np.outer(q, q))


def bad_views(p: np.ndarray, q: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Positive pairs of two corruptions of one observation ``(z, nu)``, each drawn from ``C = A (x) B``.

    ``A`` ``[n, n]`` and ``B`` ``[m, m]`` are row-stochastic with stationary distributions ``p`` and ``q``. The two
    corruptions share the observation, hence its nuisance: ``pi = C^T diag(p (x) q) C``.
    """
    C = np.kron(A, B)
    return C.T @ (np.kron(p, q)[:, None] * C)


def state_explained_covariance(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``Cov(E[f(z, nu) | z])`` of an encoder ``f`` on observations; for whitened ``f`` its eigenvalues lie in [0, 1]."""
    u = (f.reshape(len(p), len(q), -1) * q[None, :, None]).sum(1)
    u = u - p @ u
    return u.T @ (p[:, None] * u)


def nuisance_bound(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Thm. 3: ``2 (k - tr C^1/2)``, a lower bound on :func:`cross_modal_error` against any other modality."""
    c = np.clip(np.linalg.eigvalsh(state_explained_covariance(f, p, q)), 0, None)
    return float(2 * (f.shape[1] - np.sqrt(c).sum()))


def cross_modal_error(f: np.ndarray, g: np.ndarray, p: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """``min_Q E ||Q f(z, nu1) - g(z, nu2)||^2`` over paired observations of one state, nuisances independent.

    ``f`` and ``g`` are whitened on their observations (:func:`whiten` with weights ``p (x) q``).
    """
    u = (f.reshape(len(p), len(q1), -1) * q1[None, :, None]).sum(1)
    v = (g.reshape(len(p), len(q2), -1) * q2[None, :, None]).sum(1)
    sigma = u.T @ (p[:, None] * v)  # E[f g^T]: given z, the two observations are independent
    return float(f.shape[1] + g.shape[1] - 2 * np.linalg.svd(sigma, compute_uv=False).sum())


def nuisance_directions(A_pairs: np.ndarray, B_pairs: np.ndarray, k: int) -> int:
    """``r`` of Thm. 3: how many of the top-``k`` non-trivial eigenvalues ``alpha_i beta_j`` of a bad kernel have ``j >= 1``.

    ``A_pairs`` and ``B_pairs`` are the pair distributions ``A^T diag(p) A`` and ``B^T diag(q) B``.
    """
    alpha, beta = spectrum(A_pairs)[0], spectrum(B_pairs)[0]
    products = sorted(((a * b, j) for a in alpha for j, b in enumerate(beta)), reverse=True)[1 : k + 1]
    return sum(j >= 1 for _, j in products)


# -- Thm. 4: any positive pairs on observations, through the world kernel they induce ------------------------
# Observations x = 0..N-1 of one modality, each of world state ``state[x]``; positive pairs are any symmetric
# distribution ``W`` on observation pairs (good views, corruptions of one observation, or a mix), whose marginal
# ``rho`` is the law of observations. In the product layout above, ``state = np.repeat(np.arange(n), len(q))``.


def world_kernel(W: np.ndarray, state: np.ndarray, n: int) -> np.ndarray:
    """``pi_W(z, z')``: the probability that a positive pair joins an observation of ``z`` to one of ``z'``."""
    M = np.eye(n)[state]
    return M.T @ W @ M


def coverage(W: np.ndarray, state: np.ndarray, n: int) -> tuple[float, float, float]:
    """``(eta, eps_op, eps_hs)`` of Thm. 4 for positive pairs ``W`` on observations.

    On ``L2(rho)``, split functions into those of the state alone and those with zero mean given the state (the
    nuisance). ``eta`` is the largest eigenvalue of the Markov operator of ``W`` on nuisance functions (how much
    of an observation's own nuisance its positive keeps; ``-inf`` if every state has one observation) and
    ``eps`` the operator and Hilbert-Schmidt norms of the block that maps state functions to nuisance ones.
    """
    rho = W.sum(1)
    p = np.bincount(state, rho, minlength=n)
    assert (rho > 0).all() and (p > 0).all(), "every observation and every state needs positive probability"
    S = W / np.sqrt(np.outer(rho, rho))
    Qs = np.eye(n)[state] * np.sqrt(rho / p[state])[:, None]  # orthonormal: normalized state indicators
    Qn = np.linalg.svd(Qs, full_matrices=True)[0][:, n:]  # orthonormal basis of the nuisance functions
    B = Qn.T @ S @ Qs
    eta = np.linalg.eigvalsh(Qn.T @ S @ Qn).max() if Qn.shape[1] else -np.inf
    return float(eta), float(np.linalg.norm(B, 2)) if B.size else 0.0, float(np.linalg.norm(B)) if B.size else 0.0


def coverage_bound(W: np.ndarray, state: np.ndarray, n: int, k: int, tol: float = 1e-9) -> float:
    """Thm. 4: ``2 sqrt(2) min(sqrt(k) eps_op, sqrt(2) eps_hs) / delta`` with ``delta = mu_k - max(mu_k+1, eta)``.

    Bounds ``min_O ||f - Phi_k(state) O||`` in ``L2(rho)`` for the spectral solution ``f`` of ``W``, ``Phi_k`` that of
    :func:`world_kernel`; ``inf`` when ``delta <= tol`` (a tie leaves ``Phi_k`` undetermined).
    """
    mu = np.append(spectrum(world_kernel(W, state, n))[0], -np.inf)
    eta, op, hs = coverage(W, state, n)
    delta = mu[k] - max(mu[k + 1], eta)
    return 2 * np.sqrt(2) * min(np.sqrt(k) * op, np.sqrt(2) * hs) / delta if delta > tol else np.inf


def paired_distance(f: np.ndarray, g: np.ndarray, state1, state2, rho1, rho2, n: int) -> float:
    """``min_Q (E ||Q f(x1) - g(x2)||^2)^1/2`` over paired observations ``x1, x2`` of one state, drawn independently.

    ``f`` and ``g`` are centred and whitened on their observations (laws ``rho1``, ``rho2``, common state marginal
    ``p``). Thm. 4 (ii) holds for any coupling of the two observations given the state; this is the independent one.
    """
    p = np.bincount(state1, rho1, minlength=n)
    assert np.allclose(p, np.bincount(state2, rho2, minlength=n)), "the two modalities must share the state marginal"
    u, v = (np.eye(n)[s].T @ (r[:, None] * h) / p[:, None] for h, s, r in ((f, state1, rho1), (g, state2, rho2)))
    sigma = u.T @ (p[:, None] * v)  # E[f g^T]: given the state, the two observations are independent
    return float(np.sqrt(max(f.shape[1] + g.shape[1] - 2 * np.linalg.svd(sigma, compute_uv=False).sum(), 0.0)))


def random_rendering(rng: np.random.Generator, p: np.ndarray, most: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """``(state, rho)``: 1 to ``most`` observations per state and a random law ``rho`` with state marginal ``p``."""
    state = np.repeat(np.arange(len(p)), rng.integers(1, most + 1, size=len(p)))
    rho = rng.random(len(state)) + 0.1
    return state, rho * p[state] / np.bincount(state, rho)[state]


def mixed_views(rng: np.random.Generator, pi: np.ndarray, state, rho, keep: float, move: float) -> np.ndarray:
    """Good views of ``pi``, plus ``keep`` of pairs of an observation with itself and ``move`` of random pairs."""
    p = marginal(pi)
    good = pi[np.ix_(state, state)] * np.outer(rho / p[state], rho / p[state])
    K = rng.random((len(rho), len(rho)))
    return (1 - keep - move) * good + keep * np.diag(rho) + move * with_marginal(K + K.T, rho)


def thm4_tightness(worlds: int = 3000, seed: int = 0) -> dict[str, np.ndarray]:
    """Error / bound of Thm. 4 (i), the trace bound, and (ii), where each bound is informative (below sqrt(2k)).

    Random worlds with a gap of at least .05, two modalities each with 1-4 observations per state and positives
    mixing good views, pairs of an observation with itself and random pairs (:func:`mixed_views`).
    """
    rng, out = np.random.default_rng(seed), {"distance": [], "trace": [], "cross": []}
    for _ in range(worlds):
        n, k = rng.integers(6, 14), rng.integers(1, 4)
        pi, p = random_world(n, rng)
        mu = spectrum(pi)[0]
        if mu[k] - max(mu[k + 1], 0) < 0.05:
            continue
        mods = []
        for _ in range(2):
            state, rho = random_rendering(rng, p)
            W = mixed_views(rng, pi, state, rho, rng.uniform(0, 0.3), 10 ** rng.uniform(-4, -1))
            f, tau = spectral_solution(W, k), coverage_bound(W, state, n, k)
            if tau < np.sqrt(2 * k):
                P = np.eye(n)[state].T * rho / p[:, None]
                out["distance"].append(rotation_distance(f, spectral_solution(world_kernel(W, state, n), k)[state], rho) / tau)
                out["trace"].append((k - np.trace((P @ f).T @ (p[:, None] * (P @ f)))) / tau**2)
            mods.append((W, state, rho, f, tau))
        (W1, s1, r1, f1, t1), (W2, s2, r2, f2, t2) = mods
        rhs = t1 + t2 + davis_kahan_bound(world_kernel(W1, s1, n), world_kernel(W2, s2, n), k)
        if rhs < np.sqrt(2 * k):
            out["cross"].append(paired_distance(f1, f2, s1, s2, r1, r2, n) / rhs)
    return {key: np.array(v) for key, v in out.items()}


# -- Thm. 4 illustrated: classes of world states, observed through samples on a line ---------------------------------
# C classes of S states; the image side is a good-view JEPA whose world kernel links states of one class. The text side
# observes each state through samples on a line: one block of width 1 per class, one interval of width 1/S per state,
# blocks separated by ``gap`` (literal data) or adjacent (``gap = 0``, mixed data). Its bad views of reach r join two
# samples with weight exp(-d^2 / 2 r^2), the overlap of their corruptions.


def class_world(classes: int = 5, per_class: int = 6, leak: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """``(pi, p)``: uniform states, linked with weight 1 within a class and ``leak`` across classes."""
    label = np.repeat(np.arange(classes), per_class)
    p = np.full(classes * per_class, 1 / (classes * per_class))
    return with_marginal(np.where(label[:, None] == label[None, :], 1.0, leak), p), p


def line_alignment(per_state: int, gap: float, reach, seed: int, classes: int = 5, per_class: int = 6) -> float:
    """Alignment of the text side with the image side, mean canonical correlation ``1 - err^2 / 2k`` (k = C - 1).

    ``reach`` is the reach r of bad views, or ``"good"`` for good views (fresh observations of world-related states).
    """
    pi, p = class_world(classes, per_class)
    k, n = classes - 1, classes * per_class
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(n), per_state)  # stratified, so that the state marginal is p
    x = (z // per_class) * (1 + gap) + (z % per_class + rng.random(len(z))) / per_class
    rho = np.full(len(x), 1 / len(x))
    if reach == "good":
        W = pi[np.ix_(z, z)] / per_state**2
    else:
        W = with_marginal(np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * reach**2)) + 1e-300, rho)
    d = paired_distance(spectral_solution(W, k), spectral_solution(pi, k), z, np.arange(n), rho, p, n)
    return 1 - d**2 / (2 * k)
