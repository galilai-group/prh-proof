"""Plots of alignment between two encoders, of captions, and of the theory's bound.

Each function writes one figure (PDF in the paper) in the style of Balestriero & LeCun's papers (LeJEPA, 2025;
Learning by Reconstruction, 2024): text typeset by LaTeX in Computer Modern, framed axes, a dashed grid, framed
legends and matplotlib's tab10 colours, applied with ``matplotlib.rc_context`` (global settings are left untouched);
``figures.py`` makes the paper's figures with them.
"""

import functools
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .align import procrustes

# The arms keep one colour in every figure: good views blue, bad views red, next-token orange; alignment is green,
# and in diagrams images are blue and texts orange. Continuous scales run from red (bad, low) to blue (good, high).
BLUE, RED, ORANGE, GREEN = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"
INK, MUTED, OTHER, GRID = "black", "#7f7f7f", "#c7c7c7", "#b0b0b0"
#: red (bad, low) through purple to blue (good, high), with no pale middle, as in LeJEPA's Fig. 10
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("bad_to_good", [RED, "#9467bd", BLUE])
#: caption keywords that colour points in the t-SNE plots; never used in training
GROUPS = {
    "dog": (r"\bdogs?\b|\bpupp", BLUE),
    "water / beach": (r"\bwater\b|beach|ocean|\blake\b|\bpool\b|swim|surf", ORANGE),
    "bike": (r"\bbikes?\b|bicycl|motorcycl|cyclist", GREEN),
}
_UNICODE = {
    "2192": r"\to",
    "2194": r"\leftrightarrow",
    "2265": r"\geq",
    "00D7": r"\times",
    "2248": r"\approx",
    "221D": r"\propto",
}
_UNICODE |= {"03C3": r"\sigma", "03BB": r"\lambda", "0394": r"\Delta", "2713": r"\checkmark", "00B7": r"\cdot"}
#: Figures are drawn at the width they are printed (inches: the page's text width and one column of the two-column
#: paper) and included at their natural size, so no text is smaller than SMALL points, 80% of the paper's 10-point text.
PAGE, COLUMN, SMALL = 7.0, 3.37, 8
STYLE = {
    "text.usetex": shutil.which("latex") is not None,  # without LaTeX, the same layout in matplotlib's fonts
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}"
    + "".join(rf"\DeclareUnicodeCharacter{{{k}}}{{\ensuremath{{{v}}}}}" for k, v in _UNICODE.items()),
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": SMALL,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": SMALL,
    "ytick.labelsize": SMALL,
    "legend.fontsize": SMALL,
    "legend.title_fontsize": SMALL,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "figure.constrained_layout.use": True,  # the figure keeps its printed size: savefig without a tight bounding box
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def plain_log(axis):
    """Log-scale ticks written as plain numbers (0.01, 0.1, 1): exponents would print smaller than SMALL."""
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def styled(fn):
    """Run a plotting function under :data:`STYLE`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(STYLE):
            return fn(*args, **kwargs)

    return wrapper


def tex(text):
    """``text`` with LaTeX's special characters escaped, for free text (captions) in a figure."""
    if not matplotlib.rcParams["text.usetex"]:
        return text
    return re.sub(r"([%&#$_{}])", r"\\\1", text).replace('"', "''")


def _title(ax, text):
    ax.set_title(text)


def _short(text, width, lines=2):
    out = textwrap.wrap(text, width)
    return "\n".join(out[:lines]) + ("…" if len(out) > lines else "")


def end_labels(ax, ends, gap=9):
    """Value labels just right of the ends of lines, nudged apart by at least ``gap`` points.

    Args:
        ax: axes holding the lines.
        ends: ``(x, y, colour)`` of each line's last point; the label is ``y`` with three decimals.
        gap: smallest vertical distance between two labels, in points.
    """
    ax.figure.canvas.draw()  # the final layout, so that data units convert to points
    lo, hi = ax.get_ylim()
    per_point = (hi - lo) / (ax.bbox.height * 72 / ax.figure.dpi)
    placed = []
    for x, y, color in sorted(ends, key=lambda e: e[1]):
        at = max(y, placed[-1] + gap * per_point) if placed else y
        placed.append(at)
        label = f"{y:.3f}".replace("0.", ".", 1)
        ax.annotate(label, (x, y), xytext=(3, (at - y) / per_point), textcoords="offset points", va="center", color=color)


@styled
def caption_grid(paths, refs, captions, out, picks=(0, 3, 7, 11, 21, 42, 64, 99), cols=4):
    """Images with their generated caption (dark) and one human caption (grey, italic), four to a row of the page.

    Panels are placed by hand (inches) so that every image has the same size and every caption fits its column.
    """
    rows, gap = len(picks) // cols, 0.08
    width = (PAGE - gap * (cols - 1)) / cols  # one panel; 160 data units across, the same unit vertically
    height = width * 196 / 160  # the image (120 units) and five lines of captions below it
    fig = plt.figure(figsize=(PAGE, rows * height), layout="none")
    for k, i in enumerate(picks):
        r, c = divmod(k, cols)
        ax = fig.add_axes([c * (width + gap) / PAGE, 1 - (r + 1) / rows, width / PAGE, 1 / rows])
        ax.imshow(ImageOps.fit(Image.open(paths[i]).convert("RGB"), (320, 240)), extent=(0, 160, 64, 184), aspect="auto")
        ax.text(0, 61, tex(_short(captions[i], 28, 3)), va="top", fontsize=8.5, color=INK, linespacing=1.1)
        ax.text(0, 22, tex(_short(refs[i][0], 31)), va="top", color=MUTED, style="italic", linespacing=1.1)
        ax.set_xlim(0, 160)
        ax.set_ylim(-12, 184)
        ax.axis("off")
    fig.savefig(out)
    plt.close(fig)


@styled
def training_curves(
    arms,
    out,
    metrics=(
        ("mknn", "Mutual 10-NN", 0.01),
        ("R@1", "Image→caption R@1, one rotation", 0.001),
        ("R@5", "Image→caption R@5, one rotation", 0.005),
    ),
):
    """Alignment against epochs for several arms, mean and range over seeds.

    Args:
        arms: ``{label: (colour, [per-seed DataFrames with an epoch column and the metric columns])}``.
        out: output PNG.
        metrics: ``(column, title, chance level)`` of each panel.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(PAGE, 2.35))
    for ax, (col, name, chance) in zip(axes, metrics):
        ends = []
        for label, (color, runs) in arms.items():
            d = np.stack([r[col].to_numpy() for r in runs])
            epochs, mean = runs[0].epoch, d.mean(0)
            ax.fill_between(epochs, d.min(0), d.max(0), color=color, alpha=0.18, lw=0)
            ax.plot(epochs, mean, color=color, label=label)
            ends.append((epochs.iloc[-1], mean[-1], color))
        ax.axhline(chance, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("epoch (0 = random init)")
        _title(ax, name)
        end_labels(ax, ends)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(arms))
    fig.savefig(out)
    plt.close(fig)


def _split(a, b, is_test):
    va, vb = a[~is_test], b[~is_test]
    return va - va.mean(0), vb - vb.mean(0), a[is_test] - va.mean(0), b[is_test] - vb.mean(0)


@styled
def pair_figures(ra, rb, pairs, out):
    """t-SNEs, per-direction alignment and retrieval examples of one image run and one text run.

    Args:
        ra: :func:`prh.features.load_embeddings` of the image run; its last epoch in common with ``rb`` is
            compared with the random initialization (epoch 0).
        rb: the same for the text run.
        pairs: :func:`prh.data.flickr_pairs` table (captions colour the points; images illustrate retrieval).
        out: output folder (``tsne.pdf``, ``joint.pdf``, ``spectrum.pdf``, ``retrieval.pdf``).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    last = max(ra.keys() & rb.keys())
    rot = {}
    for e in (0, last):
        va, vb, ta, tb = _split(ra[e][0], rb[e][0], ra[e][2])
        rot[e] = (ta @ procrustes(va, vb), tb, vb)
    meta = pairs.set_index("pair_id").loc[ra[last][1][ra[last][2]].numpy()]
    caps = meta.captions.tolist()
    group = np.array([next((g for g, (pat, _) in GROUPS.items() if re.search(pat, " ".join(c).lower())), "") for c in caps])

    def tsne(x):
        return TSNE(2, init="pca", random_state=0).fit_transform(PCA(50).fit_transform(F.normalize(x, dim=1).numpy()))

    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 3.3))
    for ax, x, name in zip(axes, rot[last][:2], ["Image encoder (never saw text)", "Text encoder (never saw images)"]):
        xy = tsne(x)
        ax.scatter(*xy[group == ""].T, s=5, c=OTHER, lw=0, label="other")
        for g, (_, color) in GROUPS.items():
            ax.scatter(*xy[group == g].T, s=10, c=color, lw=0.3, edgecolors="white", label=g)
        ax.set_xticks([]), ax.set_yticks([])
        _title(ax, f"{name}, epoch {last}")
    axes[1].legend(markerscale=1.5)
    fig.savefig(out / "tsne.pdf")
    plt.close(fig)

    shown = np.random.default_rng(0).choice(len(caps), 60, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 3.3))
    for ax, e in zip(axes, (0, last)):
        r, t, _ = rot[e]
        xy = tsne(torch.cat([r, t]))
        for i in shown:
            ax.plot(*xy[[i, i + len(r)]].T, color=MUTED, lw=0.5, alpha=0.7)
        ax.scatter(*xy[: len(r)].T, s=5, c=BLUE, lw=0, label="image, rotated")
        ax.scatter(*xy[len(r) :].T, s=5, c=ORANGE, lw=0, label="caption")
        ax.set_xticks([]), ax.set_yticks([])
        _title(ax, "Random init, after one rotation" if e == 0 else f"Epoch {e}, after one rotation")
    axes[0].legend(markerscale=2)
    fig.savefig(out / "joint.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.75))
    for e, color in ((0, OTHER), (last, BLUE)):
        r, t, tv = rot[e]
        dirs = torch.linalg.svd(tv, full_matrices=False)[2][:64]
        axes[0].plot(
            range(1, 65),
            [np.corrcoef(r @ d, t @ d)[0, 1] for d in dirs],
            color=color,
            label="init" if e == 0 else f"epoch {e}",
        )
        var = torch.linalg.svdvals(t) ** 2
        axes[1].semilogy(range(1, 65), (var / var.sum())[:64], color=color)
    plain_log(axes[1].yaxis)
    for ax in axes:
        ax.set_xticks([1, 32, 64])
        ax.set_xlabel("text direction")
    axes[0].set_ylabel("correlation")
    axes[0].legend(loc="upper right", handlelength=1)
    _title(axes[0], "Image–text, rotated")
    axes[1].set_ylabel("variance fraction")
    _title(axes[1], "Text spectrum")
    fig.savefig(out / "spectrum.pdf")
    plt.close(fig)

    sims = F.normalize(rot[last][0], dim=1) @ F.normalize(rot[last][1], dim=1).T
    picks = list(dict.fromkeys([np.flatnonzero(group == g)[0] for g in GROUPS] + [11, 517, 733]))[:5]
    fig, axes = plt.subplots(len(picks), 2, figsize=(PAGE, 1.15 * len(picks)), gridspec_kw={"width_ratios": [1, 4.2]})
    for (ax_img, ax_txt), i in zip(axes, picks):
        ax_img.imshow(ImageOps.fit(Image.open(meta.path.iloc[i]).convert("RGB"), (400, 300)))
        ax_img.axis("off")
        top = sims[i].argsort(descending=True)[:3].tolist()
        rank = int((sims[i] > sims[i, i]).sum()) + 1
        lines = [f"{'✓' if j == i else '·'} {tex(textwrap.shorten(caps[j][0], 95))}" for j in top]
        ax_txt.text(0, 0.97, f"true caption rank {rank} / {len(caps)}", color=MUTED, va="top")
        ax_txt.text(0, 0.72, "\n".join(lines), color=INK, va="top", linespacing=1.6)
        ax_txt.axis("off")
    fig.savefig(out / "retrieval.pdf")
    plt.close(fig)


@styled
def davis_kahan(out, cases=300):
    """Thm. 2 on random finite worlds: alignment error after the best rotation against the bound."""
    from . import theory as th

    rows = []
    for case, (pi1, pi2, p, k) in enumerate(th.mismatched_kernels(cases)):
        mu = th.spectrum(pi2)[0]
        if mu[k] - mu[k + 1] > 1e-9:
            err = th.rotation_distance(th.spectral_solution(pi1, k), th.spectral_solution(pi2, k), p)
            rows.append((th.davis_kahan_bound(pi1, pi2, k), err, case % 2))
    bound, err, clustered = map(np.array, zip(*rows))
    fig, ax = plt.subplots(figsize=(COLUMN, 2.7))
    for c, color, label in ((0, BLUE, "random world"), (1, ORANGE, "clustered world (small gaps)")):
        ax.scatter(bound[clustered == c], err[clustered == c], s=7, c=color, lw=0, alpha=0.8, label=label)
    lo, hi = min(bound.min(), err[err > 0].min()), bound.max()
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--", label="error = bound")
    ax.set_xscale("log"), ax.set_yscale("log")
    plain_log(ax.xaxis), plain_log(ax.yaxis)
    ax.set_ylim(err[err > 0].min() / 2, err.max() * 5)
    ax.set_xlabel("bound of Thm. 2 (log-scale)")
    ax.set_ylabel("error after best rotation (log-scale)")
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)


ARM = {"good": (BLUE, "good-view JEPA"), "bad": (RED, "bad-view JEPA"), "ntp": (ORANGE, "next-token")}


def axes_style(ax, title=None):
    if title:
        _title(ax, title)


def hline(ax, y, label, color=MUTED, ls="--", below=False, left=False):
    """Reference line with its label inside the axes, at the right (or left) end, just above (or below) the line."""
    ax.axhline(y, color=color, lw=0.8, ls=ls, zorder=1)
    ax.annotate(
        label,
        (0.01 if left else 0.99, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, -1.5 if below else 1.5),
        textcoords="offset points",
        ha="left" if left else "right",
        va="top" if below else "bottom",
        color=color,
    )


def light(color, amount=0.45):
    """``color`` mixed with white."""
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(c + (1 - c) * amount for c in rgb)
