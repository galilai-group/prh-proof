"""The paper's figures, from evaluation outputs (``evaluate.py align --csv``, ``caption.py``) and training runs.

    python figures.py curves figs/compare.png "good-view JEPA=good_s0.csv,good_s1.csv" "bad-view JEPA=bad_s0.csv,bad_s1.csv"
    python figures.py pair <image run> <text run> figs/     # t-SNEs, per-direction alignment, retrieval examples
    python figures.py captions figs/captions_coco5k/         # examples.png from a caption.py output folder
    python figures.py theory paper/data paper/figs           # synthetic worlds: Thm. 2 and Thm. 4 figures
    python figures.py data paper/data paper/figs              # scaling, video, captioner and ablation figures

``docs/reproduce.md`` lists the inputs of each figure of the paper.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prh import data, plots
from prh.features import load_embeddings
from prh.plots import ARM, BLUE, CMAP, COLUMN, GREEN, GRID, INK, MUTED, ORANGE, PAGE, RED, axes_style, hline, light, styled, tex

BACK = {"fc": "white", "ec": "none", "pad": 0.3}  # behind a label that sits on a line
IMAGES = {2.16: "2M", 10: "10M", 50: "50M", 100: "100M", 940: "940M"}  # tick labels: web images behind the texts


@styled
def scaling(runs, refs, images, out):
    """Alignment against the amount of web text, and with image encoders trained from scratch.

    Args:
        runs: table with columns ``objective`` (good/bad/ntp), ``encoder``, ``params_M``, ``images_M``, ``seed``,
            ``wit`` (PRH-paper score with DINOv2 ViT-B) and ``r1`` (Flickr R@1, rotation from 30k pairs), and a
            ``data`` column (only the ``recap`` rows are drawn).
        refs: table with columns ``model``, ``wit`` and ``r1`` for pretrained references (empty where not measured).
        images: table with columns ``image``, ``text`` (good/ntp/bad), ``seed`` and ``wit`` (one image encoder per bar group).
        out: output PDF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.75), width_ratios=[1, 1, 0.95])
    recap = runs[runs.data == "recap"]
    for ax, col, title in ((axes[0], "wit", "WIT-1024, PRH protocol"), (axes[1], "r1", "Flickr R@1, one rotation")):
        series = sorted(
            recap.groupby(["objective", "encoder"]), key=lambda g: (["good", "ntp", "bad"].index(g[0][0]), g[0][1] != "BERT-S")
        )
        for (obj, enc), d in series:
            color, label = ARM[obj]
            mean = d.groupby("images_M")[col].mean()
            style = {"lw": 1.5, "marker": "o", "ms": 3} if enc == "BERT-S" else {"lw": 1.1, "ls": "--", "marker": "D", "ms": 2.6}
            if len(mean) == 1:  # trained at one scale only: no line to draw
                style["ls"] = "none"
            label = f"{label}, {enc} ({d.params_M.iloc[0]:.0f}M)"
            ax.plot(mean.index, mean.values, color=color, label=label, **style)
            if d.seed.nunique() > 1:  # show each seed where there are several
                ax.scatter(d.images_M, d[col], s=7, color=color, alpha=0.5, lw=0, zorder=3)
        known = refs.dropna(subset=[col])
        lms = known[known.model.str.contains("Qwen|Llama|Mistral")]
        if col == "wit":  # the language models as one band
            ax.axhspan(lms[col].min(), lms[col].max(), color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.annotate(
                "language models, 0.5–72B",
                (0.03, lms[col].max()),
                xycoords=("axes fraction", "data"),
                xytext=(0, 1.5),
                textcoords="offset points",
                va="bottom",
                color=MUTED,
            )
        for _, r in (known.drop(lms.index) if col == "wit" else known).iterrows():  # the top line's label at the left end,
            top = r[col] > known[col].mean()  # the bottom one's at the right, where the lines are clear of data
            hline(ax, r[col], r.model.replace(" (400M pairs)", ""), below=top, left=top)
        ax.set_ylim(top=ax.get_ylim()[1] + 0.012)  # room for the labels of the top lines
        ax.set_xscale("log")
        ticks = sorted(recap.images_M.unique())
        ax.set_xticks(ticks, ["" if t == 100 else IMAGES[t] for t in ticks])  # 100M sits too close to 50M for a label
        ax.minorticks_off()
        ax.set_xlabel("web images behind the texts")
        axes_style(ax, title)
    axes[0].set_ylabel("mutual 10-NN with DINOv2")
    axes[1].set_ylabel("image → caption R@1")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, handlelength=2.2)
    _image_side(axes[2], images)
    fig.savefig(out)
    plt.close(fig)


@styled
def both_sides(cells, dino, out):
    """Good and bad views on each side (MSR-VTT), and alignment with DINOv2 of each image JEPA.

    Args:
        cells: table with columns ``image`` (good/bad), ``text`` (good/bad), ``seed`` and ``mknn``.
        dino: table with columns ``side`` (``image JEPA``/``text JEPA``), ``views`` (good/bad), ``seed`` and ``mknn``.
        out: output PNG.
    """
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 3.9))
    fig.get_layout_engine().set(hspace=0.08)
    ax = axes[0]
    for i, text in enumerate(("good", "bad")):
        for j, image in enumerate(("good", "bad")):
            d = cells[(cells.text == text) & (cells.image == image)]
            x = i + (j - 0.5) * 0.36
            color = BLUE if image == "good" else RED
            ax.bar(x, d.mknn.mean(), width=0.34, color=color, label=f"image JEPA, {image} views" if i == 0 else None)
            ax.scatter([x] * len(d), d.mknn, s=7, color=INK, zorder=3, lw=0)
    ax.axhline(0.01, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["text JEPA, good views", "text JEPA, bad views"])
    ax.set_ylabel("mutual 10-NN")
    ax.set_ylim(0, cells.mknn.max() * 1.45)
    ax.legend(loc="upper right")
    axes_style(ax, "Image JEPA × text JEPA (dots: seeds)")
    ax = axes[1]
    for i, side in enumerate(("image JEPA", "text JEPA")):
        for j, views in enumerate(("good", "bad")):
            d = dino[(dino.side == side) & (dino.views == views)]
            x = i + (j - 0.5) * 0.36
            ax.bar(
                x,
                d.mknn.mean(),
                width=0.34,
                color=BLUE if views == "good" else RED,
                label=f"{views} views" if i == 0 else None,
            )
            ax.scatter([x] * len(d), d.mknn, s=7, color=INK, zorder=3, lw=0)
    ax.set_xticks([0, 1], ["image JEPA", "text JEPA"])
    ax.set_ylabel("mutual 10-NN")
    ax.set_ylim(0, dino.mknn.max() * 1.35)
    ax.legend(loc="upper right", ncol=2)
    axes_style(ax, "Each × frozen DINOv2 (crop views)")
    fig.savefig(out)
    plt.close(fig)


@styled
def captioner(noise, pairs, out):
    """CIDEr of the text-only captioner against the decoder's noise and the number of pairs fitting the map.

    Args:
        noise: table with columns ``sigma`` and ``cider`` (rotation from 1,014 pairs), and a ``matched`` column giving
            the noise matched to that rotation's cosine (same value on every row).
        pairs: table with columns ``pairs``, ``method`` (``captioner``, ``captioner + refinement``, ``retrieval``,
            ``oracle``) and ``cider``.
        out: output PNG.
    """
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 4.2))
    fig.get_layout_engine().set(hspace=0.08)
    ax = axes[0]
    ax.plot(noise.sigma, noise.cider, color=BLUE, marker="o")
    ax.axvline(noise.matched.iloc[0], color=GREEN, lw=1, ls="--")
    ax.annotate(
        "matched to the\nmap's residual",
        (noise.matched.iloc[0], noise.cider.min()),
        xytext=(4, 0),
        textcoords="offset points",
        color=GREEN,
        va="bottom",
    )
    ax.set_xlabel("decoder training noise σ")
    ax.set_ylabel("CIDEr, Flickr test")
    axes_style(ax, "Linear map from 1,014 pairs")
    ax = axes[1]
    styles = {"captioner": (BLUE, "-", "o"), "captioner + refinement": (BLUE, ":", "s"), "retrieval": (INK, "--", "^")}
    for method, (color, ls, marker) in styles.items():
        d = pairs[pairs.method == method].sort_values("pairs")
        ax.plot(d.pairs, d.cider, color=color, ls=ls, marker=marker, label=method)
    oracle = pairs[pairs.method == "oracle"].cider
    if len(oracle):
        hline(ax, oracle.iloc[0], "true caption embedding")
        ax.set_ylim(top=oracle.iloc[0] * 1.15)
    ax.set_xscale("log")
    ax.set_xticks([30, 100, 300, 1000, 30000], ["30", "100", "300", "1k", "30k"])
    ax.minorticks_off()
    ax.set_xlabel("pairs fitting the linear map (log-scale)")
    ax.set_ylabel("CIDEr, Flickr test")
    ax.legend(loc="upper left")
    axes_style(ax, "Decoder trained with noise σ = 3")
    fig.savefig(out)
    plt.close(fig)


@styled
def ablations(table, out):
    """Last-epoch R@1 of the good text JEPA with one setting changed at a time, over seeds.

    Args:
        table: table with columns ``setting`` (label), ``r1`` (empty if that seed collapsed) and ``kind``: ``default``,
            ``change`` (one row per ablation and seed), ``key`` (a change that matters, set apart below the default with
            its change from the default's mean) or ``reference`` (drawn as a vertical line, e.g. the bad-view JEPA).
        out: output PNG.
    """
    refs, runs = table[table.kind == "reference"], table[table.kind != "reference"]
    settings = list(dict.fromkeys(runs.setting))[::-1]
    base = runs[runs.kind == "default"].r1.mean()
    fig, ax = plt.subplots(figsize=(COLUMN, 0.16 * len(settings) + 0.55))
    for y, name in enumerate(settings):
        s = runs[runs.setting == name]
        ok, collapsed = s.r1.dropna(), int(s.r1.isna().sum())
        color, size = (INK, 20) if (s.kind == "default").any() else (BLUE, 12)
        if len(ok):
            ax.scatter(ok, [y] * len(ok), s=5, color=color, alpha=0.4, zorder=2, lw=0)
            ax.scatter(ok.mean(), y, s=size, color=color, zorder=3, lw=0)
        else:
            ax.scatter(0.01, y, marker="x", s=14, color=RED, lw=1)
        if (s.kind == "key").any():
            ax.annotate(f"${100 * (ok.mean() / base - 1):+.0f}\\%$", (ok.max(), y), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK, bbox=BACK)
        if collapsed:
            label = "collapses" if not len(ok) else f"{collapsed} of {len(s)} collapse"
            x = ok.max() if len(ok) else 0.01
            ax.annotate(label, (x, y), xytext=(4, 0), textcoords="offset points", va="center", color=RED, bbox=BACK)
    key = [y for y, name in enumerate(settings) if (runs[runs.setting == name].kind == "key").any()]
    if key:  # the changes that matter, then the rest
        ax.axhline(min(key) - 0.5, color=MUTED, lw=0.5)
    for _, r in refs.iterrows():  # the bad-view JEPA and next-token prediction with the default recipe, written along their lines
        bad = r.setting.startswith("bad")  # the two lines are close: each name on its outer side, beside rows without annotations
        ax.axvline(r.r1, color=RED if bad else ORANGE, lw=0.9, ls="--")
        ax.annotate(
            tex(r.setting),
            (r.r1, len(settings) - 6.5),
            xytext=(2 if bad else -2, 0),
            textcoords="offset points",
            rotation=90,
            ha="left" if bad else "right",
            va="center",
            color=RED if bad else ORANGE,
        )
    ax.set_yticks(range(len(settings)), [tex(s) for s in settings])
    ax.set_ylim(-0.6, len(settings) - 0.4)
    ax.set_xlim(0, table.r1.max() * 1.12)
    ax.set_xlabel("R@1 with frozen DINOv2, last epoch")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(out)
    plt.close(fig)


@styled
def coverage(table, out):
    """Thm. 4 on a synthetic world: alignment against samples per state, for bad views of growing reach.

    Args:
        table: rows of ``coverage_sweep.csv`` (``figures.py theory``): ``data`` (``literal`` or ``mixed``), ``views`` (``good``, ``bad``),
            ``r`` (reach of the bad views), ``per_state``, ``seed`` and ``alignment`` (1 = equal up to a rotation).
        out: output PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.4), sharey=True)
    bad = {0.005: (light(RED, 0.55), "bad, weak"), 0.08: (RED, "bad, medium"), 1.0: (INK, "bad, strong")}
    titles = {"literal": "Literal data: classes apart", "mixed": "Mixed data: classes overlap"}
    for ax, (corpus, title) in zip(axes, titles.items()):
        t = table[table.data == corpus]
        rows = [(t.views == "good", BLUE, "good views", 1.6)]
        rows += [((t.views == "bad") & np.isclose(t.r, r), c, label, 1.2) for r, (c, label) in bad.items()]
        for keep, color, label, lw in rows:
            g = t[keep].groupby("per_state").alignment
            ax.fill_between(g.mean().index, g.min(), g.max(), color=color, alpha=0.15, lw=0)
            ax.plot(g.mean().index, g.mean(), color=color, lw=lw, marker="o", ms=2.5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.minorticks_off()
        ax.set_xlabel("observations per world state (log-scale)")
        axes_style(ax, title)
    axes[0].set_ylabel("alignment with the image side")
    axes[0].set_ylim(0, 1.04)
    axes[1].legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


@styled
def phase(diagram, out):
    """Phase diagram of Thm. 4 on a synthetic world: alignment against the samples per state and the reach of bad views.

    Args:
        diagram: rows of ``phase_diagram.csv`` (``figures.py theory``): ``per_state``, ``views`` (``good``/``bad``), ``r``, ``seed``,
            ``alignment``.
        out: output PDF.
    """
    fig, (top, ax) = plt.subplots(2, 1, figsize=(COLUMN, 2.75), height_ratios=[1, 7])
    fig.get_layout_engine().set(hspace=0, h_pad=0.01)
    bad = diagram[diagram.views == "bad"].groupby(["r", "per_state"]).alignment.mean().unstack()
    n, r = bad.columns.to_numpy(float), bad.index.to_numpy(float)
    edges = lambda v: np.exp(
        np.r_[
            np.log(v[0]) - np.diff(np.log(v))[0] / 2,
            (np.log(v[1:]) + np.log(v[:-1])) / 2,
            np.log(v[-1]) + np.diff(np.log(v))[-1] / 2,
        ]
    )  # noqa: E731
    mesh = ax.pcolormesh(edges(n), edges(r), bad.to_numpy(), cmap=CMAP, vmin=0, vmax=1, shading="flat")
    ax.contour(n, r, bad.to_numpy(), levels=[0.9], colors="white", linewidths=1.0)
    # coverage: a state's samples chain up when r exceeds their largest gap, ~ log(n) / n for n samples per class
    cross = []
    for col, size in zip(bad.columns, n):
        a = bad[col].to_numpy()
        i = np.flatnonzero((a[:-1] < 0.9) & (a[1:] >= 0.9))
        if len(i):
            i = i[0]
            t = (0.9 - a[i]) / (a[i + 1] - a[i])
            cross.append((size, np.exp(np.log(r[i]) + t * (np.log(r[i + 1]) - np.log(r[i])))))
    cross = np.array(cross)
    law = lambda m: np.log(6 * m) / (6 * m)  # noqa: E731
    c = np.exp(np.mean(np.log(cross[:, 1] / law(cross[:, 0]))))
    m = np.geomspace(n[0], n[-1], 50)
    ax.plot(m, c * law(m), color="white", ls="--", lw=0.9)
    ax.annotate("Δ ∝ log N / N", (m[14], c * law(m[14])), xytext=(4, 7), textcoords="offset points", color="white")
    ax.annotate(
        "large reach blurs scenes (.95)",
        (n[0], r[-1]),
        xytext=(2, -2),
        textcoords="offset points",
        va="top",
        color="white",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    ax.set_yticks([1e-3, 1e-2, 0.1, 1], [".001", ".01", ".1", "1"])
    ax.minorticks_off()
    ax.set_xlabel("samples per world state N (log-scale)")
    ax.set_ylabel("reach Δ of bad views (log-scale)")
    good = diagram[diagram.views == "good"].groupby("per_state").alignment.mean()
    top.pcolormesh(edges(n), [0, 1], good.to_numpy()[None, :], cmap=CMAP, vmin=0, vmax=1, shading="flat")
    top.set_xscale("log", base=2)
    top.set_xticks([])
    top.set_yticks([0.5], ["good"])
    top.set_title("Synthetic world (Thm. 4)")
    ax.grid(False), top.grid(False)  # after the scales, which reset the grid
    cb = fig.colorbar(mesh, ax=[top, ax], fraction=0.05, pad=0.02)
    cb.set_label("alignment")
    fig.savefig(out)
    plt.close(fig)


@styled
def phase_law(grid, fit, out):
    """The real phase grid as a scaling law: Flickr R@1 against samples seen, one curve per fraction q of good views.

    Args:
        grid: ``paper/data/phase_grid.csv`` (``images_M``, ``samples_M``, ``q``, ``design``, ``r1``).
        fit: ``paper/data/phase_law_fit.csv``; the curves are law ``v2``, fitted on the grid cells up to 80M samples.
        out: output PNG.
    """
    from matplotlib.colors import LogNorm

    t = fit[fit.law == "v2"].set_index("param").value
    law = lambda q, m, n=10.2: t.a + t.d * np.log(m) + t.b * np.log(1 + q**t.alpha * m**t.beta * n**t.gamma / t.c)  # noqa: E731
    norm, cmap = LogNorm(1e-3, 1), plt.get_cmap(CMAP)
    color = lambda q: INK if q == 0 else cmap(norm(max(q, 1e-3)))  # noqa: E731
    cells = grid[grid.images_M == 10.2].groupby(["samples_M", "q", "design"]).r1.mean().reset_index()
    fig, ax = plt.subplots(figsize=(COLUMN, 2.9))
    m_fit, m_out = np.geomspace(2.5, 80, 100), np.geomspace(80, 400, 60)
    for q in (0, 1e-2, 0.1, 0.3, 1):
        ax.plot(m_fit, law(q, m_fit), color=color(q), lw=1.3)
        ax.plot(m_out, law(q, m_out), color=color(q), lw=1.3, ls="--")
        label = "q=0" if q == 0 else f"{q:g}"
        ax.annotate(
            label, (400, law(q, 400)), xytext=(3, -3 if q == 0 else 1), textcoords="offset points", color=color(q), va="center"
        )
    for held, marker, size in ((False, "o", 22), (True, "D", 18)):
        c = cells[(cells.design != "grid") == held]
        label = "predicted before training" if held else "fitted runs"
        ax.scatter(
            c.samples_M,
            c.r1,
            c=[color(q) for q in c.q],
            marker=marker,
            s=size,
            zorder=3,
            edgecolor=INK if held else "white",
            lw=0.6 if held else 0.4,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 40, 80, 320], ["5M", "10M", "20M", "40M", "80M", "320M"])
    ax.minorticks_off()
    ax.set_xlim(3.5, 700)
    ax.set_ylim(top=law(1, 400) + 0.045)  # room for the legend above the curves
    ax.set_xlabel("samples seen (log-scale)")
    ax.set_ylabel("Flickr R@1 with DINOv2")
    legend = ax.legend(loc="upper left")
    for handle in legend.legend_handles:  # the markers' shapes, not one run's colour
        handle.set_facecolor(MUTED)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, fraction=0.06, pad=0.02, aspect=30)
    cb.set_ticks([1e-3, 1e-2, 0.1, 1], labels=[".001", ".01", ".1", "1"])
    cb.set_label("fraction q of good views")
    axes_style(ax, "BERT-S, texts of 10.2M web images")
    fig.savefig(out)
    plt.close(fig)


def _image_side(ax, table):
    """Bars: each image encoder against the good, next-token and bad text JEPAs (dots: seeds)."""
    images = list(dict.fromkeys(table.image))
    for k, text in enumerate(("good", "ntp", "bad")):
        color, label = ARM[text]
        for i, image in enumerate(images):
            d = table[(table.image == image) & (table.text == text)]
            x = i + (k - 1) * 0.26
            ax.bar(x, d.wit.mean(), width=0.24, color=color, label=label if i == 0 else None)
            if len(d) > 1:
                ax.scatter([x] * len(d), d.wit, s=5, color=INK, zorder=3, lw=0)
    short = {"ViT-S, Recap 2M": "ViT-S\n2M", "ViT-B, Recap 10M": "ViT-B\n10M", "ViT-S, ImageNet": "ViT-S\nIN-1k"}
    ax.set_xticks(range(len(images)), [short.get(i, i.replace(" ", "\n")) for i in images])
    ax.set_ylabel("WIT mutual 10-NN")
    axes_style(ax, "Image encoders (texts of 50M)")


def _arrow(ax, a, b, color=MUTED, ls="-", both=False, lw=1.0):
    style = {"arrowstyle": "<|-|>" if both else "-|>", "color": color, "lw": lw, "ls": ls, "shrinkA": 1, "shrinkB": 1}
    ax.annotate("", xy=b, xytext=a, arrowprops=style | {"mutation_scale": 7})


def _shaded(ax, patch, extent, base, light_at):
    """Fill ``patch`` with ``base`` lit from ``light_at`` (a white highlight fading to a darker rim)."""
    x0, x1, y0, y1 = extent
    xx, yy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y1, y0, 120))
    t = np.clip(np.hypot(xx - light_at[0], yy - light_at[1]) / max(x1 - x0, y1 - y0), 0, 1)[..., None]
    rgb = np.array(matplotlib.colors.to_rgb(base))
    img = np.where(t < 0.55, 1 - (1 - rgb) * (0.25 + t / 0.55 * 0.75), rgb * (1 - 0.5 * (t - 0.55) / 0.45))
    ax.add_patch(patch)
    ax.imshow(img, extent=extent, clip_path=patch, zorder=patch.get_zorder(), interpolation="bilinear")


def _scene(ax, x, y, s, shaded=True, swap=False):
    """A blue cone and a red sphere, the world of the PRH paper's figure; ``swap`` sees it from the other side."""
    from matplotlib.patches import Circle, Polygon

    side = -1 if swap else 1
    cx, sx = x - side * 0.28 * s, x + side * 0.12 * s  # cone and sphere
    w, h, r = 0.26 * s, 0.85 * s, 0.24 * s
    t = np.linspace(np.pi, 2 * np.pi, 30)
    cone = Polygon([(cx, y + h), *zip(cx + w * np.cos(t), y + 0.12 * s * np.sin(t))], closed=True, ec="none", fc=BLUE, zorder=2)
    ball = Circle((sx, y + 0.08 * s), r, ec="none", fc=RED, zorder=3 if not swap else 1.5)
    if shaded:
        _shaded(ax, cone, (cx - w, cx + w, y - 0.12 * s, y + h), BLUE, (cx - 0.5 * w, y + 0.5 * h))
        _shaded(ax, ball, (sx - r, sx + r, y + 0.08 * s - r, y + 0.08 * s + r), RED, (sx - 0.4 * r, y + 0.08 * s + 0.4 * r))
    else:
        ax.add_patch(cone), ax.add_patch(ball)


def _teaser_world(ax):
    """The PRH picture (one world, two modalities, two encoders) with good views drawn in and bad views below."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ax.set_xlim(0, 3.35)
    ax.set_ylim(0, 3.05)
    ax.axis("off")
    ax.text(0.02, 3.03, "Good views make two JEPAs agree (Thm. 1)", fontsize=9, va="top")
    _scene(ax, 1.68, 2.3, 0.5)
    ax.text(1.98, 2.66, "world $z$", va="center")
    _arrow(ax, (1.42, 2.32), (1.05, 2.07), color=INK)
    _arrow(ax, (1.95, 2.32), (2.32, 2.07), color=INK, ls="--")
    ax.text(0.04, 2.06, "images", color=BLUE)
    for k, (x0, swap) in enumerate(((0.04, False), (0.71, True))):  # two photographs of the scene
        ax.add_patch(Rectangle((x0, 1.5), 0.6, 0.45, fc="white", ec=INK, lw=0.6))
        _scene(ax, x0 + 0.3, 1.58, 0.36 if k == 0 else 0.3, shaded=False, swap=swap)
    ax.text(3.32, 2.06, "texts", color=ORANGE, ha="right")
    for k, caption in enumerate(("A red sphere next to", "A blue cone behind")):
        rest = ("a blue cone.", "a red ball.")[k]
        ax.text(
            1.98,
            1.95 - 0.33 * k,
            rf"\texttt{{{caption}}}" + "\n" + rf"\texttt{{{rest}}}",
            va="top",
            linespacing=1.05,
            bbox={"boxstyle": "round,pad=0.22", "fc": light(ORANGE, 0.9), "ec": ORANGE, "lw": 0.6},
        )
    for x0, x1, what in ((0.04, 1.31, "two photos"), (1.94, 3.3, "two captions")):  # good views: two observations
        ax.plot([x0, x0, x1, x1], [1.4, 1.35, 1.35, 1.4], color=GREEN, lw=0.9)
        ax.text((x0 + x1) / 2, 1.3, f"good view: {what}", color=GREEN, ha="center", va="top")
    for x, name in ((0.67, "image"), (2.62, "text")):
        _arrow(ax, (x, 1.12), (x, 0.93), color=INK)
        ax.text(x + 0.06, 1.03, f"{name} JEPA", va="center")
        ax.add_patch(Rectangle((x - 0.05, 0.6), 0.1, 0.3, fc="white", ec=INK, lw=0.6))
    _arrow(ax, (0.78, 0.72), (2.51, 0.72), color=GREEN, both=True, lw=1.2)
    ax.text(1.645, 0.75, "equal up to one rotation", color=GREEN, ha="center", va="bottom")
    box = FancyBboxPatch((0.02, 0.01), 3.31, 0.5, boxstyle="round,pad=0,rounding_size=0.04", fc=light(RED, 0.93), ec=RED, lw=0.6)
    ax.add_patch(box)
    ax.text(0.07, 0.47, "bad views (Thm. 2): corruptions of one observation", color=RED, va="top")
    ax.text(0.07, 0.31, r"masked: \texttt{a red [M] next to} $\leftrightarrow$ \texttt{[M] sphere next}", va="top")
    ax.text(0.07, 0.16, r"next-token: \texttt{a red sphere next} $\to$ \texttt{to}", va="top")


@styled
def teaser(efficiency, out):
    """Page-1 figure: the PRH picture with good and bad views (left), and alignment against samples seen (right).

    Laid out by hand in inches (6.7 x 3.05, the width of the abstract's box), so every text prints at 8 points or more.

    Args:
        efficiency: rows of ``phase_real.csv`` (``tables.py``): ``model``, ``objective``, ``encoder``, ``samples``
            (training samples seen, empty for language models) and ``wit`` (WIT-1024 with DINOv2 ViT-B).
        out: output PDF (or PNG).
    """
    width, height = 6.7, 3.05
    fig = plt.figure(figsize=(width, height), layout="none")
    place = lambda x, y, w, h: fig.add_axes([x / width, y / height, w / width, h / height])  # noqa: E731 (inches)
    _teaser_world(place(0, 0, 3.35, 3.05))

    ax = place(3.97, 0.4, 2.68, 2.38)
    ours = efficiency.dropna(subset=["samples"])
    fit = np.geomspace(17e6, 1.15e9, 50)
    for obj in ("good", "ntp", "bad"):
        d = ours[ours.objective == obj]
        color, label = ARM[obj]
        slope, icept = np.polyfit(np.log(d.samples), d.wit, 1)
        ax.plot(fit, icept + slope * np.log(fit), color=color, lw=1, alpha=0.8, zorder=1)
        for enc, marker in (("BERT-S", "o"), ("BERT-B", "s")):
            e = d[d.encoder == enc]
            ax.scatter(e.samples, e.wit, color=color, marker=marker, s=16, lw=0.4, edgecolor="white", zorder=3)
        ax.plot(
            [],
            [],
            color=color,
            marker="o",
            ms=3.5,
            label={"good": "good-view JEPA (ours)", "ntp": "next-token (LLM objective)", "bad": "bad-view JEPA"}[obj],
        )
        last = d.loc[d.samples.idxmax()]
        ax.annotate(
            f"{last.wit:.3f}"[1:], (last.samples, last.wit), xytext=(5, -4), textcoords="offset points", va="center", color=color
        )
    llm = efficiency[efficiency.model.isin(["Qwen2.5-72B", "Llama-3-70B"])]
    for y in llm.wit:
        ax.axhline(y, color="black", lw=0.9, ls="--", zorder=0)
    ax.annotate(
        "Qwen2.5-72B, Llama-3-70B",
        (0.02, llm.wit.max()),
        xycoords=("axes fraction", "data"),
        xytext=(0, 1),
        textcoords="offset points",
        va="bottom",
        color="black",
    )
    good = ours[(ours.objective == "good") & (ours.encoder == "BERT-B")]
    ntp = ours[(ours.objective == "ntp") & (ours.samples == ours.samples.max())].iloc[0]
    good = good[good.wit >= ntp.wit - 0.005].sort_values("samples").iloc[0]  # the fewest samples to match next-token
    y = (good.wit + ntp.wit) / 2
    _arrow(ax, (ntp.samples, y), (good.samples, y), color=GREEN, lw=1.3)
    ax.annotate(
        f"{ntp.samples / good.samples:.0f}$\\times$ fewer samples",
        (np.sqrt(good.samples * ntp.samples), y),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=GREEN,
    )
    ax.scatter([], [], marker="o", color=MUTED, s=16, label="33M")
    ax.scatter([], [], marker="s", color=MUTED, s=16, label="109M")
    ax.set_xscale("log")
    ax.set_xticks([2e7, 5e7, 1e8, 3e8, 9e8], ["20M", "50M", "100M", "300M", "900M"])
    ax.minorticks_off()
    ax.set_xlim(1.5e7, 2e9)
    ax.set_ylim(0.04, 0.23)  # room for the legend above the data
    ax.set_yticks(np.arange(0.04, 0.201, 0.02))
    ax.set_xlabel("training samples seen (log-scale)")
    ax.set_ylabel("alignment with DINOv2 (WIT)")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.2)
    axes_style(ax, "Same alignment, far fewer samples")
    fig.savefig(out)
    plt.close(fig)


def theory(data, out):
    """The paper's synthetic-world results, recomputed from :mod:`prh.theory` (a few minutes on a CPU).

    Draws the Thm. 2 figure, prints the tightness of Thm. 4's bounds (appendix), and writes the coverage sweep and
    the data x reach phase diagram of Thm. 4 to ``data`` (``coverage_sweep.csv``, ``phase_diagram.csv``), then
    draws the coverage figure; ``figures.py data`` draws the phase diagram with our runs.
    """
    from prh import theory as th

    plots.davis_kahan(out / "davis_kahan.pdf")
    t = th.thm4_tightness()
    for key, what in (("distance", "(i) distance / tau"), ("trace", "(i) (k - tr C_f) / tau^2"), ("cross", "(ii) error / bound")):
        print(f"Thm. 4 {what}: {len(t[key])} informative cases, max {t[key].max():.3f}, median {np.median(t[key]):.4f}")
    rows = [
        (corpus, "good" if r == "good" else "bad", "" if r == "good" else r, n, seed, th.line_alignment(n, gap, r, seed))
        for gap, corpus in ((0.5, "literal"), (0.0, "mixed"))
        for n in (1, 2, 4, 8, 16, 32, 64)
        for seed in range(5)
        for r in ("good", 0.005, 0.02, 0.08, 0.3, 1.0)
    ]
    sweep = pd.DataFrame(rows, columns=["data", "views", "r", "per_state", "seed", "alignment"])
    sweep.to_csv(data / "coverage_sweep.csv", index=False, float_format="%.4f")
    reach = ("good", 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    rows = [
        (n, "good" if r == "good" else "bad", "" if r == "good" else r, seed, th.line_alignment(n, 0.5, r, seed))
        for n in (1, 2, 4, 8, 16, 32, 64, 128)
        for seed in range(3)
        for r in reach
    ]
    grid = pd.DataFrame(rows, columns=["per_state", "views", "r", "seed", "alignment"])
    grid.to_csv(data / "phase_diagram.csv", index=False, float_format="%.4f")
    coverage(pd.read_csv(data / "coverage_sweep.csv"), out / "coverage.pdf")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="figure", required=True)
    s = sub.add_parser("curves", help="alignment against epochs, mean and range over seeds")
    s.add_argument("out", type=Path)
    s.add_argument("arms", nargs="+", help="label=seed0.csv,seed1.csv,...")
    s = sub.add_parser("pair", help="t-SNEs, per-direction alignment and retrieval examples of two runs")
    s.add_argument("image_run", type=Path)
    s.add_argument("text_run", type=Path)
    s.add_argument("out", type=Path)
    s = sub.add_parser("captions", help="caption examples from a caption.py output folder")
    s.add_argument("folder", type=Path)
    s.add_argument("--picks", type=int, nargs="+", default=[0, 3, 7, 11, 21, 42, 64, 99], help="test images shown, 4 per row")
    s.add_argument("--out", type=Path, help="output PNG (default: <folder>/examples.png)")
    s = sub.add_parser("theory", help="synthetic worlds: Thm. 2 figure, Thm. 4 tightness, coverage sweep, data x reach")
    s.add_argument("data", type=Path, help="where the synthetic tables are written (paper/data)")
    s.add_argument("out", type=Path, help="where the figures are written (paper/figs)")
    s = sub.add_parser("data", help="figures drawn from the result tables in paper/data (each row names its source run)")
    s.add_argument("data", type=Path)
    s.add_argument("out", type=Path)
    args = p.parse_args(argv)
    if args.figure == "curves":
        colors = [plots.BLUE, plots.RED, plots.ORANGE, plots.MUTED]
        arms = {
            label: (colors[i], [pd.read_csv(f) for f in files.split(",")])
            for i, (label, files) in enumerate(a.rsplit("=", 1) for a in args.arms)
        }
        plots.training_curves(arms, args.out)
    elif args.figure == "pair":
        plots.pair_figures(load_embeddings(args.image_run), load_embeddings(args.text_run), data.flickr_pairs(), args.out)
    elif args.figure == "theory":
        theory(args.data, args.out)
    elif args.figure == "data":
        read = lambda name: pd.read_csv(args.data / f"{name}.csv")  # noqa: E731
        scaling(read("scaling"), read("references"), read("image_side"), args.out / "scaling.pdf")
        both_sides(read("msrvtt"), read("msrvtt_dinov2"), args.out / "both_sides.pdf")
        captioner(read("captioner_noise"), read("captioner_pairs"), args.out / "captioner.pdf")
        ablations(read("ablations"), args.out / "ablations.pdf")
        coverage(read("coverage_sweep"), args.out / "coverage.pdf")
        phase(read("phase_diagram"), args.out / "phase.pdf")
        phase_law(read("phase_grid"), read("phase_law_fit"), args.out / "phase_law.pdf")
        for name in ("teaser.pdf", "teaser.png"):  # the PNG is the README's
            teaser(read("phase_real"), args.out / name)
    else:
        out = json.load(open(args.folder / "captions.json"))
        zero = next(k for k in out["captions"] if k.startswith("zero-pair"))
        plots.caption_grid(out["files"], out["refs"], out["captions"][zero], args.out or args.folder / "examples.png", args.picks)


if __name__ == "__main__":
    main()
