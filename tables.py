"""The paper's tables that collect many runs' results, rebuilt from ``evaluate.py report`` files.

    python tables.py paper/data [--results runs/results]

Writes ``phase_grid.csv`` (the real phase diagram: unique images x fraction q of good views x samples seen, on a grid,
at random points ``phase-rand-q<q>-s<samples>``, and at 320M samples), ``phase_law_fit.csv`` (the scaling law fitted
on the grid's 5M- and 20M-sample cells, v1, and on all its cells up to 80M samples, v2), ``text_sources.csv`` (which
texts give the gain) and ``phase_real.csv`` (our encoders and language models against training text tokens), and
prints every run that a fit did not see against the prediction registered before that run trained
(``phase_law_predictions.csv``). Each run's result file is ``<results>/<tag>.txt``, its training arguments the row of
``paper/runs.csv`` with that tag. Runs not finished yet are left out.

From ``evaluate.py pairfree`` lines (``<results>/pairfree``), the zero-pair alignment of Schnaus et al.'s benchmark:
``pairfree.csv`` (each text encoder's runs pooled over training seeds and their five method seeds, and the paired and
few-pair references), ``pairfree_selection.csv`` (their objective on the unpaired pools as a seed-selection score),
``pairfree_tests.csv`` (Mann-Whitney tests between encoders), ``pairfree_captioner.csv`` (the Flickr captioner through
zero-pair maps, ``caption.py --maps``) and ``pairfree_diagnosis.csv`` (``evaluate.py pairfree-diagnose``); from
``evaluate.py video-caption`` and ``video-align`` (``<results>/video``), ``video.csv``.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, spearmanr

#: millions of unique images in --data recaptext:<files>
IMAGES = {2: 0.41, 11: 2.24, 50: 10.2, 250: 50.9}
QS = ["0", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1"]
#: text sources: (millions of images, texts, views, tag without the seed suffix)
SOURCES = [
    (2.24, "all three", "good", "rt11full-good"),
    (2.24, "LLaVA recaptions", "good", "rt11llava-good"),
    (2.24, "alt-text + text-only paraphrase", "good", "rt11altpara-good"),
    (2.24, "alt-text", "bad", "rt11alt-bad"),
    (50, "LLaVA recaptions", "good", "rt250llava-good"),
    (50, "LLaVA recaptions", "bad", "rt250llava-bad"),
    (50, "alt-text", "bad", "rt250alt-bad"),
]
#: training text tokens per sample: a good JEPA reads an image's three texts (mean lengths 15.5 + 50.9 + 62.0 tokens,
#: truncated at 64 as in training), a bad JEPA or next-token one of them; language models: reported pretraining tokens
PER_SAMPLE = {"good": 128.4, "bad": 42.8, "ntp": 42.8}
SAMPLES = {2.16: 21.6e6, 10.0: 20e6, 50.0: 40e6, 100.0: 80e6, 940.0: 900e6}  # --epochs x --epoch-size of each scale
LLM_TOKENS = {"Qwen2.5": 18e12, "Llama-3": 15e12}


def scores(path):
    """WIT-1024 mutual 10-NN with DINOv2 ViT-B and Flickr test R@1 through the 30k-pair rotation."""
    text = path.read_text()
    wit = re.search(r"mutual-kNN dinov2 x \S+: ([0-9.]+)", text)
    r1 = re.search(r"30k-pair rotation\n\s*\d+ [0-9.]+ ([0-9.]+)", text)
    return (float(wit[1]) if wit else np.nan), (float(r1[1]) if r1 else np.nan)


def phase_grid(results):
    rows = []
    for files, images in IMAGES.items():
        for q in QS:
            for seed in range(3):
                tag = f"phase-n{files}-q{q}" + (f"-s{seed}" if seed else "")
                tag = f"rt11full-good-s{seed}" if (files, q) == (11, "1") else tag  # same recipe, trained earlier
                rows.append((files, images, 20, float(q), seed, "grid", tag))
    rows += [(50, 10.2, m, float(q), 0, "grid" if m <= 80 else "long", f"phase-s{m}-q{q}") for m in (5, 40, 80, 320) for q in QS]
    for f in sorted(results.glob("phase-rand-*.txt")):  # random design points on 10.2M images
        q, m = re.fullmatch(r"phase-rand-q([0-9.e-]+)-s(\d+)", f.stem).groups()
        rows.append((50, 10.2, int(m), float(q), 0, "random", f.stem))
    out = [
        (*r[:6], *scores(results / f"{r[6]}.txt"), f"{results}/{r[6]}.txt") for r in rows if (results / f"{r[6]}.txt").exists()
    ]
    return pd.DataFrame(out, columns=["files", "images_M", "samples_M", "q", "seed", "design", "wit", "r1", "source"])


#: the law's fits: which runs each one sees (the others are held out, predicted before they trained)
FITS = {"v1": lambda c: (c.design == "grid") & (c.samples_M <= 20), "v2": lambda c: c.design == "grid"}
NAMES = ["a", "d", "b", "c", "alpha", "beta", "gamma"]


def law(x, a, d, b, c, alpha, beta, gamma):
    """The scaling law R@1 = a + d log M + b log(1 + q^alpha M^beta N^gamma / c).

    A bad-view floor rising with the samples seen M (millions), and a good-view term in the fraction q of good views,
    M and the unique images N (millions).
    """
    q, m, n = x
    return a + d * np.log(m) + b * np.log(1 + q**alpha * m**beta * n**gamma / c)


def fit_law(cells):
    x, y = (cells.q.to_numpy(float), cells.samples_M.to_numpy(float), cells.images_M.to_numpy(float)), cells.r1.to_numpy(float)
    bounds = ([0, -1, 0, 1e-3, 0.1, 0.1, -1], [1, 1, 1, 100, 5, 5, 2])
    theta, cov = curve_fit(law, x, y, p0=[0.2, 0.02, 0.03, 1.0, 1.0, 1.0, 0.1], bounds=bounds, maxfev=20000)
    return theta, np.sqrt(np.diag(cov)), np.sqrt(np.mean((y - law(x, *theta)) ** 2))


def text_sources(results, scaling):
    rows = [
        (images, texts, views, seed, *scores(results / f"{tag}-s{seed}.txt"), f"{results}/{tag}-s{seed}.txt")
        for images, texts, views, tag in SOURCES
        for seed in range(3)
        if (results / f"{tag}-s{seed}.txt").exists()
    ]
    s = scaling.query("encoder == 'BERT-S' and data == 'recap' and images_M == 50.0 and objective in ['good', 'bad']")
    rows += [(50, "all three", r.objective, r.seed, r.wit, r.r1, f"scaling.csv ({r.source})") for _, r in s.iterrows()]
    out = pd.DataFrame(rows, columns=["images_M", "texts", "views", "seed", "wit", "r1", "source"])
    return out.sort_values(["images_M", "texts", "views", "seed"])


def phase_real(scaling, references):
    """Our text encoders and language models against what they were trained on: samples seen and training text tokens."""
    s = scaling.query("data == 'recap'")
    rows = [
        (
            f"{enc}, {images:g}M images",
            obj,
            enc,
            images,
            SAMPLES[images],
            int(SAMPLES[images] * PER_SAMPLE[obj]),
            round(g.wit.mean(), 4),
            len(g),
            "scaling.csv",
        )
        for (obj, enc, images), g in s.groupby(["objective", "encoder", "images_M"])
    ]
    for _, r in references.iterrows():
        family = next((f for f in LLM_TOKENS if r.model.startswith(f)), None)
        if family:
            rows.append((r.model, "ntp", "", np.nan, np.nan, int(LLM_TOKENS[family]), r.wit, 1, f"references.csv ({r.source})"))
    columns = ["model", "objective", "encoder", "images_M", "samples", "tokens", "wit", "seeds", "source"]
    return pd.DataFrame(rows, columns=columns).sort_values(["objective", "tokens"])


#: zero-pair text encoders, pooled over their training seeds: name in the results (``prh.pairfree``) -> paper's row
PAIRFREE = {
    "ours, good JEPA, BERT-B, 940M images": ["rtall-final", "rtall-b64-l01-s1", "rtall-b64-l01-s2"],  # rtall-b64 seed 0
    "ours, good JEPA, BERT-S, 100M images": ["rt500-final", "rt500-s64-xl-s1", "rt500-s64-xl-s2"],  # rt500-s64-xl seed 0
    "ours, bad JEPA, BERT-B, 940M images": ["rtall-masked-b64-l01"],
    "ours, next-token, BERT-B, 940M images": ["rtall-ntp-b64-r", "rtall-ntp-b64-s1-low", "rtall-ntp-b64-s2"],
    "ours, good JEPA, BERT-S, 50M images": ["rt250-s64-long"],
    "ours, bad JEPA, BERT-S, 50M images": ["rt250-masked-long"],
    "ours, next-token, BERT-S, 50M images": ["rt250-ntp-long"],
    "ours, good JEPA, BERT-S, Flickr captions": ["prh-text-cap-1023828"],  # text-cap
    "all-mpnet-base-v2": ["mpnet"],
    "CLIP ViT-B/32 text tower": ["clip"],
    "Qwen3-Embedding-8B": ["qwen3-embedding-8b-red"],
    "Qwen3-8B, generative pooling": ["qwen3-8b-gen-red"],
    "BERT-base": ["bert"],
    "Qwen2.5-0.5B": ["qwen"],
}
PAIRFREE_METRICS = ["foscttm", "R@1", "R@5", "CIFAR-10", "CIFAR-100", "ImageNet-100"]
#: Mann-Whitney comparisons of zero-pair runs (COCO 2014): our 109M good JEPA against two references
PAIRFREE_TESTS = [
    ("ours, good JEPA, BERT-B, 940M images", "Qwen3-8B, generative pooling"),
    ("ours, good JEPA, BERT-B, 940M images", "CLIP ViT-B/32 text tower"),
]


def pairfree(results):
    """The zero-pair tables from the ``evaluate.py pairfree`` lines in ``results``."""
    df = pd.DataFrame([json.loads(line) for f in sorted(results.glob("*.jsonl")) for line in f.read_text().splitlines()])
    df["encoder"] = df.text.map({t: name for name, tags in PAIRFREE.items() for t in tags})
    df["source"] = [f"{results}/{b}_{t}.jsonl" for b, t in zip(df.bench, df.text)]
    rows = []
    for (bench, method, pairs, encoder), d in df.groupby(["bench", "method", "pairs", "encoder"], sort=False):
        r = {"bench": bench, "method": method, "pairs": pairs, "encoder": encoder, "texts": " ".join(dict.fromkeys(d.text))}
        r |= {"runs": len(d), "failure rate (FOSCTTM > .25)": (d.foscttm > 0.25).mean()}
        for m in [m for m in PAIRFREE_METRICS if m in d and d[m].notna().any()]:
            r |= {f"{m} mean": d[m].mean(), f"{m} std": d[m].std(), f"{m} median": d[m].median()}
            r |= {f"{m} min": d[m].min(), f"{m} max": d[m].max()}
        rows.append(r | {"source": " ".join(dict.fromkeys(d.source))})
    pooled = pd.DataFrame(rows)
    wp = df[(df.method == "wp") & (df.pairs == 0)]
    selection = []
    groups = [*wp.groupby(["bench", "text"], sort=False), *(((b, "all runs"), d) for b, d in wp.groupby("bench"))]
    for (bench, text), d in groups:
        pick = d.loc[d.objective.idxmax()]  # their objective on the unpaired pools: no test data
        r = {"bench": bench, "text": text, "runs": len(d), "Spearman(objective, FOSCTTM)": spearmanr(d.objective, d.foscttm)[0]}
        r |= {"selected seed": pick.seed if text != "all runs" else None}
        for m in [m for m in PAIRFREE_METRICS if m in d and d[m].notna().any()]:
            r |= {
                f"selected {m}": pick[m],
                f"mean {m}": d[m].mean(),
                f"best {m} (test, an upper bound)": d[m].min() if m == "foscttm" else d[m].max(),
            }
        selection.append(r)
    tests = []
    for a, b in PAIRFREE_TESTS:
        x, y = (wp[(wp.bench == "coco2014") & (wp.encoder == e)] for e in (a, b))
        for m in ["foscttm", "CIFAR-10", "CIFAR-100", "ImageNet-100"]:
            means = {"mean a": x[m].mean(), "mean b": y[m].mean(), "p": mannwhitneyu(x[m], y[m], alternative="two-sided").pvalue}
            tests.append({"metric": m, "a": a, "b": b, "runs a": len(x), "runs b": len(y), **means})
    return pooled, pd.DataFrame(selection), pd.DataFrame(tests)


def pairfree_captioner(results):
    """CIDEr of the Flickr captioner through each zero-pair map (``caption.py --maps``), and the map the objective picks."""
    scores = pd.read_csv(results / "captioner" / "scores.csv")
    maps = scores.method.str.extract(r"zero-pair map \(flickr_(.+)_wp0_s(\d+)\)")
    scores["text"], scores["seed"] = maps[0], pd.to_numeric(maps[1])
    lines = pd.DataFrame(
        [json.loads(line) for f in sorted(results.glob("flickr_*.jsonl")) for line in f.read_text().splitlines()]
    )
    lines = lines[(lines.method == "wp") & (lines.pairs == 0)][["text", "seed", "objective"]]
    scores = scores.merge(lines, on=["text", "seed"], how="left")
    zp = scores.dropna(subset=["text"])
    scores["selected by the objective"] = scores.index == zp.objective.idxmax()
    return scores.assign(source=f"{results}/captioner/scores.csv")


def pairfree_diagnosis(results):
    """Where zero-pair fits go wrong (``evaluate.py pairfree-diagnose``): centers matched, first map, trajectory."""
    rows = []
    for f in sorted((results / "diagnose").glob("*.json")):
        d = json.loads(f.read_text())
        t = d["foscttm_5k_readout_then_every_10_steps"]
        r = {"bench": d["bench"], "text": d["text"], "seed": d["seed"], "FOSCTTM": d["foscttm"]}
        r["centers matched as the oracle"] = np.mean([x["centers_matched_as_oracle"] for x in d["restarts"]])
        r["first map: cosine with the oracle"] = d["first_map_cosine_with_oracle"]
        r["FOSCTTM (5k): read-out, steps 10, 50, 100"] = f"{t[0]:.3f} {t[1]:.3f} {t[5]:.3f} {t[10]:.3f}"
        rows.append(r | {"FOSCTTM (5k) of the oracle": d["foscttm_oracle_5k"], "source": str(f)})
    return pd.DataFrame(rows)


def video(results):
    """MSR-VTT from frames (``evaluate.py video-caption`` and ``video-align``): captions and retrieval, one table."""
    caps, align = (pd.read_csv(results / f) for f in ("captioning.csv", "alignment.csv"))
    caps["table"], align["table"] = "captioning, 2,990 test clips", "alignment and retrieval, 1k-A"
    return pd.concat([caps.assign(source=f"{results}/captioning.csv"), align.assign(source=f"{results}/alignment.csv")])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="paper/data")
    p.add_argument("--results", type=Path, default=Path("runs/results"), help="evaluate.py report files")
    args = p.parse_args(argv)
    read = lambda name: pd.read_csv(args.data / f"{name}.csv")  # noqa: E731
    grid = phase_grid(args.results)
    grid.to_csv(args.data / "phase_grid.csv", index=False)
    # seed 0 of each cell, the runs the registered predictions came from; the other seeds measure seed noise
    cells = grid[grid.seed == 0].dropna(subset=["r1"]).groupby(["images_M", "samples_M", "q", "design"]).r1.mean().reset_index()
    registered = read("phase_law_predictions").set_index(["law", "samples_M", "q"]).predicted_r1
    fits = []
    for name, seen in FITS.items():
        theta, err, rmse = fit_law(cells[seen(cells)])
        fits.append(pd.DataFrame({"law": name, "param": NAMES + ["rmse"], "value": [*theta, rmse], "stderr": [*err, np.nan]}))
        print(
            f"law {name} fitted on {seen(cells).sum()} cells, RMSE {rmse:.4f}: "
            + ", ".join(f"{k} {v:.3f}" for k, v in zip(NAMES, theta))
        )
        held = cells[~seen(cells)]
        pred = np.array([registered.get((name, c.samples_M, c.q), np.nan) for c in held.itertuples()])
        for c, p in zip(held.itertuples(), pred):
            print(f"  q {c.q:.3g}, {c.samples_M:g}M samples ({c.design}): measured {c.r1:.3f}, registered {p:.3f}")
        ok = ~np.isnan(pred)
        if ok.any():
            print(f"  held-out RMSE {np.sqrt(np.mean((held.r1.to_numpy()[ok] - pred[ok]) ** 2)):.4f} on {ok.sum()} runs")
    pd.concat(fits).to_csv(args.data / "phase_law_fit.csv", index=False)
    text_sources(args.results, read("scaling")).to_csv(args.data / "text_sources.csv", index=False)
    phase_real(read("scaling"), read("references")).to_csv(args.data / "phase_real.csv", index=False)
    if (args.results / "pairfree").exists():
        pooled, selection, tests = pairfree(args.results / "pairfree")
        for name, table in (("pairfree", pooled), ("pairfree_selection", selection), ("pairfree_tests", tests)):
            table.to_csv(args.data / f"{name}.csv", index=False)
        if (args.results / "pairfree" / "captioner" / "scores.csv").exists():
            pairfree_captioner(args.results / "pairfree").to_csv(args.data / "pairfree_captioner.csv", index=False)
        pairfree_diagnosis(args.results / "pairfree").to_csv(args.data / "pairfree_diagnosis.csv", index=False)
    if (args.results / "video" / "alignment.csv").exists():
        video(args.results / "video").to_csv(args.data / "video.csv", index=False)


if __name__ == "__main__":
    main()
