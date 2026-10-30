"""Evaluate alignment between independently trained encoders. Pairs are used here only, never in training.

    python evaluate.py embed dinov2 runs/ref/flickr_dinov2 --pairs flickr-all    # frozen embeddings of the pairs
    python evaluate.py embed <run>/checkpoints/last.ckpt runs/ref/<name> --pairs flickr-all
    python evaluate.py align <image run or embeddings> <text run or embeddings>  # per-epoch alignment table
    python evaluate.py wit-embed dinov2 && python evaluate.py wit-embed <ckpt> --name <name>
    python evaluate.py wit dinov2:<name> [...]                                   # PRH-paper score on WIT-1024
    python evaluate.py sts <ckpt or model> [...]                                 # text only: STS-B Spearman
    python evaluate.py unimodal <ckpt or model> [...]                            # each encoder alone
    python evaluate.py nuisance <ckpt or model> [...]                            # Thm. 3: scene-explained fraction
    python evaluate.py report <ckpt> --name <name>      # a text JEPA's numbers in the paper's tables (tables.py)
    python evaluate.py pairfree-embed <name>=<ckpt or model> [...]  # COCO 2014 captions, as Schnaus et al. store them
    python evaluate.py pairfree coco2014|flickr <name> [...]        # zero-pair maps (prh.pairfree) and their scores
    python evaluate.py pairfree-diagnose coco2014 <name> <seed>     # where a zero-pair fit goes wrong
    python evaluate.py video-embed <captioner.pt> --rotation <rotation.pt>     # MSR-VTT clips, 8 frames each (GPU)
    python evaluate.py video-caption <captioner.pt>                            # captions of the clips, and scores
    python evaluate.py video-align <captioner.pt> --rotation <rotation.pt>     # 1k-A alignment and retrieval

``align`` fits everything (centering, rotation, whitening) on the non-test pairs and reports, on the 1,000
Flickr test pairs: mutual 10-NN (no fitting, chance ~0.01), image-to-caption R@1 and R@5 through one rotation
(chance 0.001) and the same after whitening each side. Embeddings dumped during training (``train.py``) and
frozen ones (``embed``) share one format. With ``--pairs flickr-all``, the rotation is fitted on all ~30k Flickr
train and val pairs instead of the 1,014 val pairs.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from prh import data, features, metrics
from prh.align import Rotation

WIT = Path(__file__).resolve().parent / "runs" / "prh_wit"
#: cached per-layer features of each benchmark (wit-embed / wit --bench)
BENCH = {"wit": WIT, "coco": WIT.parent / "prh_coco"}
PAIRS = {
    "flickr": lambda: data.flickr_pairs(("val", "test")),
    "flickr-all": lambda: data.flickr_pairs(("train", "val", "test")),
    "coco5k": data.coco5k_pairs,
    "msrvtt": data.msrvtt_pairs,
}


def is_image_model(model: str) -> bool:
    """Whether ``model`` (a name or a ``train.py`` checkpoint) embeds images."""
    if features.is_image_model_name(model) or model in features.TEXT_MODELS or model in ("random-bert-s", "tfidf"):
        return features.is_image_model_name(model)
    return torch.load(model, map_location="cpu", weights_only=False, mmap=True)["hyper_parameters"]["modality"] == "image"


def embed(args):
    """Frozen embeddings of one side of the pairs (images, or their first captions)."""
    pairs = PAIRS[args.pairs]()
    if is_image_model(args.model):
        x = features.embed_images(args.model, pairs.path.tolist())
    else:
        x = features.embed_texts(args.model, [c[0] for c in pairs.captions], last_token=args.last_token)
    features.save_embeddings(args.out, x, torch.tensor(pairs.pair_id.values), torch.tensor(pairs.is_test.values))
    print(args.model, tuple(x.shape))


def embed_web(args):
    """Web pairs for the demo's rotation: an image's embedding and the mean embedding of its three texts."""
    pairs = data.RecapPairs(args.shards)
    rows = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0))[: args.n].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(pairs, rows), batch_size=256, num_workers=16, collate_fn=list)
    images, texts = [], []
    for batch in loader:
        images.append(features.embed_images(args.image_model, [im for im, _ in batch]))
        flat = features.embed_texts(str(args.text_model), [t for _, ts in batch for t in ts])
        texts.append(flat.view(len(batch), -1, flat.shape[1]).mean(1))
    torch.save({"image": torch.cat(images), "text": torch.cat(texts)}, args.out)
    print(args.image_model, "web pairs", tuple(torch.cat(images).shape))


def align(args):
    """Alignment per validated epoch; a frozen side (one epoch) is paired with every epoch of the other."""
    ra, rb = features.load_embeddings(args.a, args.key), features.load_embeddings(args.b, args.key)
    if args.last:  # final checkpoints only, whatever their epochs; labelled by the later of the two
        last = max(max(ra), max(rb))
        ra, rb = {last: ra[max(ra)]}, {last: rb[max(rb)]}
    ra = dict.fromkeys(rb, next(iter(ra.values()))) if len(ra) == 1 else ra
    rb = dict.fromkeys(ra, next(iter(rb.values()))) if len(rb) == 1 else rb
    if args.test_a:  # fit on the non-test pairs of a and b, score on every pair of another dataset
        (xa, pa, _), (xb, pb, _) = (features.last_embeddings(d, args.key) for d in (args.test_a, args.test_b))
        assert torch.equal(pa, pb), "the two test sides were embedded on different pairs"
    rows = []
    for epoch in sorted(ra.keys() & rb.keys()):
        (a, pa, ta), (b, pb, tb) = ra[epoch], rb[epoch]
        assert torch.equal(pa, pb) and torch.equal(ta, tb), "the two sides were embedded on different pairs"
        if args.val_only:
            a, b, ta = a[~ta], b[~ta], ta[~ta]
        if args.split:  # ignore the dataset's split: fit on a random subset of pairs, score on the rest
            ta = torch.ones(len(a), dtype=torch.bool)
            ta[torch.randperm(len(a), generator=torch.Generator().manual_seed(0))[: args.split]] = False
        if args.test_a:
            a, b, ta = (
                torch.cat([a[~ta], xa]),
                torch.cat([b[~ta], xb]),
                torch.cat([torch.zeros(int((~ta).sum())), torch.ones(len(xa))]).bool(),
            )
        rows.append({"epoch": epoch, **metrics.alignment(a, b, ta, test_means=bool(args.test_a))})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))
    if args.csv:
        df.to_csv(args.csv, index=False)
    return df


def wit_embed(args):
    """Per-layer features of one model on WIT-1024 (or on the 5,000 COCO test pairs), cached for :func:`wit`."""
    if not args.name and args.model not in {**features.IMAGE_MODELS, **features.TEXT_MODELS}:
        raise SystemExit("wit-embed: name the features of a checkpoint with --name")
    if args.bench == "coco":  # each image and its first caption
        t = data.coco5k_pairs()
        images, texts = list(t.path), [c[0] for c in t.captions]
    else:
        ds = data.wit1024()
        images, texts = ds["image"], [t[0] for t in ds["text"]]
    if is_image_model(args.model):
        feats = features.embed_images(args.model, images, layers=True)
    else:
        feats = features.embed_texts(args.model, texts, layers=True)
    BENCH[args.bench].mkdir(parents=True, exist_ok=True)
    torch.save(feats, BENCH[args.bench] / f"{args.name or args.model}.pt")
    print(args.name or args.model, len(feats), "layers", tuple(feats[-1].shape))


def wit(args):
    """PRH-paper alignment score (max over layer pairs of mutual 10-NN, or of unbiased CKA) of cached ``a:b`` features.

    With ``--subsample``, also 95-percent intervals for each score and for its difference with the first pair, from
    random subsets of ``--subset-size`` pairs (paired across models) rescaled to the 1,024 pairs (:func:`prh.metrics.subsample_interval`).
    """
    if args.subsample and args.metric != "mknn":
        raise SystemExit("wit: --subsample intervals are for mutual k-NN only")
    full, subsets, lines = {}, {}, []
    for pair in args.pairs:
        a, b = pair.split(":")
        fa, fb = torch.load(BENCH[args.bench] / f"{a}.pt"), torch.load(BENCH[args.bench] / f"{b}.pt")
        if args.drop:  # pairs found among the Recap-DataComp training data (python -m prh.data overlap)
            rows = json.load(open(args.drop)).values()
            if args.bench == "coco":  # COCO test photos among the images
                found = set().union(*(set(r[4]) for r in rows))
                keep = torch.tensor([i not in found for i in data.coco5k_pairs().pair_id])
            else:  # WIT captions among the alt-texts
                found = set().union(*(set(r[1]) | set(r[2]) for r in rows))
                keep = torch.tensor([i not in found for i in range(len(fa[0]))])
            fa, fb = [x[keep] for x in fa], [x[keep] for x in fb]
        full[pair], i, j = metrics.prh_score(fa, fb, metric=args.metric)
        lines.append(
            f"PRH {'WIT-1024' if args.bench == 'wit' else 'COCO-5k'} {'mutual-kNN' if args.metric == 'mknn' else 'unbiased-CKA'} {a} x {b}: {full[pair]:.3f} (layers {i}/{len(fa) - 1}, {j}/{len(fb) - 1})"
        )
        print(lines[-1], flush=True)
        if args.subsample:
            subsets[pair] = metrics.prh_subsample(fa, fb, args.subsample, args.subset_size)
            n = len(fa[0])
    first = args.pairs[0]
    for pair, x in subsets.items():
        lo, hi = metrics.subsample_interval(full[pair], x, n, args.subset_size)
        print(f"  {pair}: {full[pair]:.3f} [{lo:.3f}, {hi:.3f}]")
        if pair != first:
            d = full[first] - full[pair]
            lo, hi = metrics.subsample_interval(d, subsets[first] - x, n, args.subset_size)
            print(f"  {first} - {pair}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return lines


def sts(args):
    """STS-B test Spearman correlation of text encoders, no fine-tuning."""
    df = data.stsb()
    for model in args.models:
        a, b = (features.embed_texts(model, df[c].tolist()) for c in ("sentence1", "sentence2"))
        print(f"{model}: STS-B Spearman {metrics.sts_spearman(a, b, df.score):.3f}", flush=True)


def unimodal(args):
    """Each encoder alone: Imagenette 20-NN top-1, or caption-to-caption R@1 on Flickr.

    Image encoders classify Imagenette validation images by their 20 nearest training images; text encoders (and
    ``tfidf``) retrieve, among the 5,000 Flickr test captions, one of the four others describing the same image.
    ``random-vit-s`` and ``random-bert-s`` are the untrained architectures (seed 0).
    """
    import torch.nn.functional as F

    for model in args.models:
        if is_image_model(model):
            import stable_pretraining as spt

            split = {}
            for s in ("train", "validation"):
                d = spt.data.HFDataset("frgfm/imagenette", split=s, revision="refs/convert/parquet")
                split[s] = (
                    F.normalize(features.embed_images(model, (x["image"] for x in d)), dim=1),
                    torch.tensor([x["label"] for x in d]),
                )
            (bank, labels), (query, truth) = split["train"], split["validation"]
            votes = labels[(query @ bank.T).topk(20, dim=1).indices].mode(1).values
            print(f"{model}: Imagenette 20-NN top-1 {(votes == truth).float().mean():.3f}", flush=True)
            continue
        captions = [c for cs in data.flickr_pairs(("test",)).captions for c in cs[:5]]
        if model == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            x = torch.tensor(TfidfVectorizer().fit_transform(captions).toarray(), dtype=torch.float32)
        else:
            x = features.embed_texts(model, captions)
        sim = F.normalize(x, dim=1) @ F.normalize(x, dim=1).T
        image = torch.arange(len(captions)) // 5
        r1 = (image[sim.fill_diagonal_(-2).argmax(1)] == image).float().mean()
        print(f"{model}: Flickr caption-to-caption R@1 {r1:.3f}", flush=True)


def nuisance(args):
    """Thm. 3 on text encoders: how much of each embedding the scene explains, on the Flickr val and test images.

    Embeds the five captions of each of the 2,014 images, keeps each model's top-``k`` principal directions, whitened,
    and prints the scene-explained fraction ``tr C_f^(1/2) / k``, Thm. 3's bound ``2(k - tr C_f^(1/2))`` on the
    best-rotation error, and that error between the first caption and the image's embedding (``--image``, same ``k``).
    """
    pairs = data.flickr_pairs()
    captions = [c[:5] for c in pairs.captions]
    img, pid, _ = features.last_embeddings(args.image)
    order = {int(p): i for i, p in enumerate(pid)}
    img = img[[order[int(p)] for p in pairs.pair_id]].double()
    for model in args.models:
        x = features.embed_texts(model, [t for c in captions for t in c]).double()
        for k in args.k:
            f = metrics.whiten_top(x, k).view(len(captions), 5, k)
            explained = metrics.scene_explained(f)
            error = metrics.rotation_error(metrics.whiten_top(f[:, 0], k), metrics.whiten_top(img, k))
            print(
                f"{model} k={k}: scene-explained fraction {explained:.3f}, Thm. 3 bound {2 * k * (1 - explained):.1f}, "
                f"best-rotation error with the image {error:.1f}",
                flush=True,
            )


def report(args):
    """A text checkpoint's numbers in the paper's tables, written to ``<results>/<name>.txt`` (read by ``tables.py``).

    Flickr test pairs through one rotation fitted on the ~30k train and val pairs (``align``), and the PRH score on
    WIT-1024 with DINOv2 ViT-S/B/L/g (``wit``). Needs ``embed dinov2 runs/ref/flickr_dinov2_all --pairs flickr-all``
    and ``wit-embed dinov2[-s|-l|-g]`` once beforehand.
    """

    def run(*argv):  # one subcommand, parsed as on the command line
        a = parser().parse_args(argv)
        return a.fn(a)

    ref, name = Path("runs/ref"), args.name
    run("embed", str(args.ckpt), str(ref / f"all_{name}"), "--pairs", "flickr-all")
    flickr = run("align", str(ref / "flickr_dinov2_all"), str(ref / f"all_{name}"))
    run("wit-embed", str(args.ckpt), "--name", f"text-{name}")
    lines = run("wit", *[f"{m}:text-{name}" for m in ("dinov2-s", "dinov2", "dinov2-l", "dinov2-g")])
    out = Path(args.results) / f"{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    row = flickr.tail(1).to_string(header=False, index=False, float_format="%.4f")
    out.write_text("\n".join([f"DINOv2 x {name}, 30k-pair rotation", row, *lines]) + "\n")


def pairfree_embed(args):
    """Our text encoders on the COCO 2014 captions and class prompts of Schnaus et al. (2026), in their layout."""
    from prh import pairfree

    for spec in args.models:
        name, model = spec.split("=", 1)
        pairfree.embed(name, model)
        print(name, "embedded", flush=True)


def pairfree_fit(args):
    """Zero-pair maps from DINOv2 to text encoders (``prh.pairfree``), one JSON line of scores per map.

    A map is fitted on the unpaired halves of the corpus and saved in ``<results>/maps``, or read from there if it
    exists; its line goes to ``<results>/<bench>_<text>.jsonl``: FOSCTTM, R@1 and R@5 (Flickr) or zero-shot accuracies
    (COCO 2014), and for zero-pair maps their objective on the unpaired pools, which selects a seed without test data.
    ``--procrustes`` adds the paired references: Procrustes on the known pairs of each seed, or on all pairs.
    """
    from prh import pairfree as pf

    seeds = args.seeds or (pf.SEEDS if args.bench == "coco2014" else list(range(5)))
    method = "wp-shuffled" if args.shuffle else "wp"
    for text in args.texts:
        x, y, val_x, val_y = pf.load(args.bench, text)
        zero_shot = pf.zero_shot_data(text) if args.bench == "coco2014" else None
        out = args.results / f"{args.bench}_{text}.jsonl"
        done = (
            {(r["method"], r["pairs"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else ()
        )

        def write(row, m, seed):
            row = {"bench": args.bench, "text": text, **row, "seed": seed, **pf.scores(m, val_x, val_y, seed, zero_shot)}
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")

        for seed in seeds:
            for k in args.pairs:
                path = args.results / "maps" / f"{args.bench}_{text}_wp{k}_s{seed}{'_shuffled' if args.shuffle else ''}.pt"
                if (method, k, seed) in done:
                    continue
                row, start = {"method": method, "pairs": k}, time.time()
                if path.exists():
                    m = pf.Map.load(path)
                else:
                    m = pf.fit(x, y, seed, k, args.shuffle, args.workers)
                    row["fit_seconds"] = time.time() - start
                    path.parent.mkdir(parents=True, exist_ok=True)
                    m.save(path, bench=args.bench, text=text, pairs=k, seed=seed)
                if not k and not args.shuffle:
                    row["objective"] = pf.objective(m, x, y, seed)
                write(row, m, seed)
            for k in args.procrustes:
                n = len(x) if k == "all" else int(k)
                if ("procrustes", n, seed) in done or (k == "all" and seed != seeds[0]):
                    continue
                rows = torch.arange(len(x)) if k == "all" else pf.split(len(x), seed, n)[3]  # the pairs a fit with k gets
                write({"method": "procrustes", "pairs": n}, pf.procrustes(x, y, rows), seed)


def pairfree_diagnose(args):
    """Where a zero-pair fit goes wrong (``prh.pairfree.diagnose``): ``<results>/diagnose/<bench>_<text>_s<seed>.json``."""
    from prh import pairfree

    report = pairfree.diagnose(*pairfree.load(args.bench, args.text), args.seed)
    out = args.results / "diagnose" / f"{args.bench}_{args.text}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bench": args.bench, "text": args.text, **report}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "restarts"}))


#: the 2,990 MSR-VTT test clips (video7010 to video9999) that zero-shot video captioning is scored on
VIDEO_TEST = data.MSRVTT / "splits/collaborative-experts/test_list_full.txt"
#: video decodings: (beam, samples), the captioner's consensus of a beam and 32 samples, and beam search
VIDEO_DECODINGS = {"consensus32": (1, 32), "beam3": (3, 0)}


def _clip_frames(ids):
    """A batch of clips: the ``data.N_FRAMES`` frames of each, as the image encoders' input ``[B, 8, 3, 224, 224]``."""
    from PIL import Image

    from prh.views import image_eval_view

    view = image_eval_view()
    load = lambda v, k: view({"image": Image.open(data.FRAMES / v / f"{k}.jpg").convert("RGB")})["image"]  # noqa: E731
    return torch.stack([torch.stack([load(v, k) for k in range(data.N_FRAMES)]) for v in ids])


@torch.no_grad()
def video_embed(args):
    """MSR-VTT clips through a captioner bundle (``prh.captioner``) and a second rotation, nothing fitted on video.

    Writes ``<out>/feats.pt``: per clip, its image features averaged over 8 frames and of its middle frame, every frame
    rotated by the bundle's map and by ``--rotation``, the bundle text JEPA's embeddings of the clip's 20 captions
    (their mean, and the first), and of the 1,000 JSFUSION test sentences.
    """
    from prh.captioner import Captioner
    from prh.features import encode_images

    cap, extra = Captioner.load(args.captioner), Rotation.load(args.rotation)
    ids = VIDEO_TEST.read_text().split()[: args.limit] if args.limit else [f"video{i}" for i in range(10_000)]
    frames, rows = torch.utils.data.DataLoader(ids, batch_size=32, num_workers=16, collate_fn=_clip_frames), []
    for x in frames:
        f = encode_images(cap.image_model, x.flatten(0, 1), torch.bfloat16).double().view(len(x), data.N_FRAMES, -1)
        rows.append([f.mean(1).half(), f[:, data.N_FRAMES // 2].half(), cap.rot(f).half(), extra(f).half()])
        print(f"{sum(len(r[0]) for r in rows)}/{len(ids)} clips", flush=True)
    raw_mean, raw_mid, rot, rot_extra = (torch.cat(c) for c in zip(*rows))
    captions = {}
    for c in json.load(open(data.MSRVTT / "annotations/MSRVTT_data.json"))["sentences"]:
        captions.setdefault(c["video_id"], []).append(c["caption"])
    flat = [c for v in ids for c in captions[v]]
    t = torch.cat([cap.text.encode(flat[i : i + 1024]).float().cpu() for i in range(0, len(flat), 1024)]).view(len(ids), 20, -1)
    js = pd.read_csv(data.MSRVTT / "annotations/MSRVTT_JSFUSION_test.csv")
    js = js[js.video_id.isin(ids)]
    feats = {"ids": ids, "raw_mean": raw_mean, "raw_mid": raw_mid, "rot": rot, "rot_extra": rot_extra}
    feats |= {"text_mean": t.mean(1), "text_first": t[:, 0].clone(), "js_ids": js.video_id.tolist()}
    feats["js_text"] = cap.text.encode(js.sentence.tolist()).float().cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(feats, args.out / "feats.tmp")  # then renamed: a preempted job never leaves half a file
    (args.out / "feats.tmp").rename(args.out / "feats.pt")


def video_caption(args):
    """Captions of the 2,990 MSR-VTT test clips from the mean of their frames' rotated embeddings, and their scores.

    Decodes each clip embedding (the mean of 8 frames or the middle frame through the bundle's map, the mean through
    ``--rotation``'s, and the text JEPA embedding of a human caption as an oracle) with the captioner's consensus of
    32 samples and with beam search, appending to ``<out>/captions.json``, then scores every decoding against the
    clips' 20 human captions in ``<out>/captioning.csv`` (CIDEr, BLEU-4, METEOR, ROUGE-L; METEOR needs java).
    """
    from prh.captioner import Captioner

    feats, path = torch.load(args.out / "feats.pt"), args.out / "captions.json"
    at = {v: i for i, v in enumerate(feats["ids"])}
    test = [v for v in VIDEO_TEST.read_text().split() if v in at]
    idx = torch.tensor([at[v] for v in test])
    done = json.load(open(path)) if path.exists() else {"ids": test, "captions": {}}
    if not args.score_only:
        cap = Captioner.load(args.captioner)
        clips = {
            "mean of 8 frames": feats["rot"].double().mean(1),
            "middle frame": feats["rot"].double()[:, data.N_FRAMES // 2],
            "mean of 8 frames, --rotation": feats["rot_extra"].double().mean(1),
            "oracle: text JEPA of a human caption": feats["text_first"].double(),
        }
        for name, z in clips.items():
            for decoding, (beam, samples) in VIDEO_DECODINGS.items():
                key = f"{name} | {decoding}"
                if key not in done["captions"] and not (name.startswith("oracle") and samples == 0):
                    done["captions"][key] = _decode(cap, z[idx], beam, samples)
                    json.dump(done, open(path, "w"), indent=1)  # after every condition: a preempted job loses one
    captions = {}
    for c in json.load(open(data.MSRVTT / "annotations/MSRVTT_data.json"))["sentences"]:
        captions.setdefault(c["video_id"], []).append(c["caption"])
    refs = [captions[v] for v in done["ids"]]
    rows = [
        {"clip": "human captions", "words": sum(len(metrics._norm(r).split()) for rs in refs for r in rs) / sum(map(len, refs))}
    ]
    for key, preds in done["captions"].items():
        clip, decoding = key.split(" | ")
        rows.append({"clip": clip, "decoding": decoding, **metrics.caption_scores(preds, refs, more=True), "clips": len(preds)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(args.out / "captioning.csv", index=False)


@torch.no_grad()
def _decode(cap, z, beam, samples, chunk=256):
    """As ``caption.py``: batched beam search, or the consensus of the ``beam`` best beams and ``samples`` samples."""
    torch.manual_seed(0)
    out = []
    for part in z.split(chunk):
        with torch.autocast("cuda", torch.bfloat16, enabled=features.DEVICE == "cuda"):
            c = cap.dec.condition(part.to(features.DEVICE))
            if not samples:
                out += cap.dec.generate(c, beam=beam)
                continue
            candidates = cap.dec.generate(c, beam=beam, nbest=beam, samples=samples)
        out += [metrics.consensus(candidates[i : i + beam + samples]) for i in range(0, len(candidates), beam + samples)]
        print(f"  {len(out)}/{len(z)}", flush=True)
    return out


def video_align(args):
    """Clips against captions on MSR-VTT 1k-A: mutual 10-NN (fits nothing), and retrieval through one map.

    Each clip is the mean of its 8 frames or its middle frame; each caption, the bundle text JEPA's embedding of its
    JSFUSION sentence; maps: the bundle's, ``--rotation``'s, and Procrustes refitted on the 9,000 training clips
    (each with the mean embedding of its 20 captions). Writes ``<out>/alignment.csv``.
    """
    bundle = torch.load(args.captioner, map_location="cpu", weights_only=True, mmap=True)
    maps = {
        "captioner map": Rotation(**{k: v.double() for k, v in bundle["rotation"].items()}),
        "--rotation": Rotation.load(args.rotation),
    }
    feats = torch.load(args.out / "feats.pt")
    at = {v: i for i, v in enumerate(feats["ids"])}
    train = torch.tensor([at[v] for v in pd.read_csv(data.MSRVTT / "annotations/MSRVTT_train.9k.csv").video_id])
    test = torch.tensor([at[v] for v in feats["js_ids"]])
    b_train, b = feats["text_mean"][train].double(), feats["js_text"].double()
    mu_b, rows = b_train.mean(0), []  # the caption mean: an unpaired statistic
    for clip, key in (("mean of 8 frames", "raw_mean"), ("middle frame", "raw_mid")):
        a_train, a = feats[key][train].double(), feats[key][test].double()
        mknn = metrics.mutual_knn(a - a_train.mean(0), b - mu_b, 10)
        for name, m in {**maps, "refitted on the 9k training clips": Rotation.fit(a_train, b_train)}.items():
            v, t = m(a) - m.mu_b, b - m.mu_b
            s = torch.nn.functional.normalize(t, dim=1) @ torch.nn.functional.normalize(v, dim=1).T
            rank = {"t2v": (s >= s.diag()[:, None]).sum(1), "v2t": (s.T >= s.diag()[:, None]).sum(1)}  # 1-based, ties against
            row = {"clip": clip, "map": name, "mutual 10-NN (no map)": mknn}
            rows.append(row | {f"{d} R@{k}": (r <= k).float().mean().item() for d, r in rank.items() for k in (1, 5, 10)})
    df = pd.DataFrame(rows).round(4)
    print(df.to_string(index=False))
    df.to_csv(args.out / "alignment.csv", index=False)


def parser():
    """The command line: one subcommand per function above."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("embed", help=embed.__doc__)
    s.add_argument("model", help="dinov2[-s|-l|-g], a pretrained text model, or a train.py checkpoint")
    s.add_argument("out", type=Path)
    s.add_argument("--pairs", choices=list(PAIRS), default="flickr")
    s.add_argument("--last-token", action="store_true", help="text checkpoints: the last token's state, not the model's pooling")
    s.set_defaults(fn=embed)
    s = sub.add_parser("embed-web", help=embed_web.__doc__)
    s.add_argument("image_model", help="frozen image encoder, e.g. dinov2-g+avg")
    s.add_argument("text_model", type=Path, help="the text JEPA's checkpoint")
    s.add_argument("out", type=Path)
    s.add_argument("--n", type=int, default=200_000, help="random web pairs")
    s.add_argument("--shards", type=int, default=300, help="Recap-DataComp tar shards to draw them from")
    s.set_defaults(fn=embed_web)
    s = sub.add_parser("align", help=align.__doc__)
    s.add_argument("a", type=Path, help="image side: a train.py run dir or an embed output")
    s.add_argument("b", type=Path, help="text side")
    s.add_argument("--key", default="embedding", choices=["embedding", "proj"], help="backbone or projector output")
    s.add_argument("--split", type=int, default=0, help="fit on this many random pairs, score on the others")
    s.add_argument("--val-only", action="store_true", help="drop the test pairs first (with --split: no test pair is used)")
    s.add_argument("--last", action="store_true", help="compare the final validation of each run only")
    s.add_argument("--test-a", type=Path, help="score on all pairs of another dataset (image side), centred by its own means")
    s.add_argument("--test-b", type=Path, help="text side of those pairs")
    s.add_argument("--csv", type=Path)
    s.set_defaults(fn=align)
    s = sub.add_parser("wit-embed", help=wit_embed.__doc__)
    s.add_argument("model")
    s.add_argument("--name", help="cache name (required for checkpoints)")
    s.add_argument("--bench", choices=list(BENCH), default="wit", help="WIT-1024 pairs or the 5,000 COCO test pairs")
    s.set_defaults(fn=wit_embed)
    s = sub.add_parser("wit", help=wit.__doc__)
    s.add_argument("pairs", nargs="+", help="name_a:name_b")
    s.add_argument("--bench", choices=list(BENCH), default="wit", help="which cached features to compare")
    s.add_argument("--subsample", type=int, default=0, help="number of random subsets for intervals")
    s.add_argument("--subset-size", type=int, default=800, help="pairs per random subset")
    s.add_argument("--metric", choices=["mknn", "cka"], default="mknn", help="mutual 10-NN or unbiased linear CKA")
    s.add_argument("--drop", type=Path, help="leave out the WIT pairs matched in this overlap file (python -m prh.data overlap)")
    s.set_defaults(fn=wit)
    s = sub.add_parser("sts", help=sts.__doc__)
    s.add_argument("models", nargs="+")
    s.set_defaults(fn=sts)
    s = sub.add_parser("unimodal", help=unimodal.__doc__)
    s.add_argument("models", nargs="+", help="checkpoints, frozen models, tfidf, random-vit-s or random-bert-s")
    s.set_defaults(fn=unimodal)
    s = sub.add_parser("nuisance", help=nuisance.__doc__)
    s.add_argument("models", nargs="+", help="text checkpoints or pretrained text models")
    s.add_argument("--image", type=Path, default=Path("runs/ref/flickr_dinov2"), help="image embeddings of the same pairs")
    s.add_argument("--k", type=int, nargs="+", default=[32, 128], help="principal directions kept")
    s.set_defaults(fn=nuisance)
    s = sub.add_parser("report", help=report.__doc__)
    s.add_argument("ckpt", type=Path, help="a text JEPA's checkpoint")
    s.add_argument("--name", required=True, help="the run's name in the results folder and in paper/runs.csv")
    s.add_argument("--results", default="runs/results", help="folder of result files read by tables.py")
    s.set_defaults(fn=report)
    results = Path("runs/results/pairfree")
    s = sub.add_parser("pairfree-embed", help=pairfree_embed.__doc__)
    s.add_argument("models", nargs="+", help="name=checkpoint or pretrained text model")
    s.set_defaults(fn=pairfree_embed)
    s = sub.add_parser("pairfree", help=pairfree_fit.__doc__)
    s.add_argument("bench", choices=["coco2014", "flickr"])
    s.add_argument("texts", nargs="+", help="text encoders: their name in data/rosetta (coco2014) or runs/ref/all_<name>")
    s.add_argument("--seeds", type=int, nargs="+", help="default: their five seeds (coco2014), or 0-4 (flickr)")
    s.add_argument("--pairs", type=int, nargs="+", default=[0], help="known pairs given to their method (0: none)")
    s.add_argument("--procrustes", nargs="+", default=[], help="paired references: Procrustes on k known pairs, or all")
    s.add_argument("--shuffle", action="store_true", help="negative control: text coordinates permuted across the pool")
    s.add_argument("--workers", type=int, default=1, help="processes for their restarts (same maps)")
    s.add_argument("--results", type=Path, default=results)
    s.set_defaults(fn=pairfree_fit)
    s = sub.add_parser("pairfree-diagnose", help=pairfree_diagnose.__doc__)
    s.add_argument("bench", choices=["coco2014", "flickr"])
    s.add_argument("text")
    s.add_argument("seed", type=int)
    s.add_argument("--results", type=Path, default=results)
    s.set_defaults(fn=pairfree_diagnose)
    for name, fn in (("video-embed", video_embed), ("video-caption", video_caption), ("video-align", video_align)):
        s = sub.add_parser(name, help=fn.__doc__)
        s.add_argument("captioner", type=Path, help="a captioner bundle (python -m prh.captioner --save)")
        s.add_argument("--out", type=Path, default=Path("runs/results/video"))
        s.set_defaults(fn=fn)
        if fn is not video_caption:
            s.add_argument("--rotation", type=Path, required=True, help="a second map (caption.py rotation.pt)")
    s = sub.choices["video-embed"]
    s.add_argument("--limit", type=int, default=0, help="smoke test: only the first test clips")
    sub.choices["video-caption"].add_argument("--score-only", action="store_true", help="score the captions already decoded")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    torch.backends.cuda.enable_cudnn_sdp(False)  # some cluster nodes cannot build cuDNN SDPA plans
    args.fn(args)


if __name__ == "__main__":
    main()
