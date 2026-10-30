"""Zero-pair captioning: image -> frozen image encoder -> one rotation -> text-JEPA space -> text-only decoder.

No network is trained on image-caption pairs: pairs only fit the rotation (``--pairs`` of the non-test pairs of
the embeddings). Writes ``scores.csv`` (BLEU-4, CIDEr per method), ``captions.json``, ``examples.png`` and the
rotation fitted on the first ``--pairs`` value (``rotation.pt``, for a captioner bundle: ``python -m prh.captioner``).

    python caption.py --decoder <decode run> --image runs/ref/flickr_dinov2 --text runs/ref/<name> --out figs/captions
    python caption.py ... --test coco5k --test-image runs/ref/coco5k_dinov2 --test-text runs/ref/coco5k_<name> --beam 3
    python caption.py ... --pairs 30014 --maps runs/results/pairfree/maps/flickr_<name>_wp0_s*.pt   # no pair at all
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from prh import data, pairfree, plots
from prh.align import Rotation, project
from prh.captioner import checkpoint
from prh.features import last_embeddings
from prh.metrics import caption_scores, consensus
from prh.models import TextEncoder, load_decoder, rerank, tokenizer


def supervised_maps(a, b, mu_a, mu_b, ridge=1e-3, hidden=2048, steps=4000, seed=0):
    """Baselines trained on the same pairs as the rotation: any linear map (ridge) and a two-layer MLP.

    The MLP regresses standardized text embeddings with MSE and keeps its best step on 10% of the pairs held out.
    Returns ``{name: map}``; each map sends image embeddings to the text space, as the rotation does.
    """
    x, y = a - mu_a, b - mu_b
    gram = x.T @ x
    W = torch.linalg.solve(gram + ridge * gram.diagonal().mean() * torch.eye(len(gram), dtype=x.dtype), x.T @ y)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    held, fit = perm[: len(x) // 10], perm[len(x) // 10 :]
    sx, sy = x[fit].std(0) + 1e-6, y[fit].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = (x / sx).float().to(dev), (y / sy).float().to(dev)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state = float("inf"), None
    for step in range(steps):
        i = fit[torch.randint(len(fit), (512,), generator=g)].to(dev)
        loss = torch.nn.functional.mse_loss(net(X[i]), Y[i])
        opt.zero_grad(), loss.backward(), opt.step()
        if step % 100 == 99:
            with torch.no_grad():
                held_loss = torch.nn.functional.mse_loss(net(X[held.to(dev)]), Y[held.to(dev)]).item()
            if held_loss < best:
                best, state = held_loss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)

    def mlp(z):
        with torch.no_grad():
            return (net(((z - mu_a) / sx).float().to(dev)).cpu().to(z.dtype)) * sy + mu_b

    return {"linear map, any matrix": lambda z: (z - mu_a) @ W + mu_b, "two-layer MLP": mlp}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder", type=Path, required=True, help="decode run (train.py --objective decode) or its checkpoint")
    p.add_argument("--image", type=Path, required=True, help="image embeddings of the Flickr pairs (evaluate.py embed)")
    p.add_argument("--text", type=Path, required=True, help="text-JEPA embeddings of the same pairs")
    p.add_argument("--pairs", type=int, nargs="+", default=[1014, 300, 100], help="pairs that fit the rotation")
    p.add_argument("--refine", action="store_true", help="refine few-pair rotations on the unpaired non-test embeddings")
    p.add_argument("--beam", type=int, default=1, help="beam width (1: greedy)")
    p.add_argument(
        "--rerank", type=int, default=0, help="of this many beam candidates, keep the one the decoder's text JEPA embeds closest"
    )
    p.add_argument(
        "--samples", type=int, default=0, help="LLM decoders: add this many samples to the n-best beams, keep their consensus"
    )
    p.add_argument(
        "--project", type=float, default=0, help="temperature: decode the rotated embedding projected on the non-test texts"
    )
    p.add_argument("--web-pairs", type=Path, help="evaluate.py embed-web output: the saved rotation also fits these pairs")
    p.add_argument(
        "--supervised", action="store_true", help="baselines: a linear map and an MLP trained on the first --pairs value"
    )
    p.add_argument(
        "--maps", type=Path, nargs="+", default=[], help="zero-pair maps (evaluate.py pairfree flickr): decode through each"
    )
    p.add_argument("--test", choices=["flickr", "coco5k"], default="flickr", help="images to caption")
    p.add_argument("--test-image", type=Path, help="coco5k: image embeddings of the test images")
    p.add_argument("--test-text", type=Path, help="coco5k: text-JEPA embeddings of their first captions (oracle)")
    p.add_argument("--out", type=Path, default=Path("figs/captions"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = load_decoder(checkpoint(args.decoder)).to(device)
    if args.rerank:  # the frozen text JEPA the decoder was trained on
        hp = torch.load(checkpoint(args.decoder), map_location="cpu", weights_only=False, mmap=True)["hyper_parameters"]
        enc = TextEncoder.from_checkpoint(hp["encoder"], tokenizer().pad_token_id).to(device)
    (a, pid, is_test), (b, text_id, _) = last_embeddings(args.image), last_embeddings(args.text)
    assert torch.equal(pid, text_id), "the image and text embeddings are of different pairs"
    va, vb, ta, tb, test_id = a[~is_test], b[~is_test], a[is_test], b[is_test], pid[is_test]
    if args.test == "coco5k":  # rotation from Flickr pairs, captions of COCO images
        (ta, test_id, _), (tb, text_id, _) = last_embeddings(args.test_image), last_embeddings(args.test_text)
        assert torch.equal(test_id, text_id)
    table = (data.coco5k_pairs() if args.test == "coco5k" else data.flickr_pairs()).set_index("pair_id").loc[test_id.numpy()]
    refs = table.captions.tolist()

    mu_a, mu_b = va.mean(0), vb.mean(0)  # means need no pairs

    def decode(z):
        out = []
        for chunk in z.split(256):
            if args.project:  # onto the cloud of non-test text embeddings (no images involved)
                chunk = project(chunk, vb, mu_b, args.project)
            c = dec.condition(chunk.to(device))
            if args.rerank:
                out += rerank(dec.generate(c, beam=max(args.beam, args.rerank), nbest=args.rerank), chunk, enc, mu_b)
            elif args.samples:  # minimum-Bayes-risk decoding (prh.metrics.consensus)
                cands, k = dec.generate(c, beam=args.beam, nbest=args.beam, samples=args.samples), args.beam + args.samples
                out += [consensus(cands[i : i + k]) for i in range(0, len(cands), k)]
            else:
                out += dec.generate(c, beam=args.beam)
        return out

    captions = {"oracle (true text embedding)": decode(tb)}
    for n in args.pairs:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(va), generator=g)[:n]
        rot = Rotation.fit(va[idx], vb[idx], mu_a, mu_b)
        if args.refine and n < len(va):  # the non-test texts enter shuffled: only their distribution is used
            rot = rot.refine(va, vb[torch.randperm(len(vb), generator=g)])
        if n == args.pairs[0] and not args.web_pairs:
            rot.save(args.out / "rotation.pt")
        captions[f"zero-pair captioner, rotation from {n} pairs"] = decode(rot(ta))
        if args.supervised and n == args.pairs[0]:
            for name, fmap in supervised_maps(va[idx], vb[idx], mu_a, mu_b).items():
                captions[f"supervised baseline: {name} trained on {n} pairs"] = decode(fmap(ta))
    if args.web_pairs:  # the demo's map: the Flickr pairs and web pairs, pooled
        web = torch.load(args.web_pairs)
        rot = Rotation.fit_pooled([(va, vb), (web["image"].double(), web["text"].double())])
        rot.save(args.out / "rotation.pt")
        captions[f"zero-pair captioner, rotation from {len(va)} Flickr and {len(web['image'])} web pairs"] = decode(rot(ta))
    random = torch.linalg.qr(torch.randn(va.shape[1], vb.shape[1], dtype=va.dtype))[0]
    captions["random rotation"] = decode(Rotation(random, mu_a, mu_b)(ta))
    rot = Rotation.fit(va, vb, mu_a, mu_b)
    nearest = torch.nn.functional.normalize(rot(ta) - mu_b, dim=1) @ torch.nn.functional.normalize(vb - mu_b, dim=1).T
    train_caps = data.flickr_pairs(("train", "val")).set_index("pair_id").loc[pid[~is_test].numpy()].captions
    captions[f"retrieve nearest caption ({len(va)} pairs)"] = [train_caps.iloc[i][0] for i in nearest.argmax(1).tolist()]
    scale = (vb - mu_b).norm(dim=1).median()
    for path in args.maps:  # a zero-pair map lands on the unit sphere: give it a text embedding's typical norm and mean
        m = pairfree.Map.load(path)
        captions[f"zero-pair map ({path.stem})"] = decode(
            torch.nn.functional.normalize(m.transform(ta), dim=1) * scale + m.mean_y[0]
        )
    rows = [
        {"method": name, **{k: round(v, 3) for k, v in caption_scores(preds, refs).items()}} for name, preds in captions.items()
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(args.out / "scores.csv", index=False)
    json.dump({"files": [str(f) for f in table.path], "refs": refs, "captions": captions}, open(args.out / "captions.json", "w"))
    plots.caption_grid(
        table.path.tolist(),
        refs,
        captions[f"zero-pair captioner, rotation from {args.pairs[0]} pairs"],
        args.out / "examples.png",
    )


if __name__ == "__main__":
    main()
