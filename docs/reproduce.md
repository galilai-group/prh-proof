# Reproducing the paper

Every figure and table comes from five scripts and one registry of runs:

| step | script | |
|---|---|---|
| train | `train.py` | one run, on stable-pretraining: `python train.py <args>`, or `sbatch -J prh-<tag> submit.sh <args>` on SLURM |
| score | `evaluate.py` | embeddings, alignment through one rotation, the PRH-paper protocol, unimodal checks; `report` writes one run's table entries |
| caption | `caption.py`, `python -m prh.captioner` | captioner scores (BLEU-4, CIDEr); the captioner as one portable bundle |
| collect | `tables.py` | the tables that gather many runs (`phase_grid`, `phase_law_fit`, `text_sources`, `phase_real`) |
| draw | `figures.py` | every figure: `theory` (synthetic worlds), `data` (from `paper/data/*.csv`), `curves`, `pair`, `captions` |

[`paper/runs.csv`](../paper/runs.csv) lists every training run behind the paper by its tag: its exact `train.py`
arguments, the final SLURM state, the job ids and the run directory (`runs/runs/<date>/<time>/<id>/`, holding
`checkpoints/last.ckpt` and the validation embeddings). Tags ending in `-s1`, `-s2` are seeds 1 and 2; a run
cancelled and resumed (`--resume <run>/checkpoints/last.ckpt`) keeps one directory. Runs with `--devices 8 --nodes N`
need `sbatch --nodes=N --gres=gpu:8 --ntasks-per-node=8`. Each row of `paper/data/*.csv` names the run, command or
result file it comes from. All tables use **final checkpoints**; nothing is selected on test pairs (the one
exception, the best-epoch column of the robustness table, is marked as such).

Below, `<ckpt>` is `<run>/checkpoints/last.ckpt` of the run with that tag.

**Software.** The runs used stable-pretraining at revision `ab836bf` (`pip install "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@ab836bf"` to match it exactly), PyTorch 2.11, in bfloat16 mixed precision. At that revision SIGReg is computed on each GPU's batch, so its effective weight on multi-GPU runs depends on the per-GPU batch (Sec. 5 below); a later revision that gathers SIGReg across GPUs needs the weight rescaled to reproduce the 8-GPU runs.

## 0. Once: data and frozen references

Datasets: [`data.md`](data.md). Then the frozen DINOv2 embeddings every alignment is measured against:

```bash
python evaluate.py embed dinov2 runs/ref/flickr_dinov2                         # Flickr val+test: 1k-pair rotation
python evaluate.py embed dinov2 runs/ref/flickr_dinov2_all --pairs flickr-all  # Flickr train+val+test: 30k-pair rotation
for m in dinov2-s dinov2 dinov2-l dinov2-g; do python evaluate.py wit-embed $m; done   # WIT-1024 (PRH protocol)
```

## 1. Index

| paper | runs (`paper/runs.csv`) | numbers from | made by |
|---|---|---|---|
| Fig. `fig:teaser` (page 1) | left: none; right: rows of `scaling.csv`, language models of `references.csv` | `tables.py` → `phase_real.csv` (samples seen and text tokens of each run) | `figures.py data` |
| Fig. `fig:positives` | — | — | TikZ, in the paper's LaTeX source |
| Fig. `fig:dk` (Thm. 2 on random worlds) | — | `prh.theory` | `figures.py theory` |
| Fig. `fig:phase` (synthetic world: data against positives) | — | `figures.py theory` → `phase_diagram.csv` | `figures.py data` |
| Tab. `tab:main`, Fig. `fig:compare` (Flickr arms) | `image`, `text`, `text-cap`, `text-ntp`, `text-ntp-pair-s0`, and `-s1`, `-s2` | `evaluate.py align` (§2) | table typed; `figures.py curves` |
| Fig. `fig:spectrum`, `fig:tsne`, `fig:retrieval` | `image`, `text-cap` | — | `figures.py pair` |
| Fig. `fig:ablations`, Tab. `tab:sweep` | `sw-*` (seed 0), `sweep-*` (seeds 1, 2) | `evaluate.py align --csv` → `ablations.csv` | `figures.py data` |
| Fig. `fig:both_sides`, Tab. `tab:msrvtt` | `msrvtt-*` | `evaluate.py align` → `msrvtt.csv`, `msrvtt_dinov2.csv` | `figures.py data` |
| Fig. `fig:scaling` | rows of `scaling.csv` (`coco-text-cap-*`, `recap2-*`, `rt*`), `recap-image`, `recap10m-vitb`, `in1k-vits*` | `evaluate.py report` → `scaling.csv`; `evaluate.py wit` → `references.csv`, `image_side.csv` | `figures.py data` |
| Tab. `tab:calib` | Flickr arms, `text-cap-base`, `coco-text-cap-*`, rows of `scaling.csv` | `evaluate.py align`, `evaluate.py report` | typed |
| Fig. `fig:captions` | `rt500-s64-xl`, `lmdec6-q15-web2cap` | `caption.py` (§6) | `figures.py captions` |
| Tab. `tab:text_only` | `image`, `text`, `text-cap`, `text-ntp` | `evaluate.py unimodal`, `evaluate.py align --key proj` | typed |
| Tab. `tab:nuisance` | Flickr arms; `rt250-s64-long`, `rt250-masked-long`, `rt250-ntp-long`, `rt500-s64-xl`, `rtall-b64` | `evaluate.py nuisance` → `nuisance_fraction.csv` | typed |
| Tab. `tab:sources` | `rt11{full,llava,altpara}-good-s*`, `rt11alt-bad-s*`, `rt250{llava,alt}-*` | `evaluate.py report` → `tables.py` → `text_sources.csv` | typed |
| Fig. `fig:coverage`; Thm. 4 tightness (appendix) | — | `prh.theory` | `figures.py theory` |
| Tab. `tab:caption`, Fig. `fig:captioner` | `decode-n{0.5,1.0,2.0,3.0,4.0}`, `text-cap` | `caption.py` → `captioner_noise.csv`, `captioner_pairs.csv` | typed; `figures.py data` |
| Tab. `tab:captioner_scale` | `lmdec2-q15-rt500`, `lmdec4-*`, `lmdec5-q15-web`, `lmdec6-q15-web2cap` | `caption.py` (§6) | typed |
| Tab. `tab:image_encoders` | `rt250-b64-long`, `rt500-s64-xl` | `evaluate.py embed` + `align` → `image_encoders.csv` | typed |
| Tab. `tab:bigscale` | `rtall-b64` (its epoch checkpoints), `rtall-b64-s1`, `rtall-b64-l01-s*`, `rt1000-b64` | `evaluate.py report` | typed |
| Tab. `tab:wit`, `tab:llms`, `tab:coco` | rows of `scaling.csv` | `evaluate.py wit-embed` + `wit` → `references.csv`, `llm_scale.csv`, `bench_coco.csv` | typed |
| Tab. `tab:vision_families` | `rt250-*-long`, `rtall-b64` | `evaluate.py wit` → `vision_families.csv` | typed |
| Appendix: second metric | rows of `scaling.csv` | `evaluate.py wit --metric cka` → `wit_cka.csv` | typed |
| Tab. `tab:pairfree` (Sec. 4.8) and the appendix "Alignment without any pair, in detail" | `rtall-b64`, `rtall-b64-l01-s*`, `rt500-s64-xl*`, `rt250-*-long`, `rtall-masked-b64-l01`, `rtall-ntp-b64-*`, `text-cap` | `evaluate.py pairfree` (§8) | `tables.py` → `pairfree*.csv`; their numbers: `schnaus2026_reported.csv` |
| Tab. `tab:video` (MSR-VTT from frames; Sec. 4.7) | `rt500-s64-xl`, `lmdec6-q15-web2cap` | `evaluate.py video-embed`, `video-caption`, `video-align` (§7) | `tables.py` → `video.csv` |
| Fig. `fig:phase_law` (Sec. 4.6, scaling law) and its appendix details | `phase-n*-q*`, `phase-s*-q*`, `phase-rand-*`, `rt11full-good-s*` | `evaluate.py report` → `tables.py` → `phase_grid.csv`, `phase_law_fit.csv` | `figures.py data` (`phase_law.pdf`) |

## 2. Theory (Sec. 2)

```bash
pytest tests/test_theory.py                  # every statement of Sec. 2, numerically, on random finite worlds
python figures.py theory paper/data paper/figs
```

`figures.py theory` draws `davis_kahan.pdf` and `coverage.pdf`, prints the tightness of Thm. 4's bounds (appendix,
"Numerical checks"), and writes the synthetic tables `coverage_sweep.csv` and `phase_diagram.csv` (a few minutes on
a CPU). The finite worlds are `prh.theory.class_world` and `prh.theory.line_alignment`.

## 3. Flickr30k: good views, bad views, next-token prediction (Sec. 4.1, 4.2, 4.4)

The Flickr arms (200 epochs, batch 256, AdamW lr 5e-4, LeJEPA with SIGReg weight 0.02, seeds 0, 1 and 2):

| arm | tag | `train.py` arguments |
|---|---|---|
| image JEPA, ViT-S/16 | `image` | `--modality image` |
| good-view text JEPA (other captions as views) | `text-cap` | `--modality text --caption-views` |
| bad-view text JEPA (spans of one caption) | `text` | `--modality text` |
| next-token prediction | `text-ntp` | `--modality text --objective ntp` |
| next-token prediction on caption pairs | `text-ntp-pair-s0` | `--modality text --objective ntp --caption-views` |
| image JEPA, 1000 epochs | `image-long` | `--modality image --epochs 1000 --val-every 20` |

Add `--seed 1` or `--seed 2` for the other seeds. Longer or larger text runs need gradient clipping and a lower
learning rate (`--lr 1e-4 --clip 1.0`), without which a post-LN BERT eventually collapses.

```bash
python evaluate.py align <image run> <text run> --csv figs/curves/<arm>_s<seed>.csv    # tab:main: last row
c=figs/curves; python figures.py curves paper/figs/compare.pdf \
    "good-view JEPA (caption views)=$c/good_s0.csv,$c/good_s1.csv,$c/good_s2.csv" \
    "bad-view JEPA (masked spans)=$c/bad_s0.csv,$c/bad_s1.csv,$c/bad_s2.csv" \
    "next-token prediction=$c/ntp_s0.csv,$c/ntp_s1.csv,$c/ntp_s2.csv"
python figures.py pair <image run> <text-cap run> paper/figs/      # spectrum.pdf, joint.pdf, tsne.pdf, retrieval.pdf
python evaluate.py align <image-long run> <text-cap run> --last    # final checkpoints of runs of different lengths
python evaluate.py align runs/ref/flickr_dinov2 <text run>         # tab:calib, against frozen DINOv2
python evaluate.py unimodal <text-cap ckpt> <text ckpt> <text-ntp ckpt> tfidf random-bert-s <image ckpt> random-vit-s
python evaluate.py align <image run> <text run> --key proj         # tab:text_only, projector outputs
python evaluate.py nuisance <text-cap ckpt> <text ckpt> <text-ntp ckpt>          # tab:nuisance (k = 32 and 128)
python evaluate.py embed <text-ntp ckpt> runs/ref/ntp_last --last-token         # appendix: last-token pooling
python evaluate.py align runs/ref/flickr_dinov2 runs/ref/ntp_last
python evaluate.py align runs/ref/flickr_dinov2 <text run> --val-only --split 507   # appendix: no test pair at all
python evaluate.py sts <text ckpt> bert mpnet qwen0.5b                           # appendix: STS-B
```

Robustness of the good-view recipe (`fig:ablations`, `tab:sweep`): `--modality text --caption-views` plus one of
`--mask 0`, `--mask 0.3`, `--gscale 1 1 --lscale 1 1`, `--gscale 0.7 1 --lscale 0.4 0.8`, `--views 2 2`,
`--views 2 10`, `--views 4 4`, `--lamb 0.1`, `--lamb 0.05`, `--lamb 0.005`, `--lr 1e-3 --clip 1.0`, `--lr 2e-4`,
`--arch tiny`, `--epochs 100`, `--batch-size 512`, `--arch base --lr 1e-4 --clip 1.0`, each with `--seed {0,1,2}`
(tags `sw-*` and `sweep-*`); and, seed 0 only, `--lamb {0.005,0.05,0.1} --clip 1.0`, and `--views 2 2 --clip 1.0`
with `--seed {0,1,2}`. Each run: `python evaluate.py align runs/ref/flickr_dinov2 <run> --csv
runs/results/sweep/<name>.csv` (last epoch; collapse: R@1 below .05), gathered in `ablations.csv`.

## 4. Good views on both sides: MSR-VTT (Sec. 4.3)

`--data msrvtt --val msrvtt --epochs 800 --val-every 40` (the Flickr recipe with as many optimizer steps), seeds
0, 1 and 2, after `python -m prh.data frames`:

| arm | tag | `train.py` arguments |
|---|---|---|
| image JEPA, good views (each view draws its own frame of the clip) | `msrvtt-image-good` | `--modality image --good-views` |
| image JEPA, bad views (crops of one frame) | `msrvtt-image-bad` | `--modality image` |
| text JEPA, good views (each view draws its own caption of the clip) | `msrvtt-text-good` | `--modality text --good-views` |
| text JEPA, bad views (spans of one caption) | `msrvtt-text-bad` | `--modality text` |

```bash
python evaluate.py align <image run> <text run>                    # the four pairs: msrvtt.csv
python evaluate.py embed dinov2 runs/ref/msrvtt_dinov2 --pairs msrvtt
python evaluate.py align runs/ref/msrvtt_dinov2 <run>              # each run against DINOv2: msrvtt_dinov2.csv
```

## 5. Scaling the text side, and the PRH benchmark (Sec. 4.4, 4.5, appendix)

With `RT="--caption-views --lr 1e-4 --clip 1.0 --max-len 64 --epoch-size 1000000"`; bad-view JEPAs drop
`--caption-views`, next-token runs add `--objective ntp`:

| run | tags | `train.py --modality text` arguments |
|---|---|---|
| good-view JEPA, BERT-B, Flickr | `text-cap-base` | `--caption-views --arch base --lr 1e-4 --clip 1.0` |
| good-view JEPA, Flickr + COCO, BERT-T/S/B/L | `coco-text-cap-{tiny,small,base,large}` | `--caption-views --data flickr+coco --arch {tiny,small,base,large} --epochs 100 --val-every 5 --lr 1e-4 --clip 1.0` |
| Recap 2.16M images: good / bad / next-token | `recap2-text-cap`, `recap2-text`, `recap2-text-ntp` | `[--caption-views \| \| --objective ntp] --data recap:300 --epochs 10 --val-every 2 --lr 1e-4 --clip 1.0` |
| Recap 10M images, BERT-S / BERT-B | `rt50-s64`, `rt50-b64` | `$RT --data recaptext:50 --epochs 20 --val-every 2 [--arch base]` |
| Recap 50M images, BERT-S / BERT-B | `rt250-s64-long`, `rt250-b64-long` | `$RT --data recaptext:250 --epochs 40 --val-every 4 [--arch base]` |
| Recap 50M images: bad / next-token controls | `rt250-masked-long`, `rt250-ntp-long` | same, without `--caption-views` / with `--objective ntp` |
| Recap 100M images | `rt500-s64-xl` | `$RT --data recaptext:500 --epochs 80 --val-every 8` |
| Recap 200M images, BERT-B | `rt1000-b64` | see `paper/runs.csv` |
| Recap 940M images (all files), BERT-B, 8 GPUs | `rtall-b64-l01-s1`, `rtall-b64-l01-s2` | `--caption-views --lr 2e-4 --clip 1.0 --max-len 64 --epoch-size 10000000 --data recaptext:4627 --arch base --epochs 90 --val-every 4 --devices 8 --nodes 1 --batch-size 256 --shard-data --lamb 0.01` |
| Recap 940M images: bad-view JEPA | `rtall-masked-b64-l01` (`-s1`, `-s2`) | the row above without `--caption-views` |
| Recap 940M images: next-token prediction | `rtall-ntp-b64-r`, `rtall-ntp-b64-s1-low`, `rtall-ntp-b64-s2` | the row above without `--caption-views --lamb 0.01`, with `--objective ntp` |
| Flickr + COCO + Recap 10M (mixed) | `mix50-b64` | `$RT --data flickr+coco+recaptext:50 --arch base --epochs 20 --val-every 2` |

The 940M seed-0 run (`rtall-b64`) used 2 nodes x 8 GPUs x 128 (`--nodes 2 --batch-size 128`, SIGReg weight 0.02).
stable-pretraining's SIGReg is computed on each GPU's batch and scaled by its size, so its effective weight is
proportional to the per-GPU batch: `--lamb 0.01` at 256 per GPU gives seed 0's effective weight (`rtall-b64-s1`:
0.02 at 256, twice that weight; paper, appendix). `--shard-data` gives each rank its own files.

Image side: `recap-image` (ViT-S, `--modality image --data recap:300 --epochs 10 --val-every 2`), `recap10m-vitb`
(`--data recap:1400 --arch base --devices 8 --batch-size 128 --lr 2e-3 --epochs 8 --val-every 2`), `in1k-vits`
(`--data imagenet --devices 8 --batch-size 128 --lr 2e-3 --epochs 300 --val-every 10`).

```bash
python evaluate.py report <ckpt> --name <tag>      # Flickr 30k-pair R@1 and WIT with DINOv2 S/B/L/g: runs/results/<tag>.txt
python evaluate.py report <rtall-b64 run>/checkpoints/epoch=<N>.ckpt --name rtall-b64-e<N>   # tab:bigscale
# language models and other references (references.csv, llm_scale.csv): the last layer through one rotation, and WIT
python evaluate.py embed <qwen0.5b|bert|mpnet|clip> runs/ref/all_<model> --pairs flickr-all
python evaluate.py align runs/ref/flickr_dinov2_all runs/ref/all_<model>
python evaluate.py wit-embed <qwen0.5b ... qwen72b|llama3-8b|llama3-70b|mistral7b|bert|mpnet|clip>
python evaluate.py wit-embed <ckpt> --name text-<tag>
python evaluate.py wit dinov2:text-<tag> dinov2:qwen7b ...                          # tab:wit
python evaluate.py wit dinov2:text-<tag> dinov2:qwen7b dinov2:qwen3b --subsample 200   # 95% intervals (appendix)
python evaluate.py wit dinov2:text-<tag> ... --metric cka                           # second metric: wit_cka.csv
python evaluate.py wit-embed <model> --bench coco && python evaluate.py wit <pairs> --bench coco   # tab:coco
python evaluate.py wit-embed <mae-b|mae-l|mae-h|ijepa-h|clip-b|clip-l|clip-h>       # tab:vision_families (clip-*: CLIP's image encoders)
python evaluate.py wit-embed <image ckpt> --name image-jepa-<tag>                   # image JEPAs: image_side.csv
python -m prh.data overlap && python evaluate.py wit <pairs> --drop runs/results/overlap.json   # WIT captions, Flickr30k and COCO photos in Recap
```

COCO transfer (rotation fitted on Flickr pairs): `evaluate.py embed <model> runs/ref/coco5k_<name> --pairs coco5k`
for both sides, then `evaluate.py align runs/ref/flickr_dinov2_all runs/ref/all_<name> --test-a
runs/ref/coco5k_dinov2 --test-b runs/ref/coco5k_<name>`; in-domain reference: `evaluate.py align
runs/ref/coco5k_dinov2 runs/ref/coco5k_<name> --split 2500`. Frozen image encoders (`tab:image_encoders`): for each
of `dinov2-l dinov2-g dinov2-g+avg dinov3-l dinov3-l+avg dinov3-h dinov3-h+avg dinov3-7b dinov3-7b+avg ijepa-h
ijepa-g` and the concatenations `dinov2-g+avg,dinov3-l+avg` and `dinov2-g+avg,dinov3-7b+avg`: `evaluate.py embed
<model> runs/ref/flickr_<model>_all --pairs flickr-all`, then `evaluate.py align runs/ref/flickr_<model>_all
runs/ref/all_<tag>`.

## 6. Which positives, which data: text sources, phase grid and scaling law (Sec. 2, 4.6, appendix)

Text sources (`tab:sources`; BERT-S, lr 1e-4, clipping 1, 64 tokens, seeds 0, 1, 2): `--data recaptext:11`
(all three texts), `recaptext-llava:11` (the two LLaVA recaptions), `recaptext-alt:11` (the alt-text alone) and
`recaptext-altpara:11` (the alt-text and a text-only paraphrase, written beforehand by `python -m prh.data
paraphrase <file index>` for files 0 to 10), with 20 epochs of 1M samples; the same at `:250` with 40 epochs. Exact
arguments: tags `rt11*` and `rt250*` in `paper/runs.csv`.

Real phase grid (`phase-*`): BERT-S, `--modality text --good-views <q> --data recaptext:<files> --epoch-size 1000000
--epochs <M> --lr 1e-4 --clip 1.0 --max-len 64`, with a fraction `q` of training items drawing good views and the rest
bad views; `files` in {2, 11, 50, 250} (0.41M to 50.9M unique images) and `q` in {0, .001, .003, .01, .03, .1, .3, 1}
at 20M samples (tags `phase-n<files>-q<q>`), `q` in {0, .01, .1, .3, 1} at 5M, 40M and 80M samples on 10.2M
images (`phase-s<M>-q<q>`), 25 runs at random `q` and `M` on 10.2M images (`phase-rand-q<q>-s<M>`, design and
predictions registered before training in `phase_law_predictions.csv`), and `q` in {0, .1, 1} at 320M samples.

```bash
python evaluate.py report <ckpt> --name <tag>   # every run above
python tables.py paper/data                     # phase_grid.csv, phase_law_fit.csv, text_sources.csv, phase_real.csv
python evaluate.py nuisance <ckpt> ...          # the scene-explained column of tab:sources
```

`tables.py` fits the scaling law on the grid's 5M- and 20M-sample cells (v1) and on all its cells up to 80M samples
(v2), and prints every run a fit did not see against the prediction registered before that run trained
(`phase_law_predictions.csv`).

## 7. Captioning (Sec. 4.7, appendix)

Caption decoders (`train.py --modality text --objective decode`; `--encoder` is the frozen text JEPA's checkpoint).
Every decoder sees text only: `--data` names caption corpora (`prh.data.Captions`), never images, and `--noise` is
matched to the residual of the rotation it will decode through (`prh.metrics.matched_noise`).

| decoder | tags | arguments |
|---|---|---|
| Flickr, noise sweep (`fig:captioner`) | `decode-n{0.5,1.0,2.0,3.0,4.0}` | `--encoder <text-cap ckpt> --epochs 40 --val-every 5 --lr 3e-4 --clip 1.0 --noise {0.5,1,2,3,4}` |
| Flickr + COCO captions (COCO transfer) | `decode-coco-n2.0` | `--encoder <coco-text-cap-small ckpt> --data flickr+coco --epochs 20 --val-every 5 --lr 3e-4 --clip 1.0 --noise 2` |
| language-model decoder (captions as written; `tab:captioner_scale`) | `lmdec4-q15-coco`, `lmdec4-q3-coco` | `--encoder <rt500-s64-xl ckpt> --lm Qwen/Qwen2.5-1.5B --data flickr+coco --epoch-size 350000 --epochs 4 --val-every 2 --lr 3e-5 --clip 1.0 --noise 1.9` (Qwen2.5-3B: `--lm Qwen/Qwen2.5-3B --lr 2e-5 --batch-size 128`) |
| the demo's decoder: web sentences, then human captions | `lmdec5-q15-web`, then `lmdec6-q15-web2cap` | `--encoder <same> --lm Qwen/Qwen2.5-1.5B --data recapsent:25 --epoch-size 1000000 --epochs 4 --val-every 2 --lr 3e-5 --clip 1.0 --noise 1.9`, then the row above with `--init <that run's ckpt>` |

```bash
python evaluate.py embed <text-cap ckpt> runs/ref/all_text-cap --pairs flickr-all
python caption.py --decoder <decode-n3.0 run> --image runs/ref/flickr_dinov2 --text <text-cap run> --pairs 1014 300 100 --out figs/captions
python caption.py --decoder <decode-n3.0 run> --image runs/ref/flickr_dinov2 --text <text-cap run> --pairs 100 30 --refine --out figs/captions_refine
python caption.py --decoder <decode-n3.0 run> --image runs/ref/flickr_dinov2_all --text runs/ref/all_text-cap --pairs 30014 --out figs/captions_30k
python caption.py --decoder <decode-coco-n2.0 run> --image runs/ref/flickr_dinov2_all --text runs/ref/all_<coco-text-cap-small> --pairs 30014 \
    --test coco5k --test-image runs/ref/coco5k_dinov2 --test-text runs/ref/coco5k_<coco-text-cap-small> --beam 3 --out figs/captions_coco5k
```

The decoder noise matches the residual cosine `c` of the rotation, `σ = (c^-2 - 1)^(1/2)` (`prh.metrics.matched_noise`):
`evaluate.py align` prints `c` in its `cos` column, on test pairs, and on held-out validation pairs with
`--val-only --split 507` (30k-pair rotation: `align runs/ref/flickr_dinov2_all runs/ref/all_<name> --val-only --split 15007`).

Matching the two clouds with no pair at all (Gromov-Wasserstein, at chance): `prh.align.Rotation.unpaired`.

The strongest captioner (`tab:captioner_scale`, `fig:captions`): two frozen image encoders, the text JEPA trained on
the texts of 100M web images, the web-pretrained language-model decoder, and the consensus of 32 samples.

```bash
M=dinov2-g+avg,dinov3-l+avg; T=<rt500-s64-xl run>; D=<lmdec6-q15-web2cap run>
python evaluate.py embed $M runs/ref/flickr_gl_all --pairs flickr-all && python evaluate.py embed $M runs/ref/coco5k_gl --pairs coco5k
python evaluate.py embed $T/checkpoints/last.ckpt runs/ref/all_T --pairs flickr-all && python evaluate.py embed $T/checkpoints/last.ckpt runs/ref/coco5k_T --pairs coco5k
python caption.py --decoder $D --image runs/ref/flickr_gl_all --text runs/ref/all_T --pairs 30014 1014 --beam 1 --samples 32 --out figs/cap_flickr
python caption.py --decoder $D --image runs/ref/flickr_gl_all --text runs/ref/all_T --pairs 30014 --beam 1 --samples 32 \
    --test coco5k --test-image runs/ref/coco5k_gl --test-text runs/ref/coco5k_T --out figs/cap_coco
python figures.py captions figs/cap_coco --picks 0 7 21 99 --out paper/figs/captions_coco.pdf
```

**From photos to videos (`tab:video`).** The same captioner on the 2,990 MSR-VTT test clips, from the mean of 8
frames, with no network trained on video:

```bash
python -m prh.data frames                                   # 8 frames per MSR-VTT clip (pip install -e ".[video]")
python -m prh.captioner --decoder $D --rotation figs/cap_demo/rotation.pt --image-model $M --beam 1 --samples 32 \
    --pairs 230014 --save captioner.pt                      # the captioner as one file (rotation on 230k pairs)
python evaluate.py video-embed captioner.pt --rotation figs/cap_flickr/rotation.pt   # clips and captions, both maps
python evaluate.py video-caption captioner.pt               # CIDEr, BLEU-4, METEOR (needs java), ROUGE-L
python evaluate.py video-align captioner.pt --rotation figs/cap_flickr/rotation.pt   # JSFUSION 1k-A retrieval, mutual 10-NN
```

## 8. Alignment without any pair (Sec. 4.8, appendix)

The method of Schnaus et al. (arXiv 2610.09411), their code unchanged, on our text encoders:

```bash
pip install -e ".[pairfree]"                                # their package at a pinned commit, pylibmgm 1.1.2
python -m prh.data rosetta                                  # their stored embeddings (37 GB; `rosetta no-qwen`: 1.7 GB)
python evaluate.py pairfree-embed rtall-final=<rtall-b64 ckpt> rt500-final=<rt500-s64-xl ckpt> ... clip=clip   # GPU
python evaluate.py pairfree coco2014 <names> --procrustes all          # tab:pairfree: their protocol, their 5 seeds
python evaluate.py pairfree flickr <names> --procrustes 10 30 100 all  # Flickr, disjoint unpaired halves
python evaluate.py pairfree flickr rtall-final clip --seeds 0 --pairs 10           # given 10 known pairs
python evaluate.py pairfree coco2014 rt500-final --shuffle --seeds 734796314 576165995   # negative control
python evaluate.py pairfree-diagnose flickr rtall-final 1                          # why some seeds fail
python caption.py --decoder <lmdec4-q15-coco run> --image runs/ref/flickr_dinov2_all --text runs/ref/all_rt500-final \
    --pairs 30014 --maps runs/results/pairfree/maps/flickr_rt500-final_wp0_s{0..4}.pt   # captions with zero pairs
python tables.py paper/data                                 # pairfree.csv, _selection, _tests, _captioner, _diagnosis
```

Each fit takes 2–3 CPU-hours on one thread (`--workers N` runs the restarts in parallel, with the same map). Nothing
from a test or validation pair, and no label, enters a fit, its centering means or its seed selection, which uses
the method's own objective on the unpaired training pools.

## 9. All figures at once

```bash
python figures.py theory paper/data paper/figs   # davis_kahan, coverage (and the synthetic tables)
python tables.py paper/data                      # tables gathered from runs/results
python figures.py data paper/data paper/figs     # teaser, scaling, both_sides, captioner, ablations, coverage, phase, phase_law
```

plus `figures.py curves`, `pair` and `captions` above, which read runs and caption folders rather than
`paper/data`.
