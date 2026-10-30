"""Train one arm with stable-pretraining: a per-modality JEPA, a next-token model, or a text-only caption decoder.

All arms share stable-pretraining's ``LeJEPA`` (invariance to the global views' mean, plus SIGReg), projector,
optimizer, schedule, seeding, ``spt.Manager`` (loops, checkpoints, SLURM requeue) and callbacks (``OnlineWriter``
dumps validation embeddings, ``RankMe``). They differ only in the backbone (a ViT, or a BERT of the same width),
the views and the objective. No arm ever sees the other modality; pairs are used only offline (``evaluate.py``).

    python train.py --modality image                                   # image JEPA (Flickr30k)
    python train.py --modality text --caption-views                    # good text JEPA: other captions as views
    python train.py --modality text                                    # bad text JEPA: spans of one caption
    python train.py --modality text --objective ntp                    # next-token prediction
    python train.py --modality text --caption-views --data recaptext:50 --epoch-size 1000000   # web-scale text
    python train.py --modality text --objective decode --encoder <ckpt>   # caption decoder on a frozen text JEPA
    python train.py --modality text --objective decode --encoder <ckpt> --lm Qwen/Qwen2.5-1.5B --lr 3e-5   # same, LLM decoder

Every run validates on the val and test pairs of Flickr30k (or MSR-VTT, ``--val msrvtt``), whatever it trains on. ``docs/reproduce.md`` lists the
exact command of every result in the paper.
"""

import argparse
import os
from pathlib import Path

import stable_pretraining as spt
import torch
from stable_pretraining.backbone.utils import EvalOnly
from stable_pretraining.methods.lejepa import LeJEPA
from transformers import AutoTokenizer

from prh import data as prh_data
from prh import forward
from prh.models import ARCH, TOKENIZER, CaptionDecoder, LMDecoder, TextEncoder, load_decoder
from prh.views import view_transforms


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=["image", "text"], required=True)
    p.add_argument(
        "--objective",
        choices=["lejepa", "ntp", "decode"],
        default="lejepa",
        help="ntp: next-token prediction; decode: caption decoder on a frozen text JEPA (--encoder)",
    )
    p.add_argument(
        "--good-views",
        "--caption-views",
        dest="caption_views",
        type=float,
        nargs="?",
        const=1.0,
        default=0.0,
        metavar="Q",
        help="views draw different captions of the image, or different frames of the clip (good views), for a fraction "
        "Q of the training items (default 1 with the flag); ntp: caption pairs",
    )
    p.add_argument(
        "--data",
        default="flickr",
        help="training data, '+'-joined: flickr, coco (text), "
        "recaptext:<n_files> (text), recap:<n_shards>, msrvtt, or imagenet (image, alone)",
    )
    p.add_argument("--val", choices=["flickr", "msrvtt"], default="flickr", help="validation pairs (val and test splits)")
    p.add_argument("--arch", choices=list(ARCH), default="small", help="ViT-{T,S,B,L}/16, or a BERT of the same width")
    p.add_argument("--views", type=int, nargs=2, default=[2, 6], help="numbers of global and local views")
    p.add_argument("--gscale", type=float, nargs=2, default=[0.5, 1.0], help="text: kept fraction of global views")
    p.add_argument("--lscale", type=float, nargs=2, default=[0.15, 0.5], help="text: kept fraction of local views")
    p.add_argument("--mask", type=float, default=0.15, help="text: masked fraction of view tokens")
    p.add_argument("--max-len", type=int, default=64, help="text: tokens per training view")
    p.add_argument("--encoder", type=Path, help="decode: checkpoint of the frozen text JEPA")
    p.add_argument("--noise", type=float, default=1.0, help="decode: relative noise on the conditioning embedding")
    p.add_argument("--lm", help="decode: fine-tune this pretrained causal language model (HuggingFace name) as the decoder")
    p.add_argument("--init", type=Path, help="decode: start from the decoder of this decode run's checkpoint")
    p.add_argument("--resume", type=Path, help="continue from this checkpoint's full state in a new run (e.g. another queue)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--epoch-size", type=int, default=0, help="samples per epoch, drawn at random (0: the whole dataset)")
    p.add_argument("--steps", type=int, default=-1, help="cap on optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=256, help="per GPU")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lamb", type=float, default=0.02, help="SIGReg weight")
    p.add_argument("--clip", type=float, default=0.0, help="gradient-norm clipping (0: off)")
    p.add_argument("--val-every", type=int, default=10, help="epochs between validations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--devices", type=int, default=1, help="GPUs per node (DDP)")
    p.add_argument("--nodes", type=int, default=1, help="nodes (DDP)")
    p.add_argument("--shard-data", action="store_true", help="recaptext: each GPU loads only its share of the files")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = p.parse_args(argv)
    # The sanity pass dumps the random initialization as epoch 0; real validations land on epochs
    # val_every - 1, 2 * val_every - 1, ... and must not overwrite it.
    if args.val_every < 2:
        p.error("--val-every must be at least 2 (epoch 0 holds the random initialization)")
    if not 0 <= args.caption_views <= 1:
        p.error("--good-views takes a fraction in [0, 1]")
    if args.objective != "lejepa" and args.modality != "text":
        p.error(f"--objective {args.objective} is text only")
    if (args.objective == "decode") != (args.encoder is not None) or ((args.lm or args.init) and args.objective != "decode"):
        p.error("--encoder (and optionally --lm, --init) go with --objective decode, and only with it")
    return args


def _shard():
    """``(rank, world size)`` of this process, from the launcher's environment (srun or torchrun)."""
    env = os.environ
    return int(env.get("SLURM_PROCID", env.get("RANK", 0))), int(env.get("SLURM_NTASKS", env.get("WORLD_SIZE", 1)))


def build_data(args, tok):
    """Training set of ``--data`` and the Flickr30k val+test pairs, with the arm's views."""
    train_tf, val_tf, gpu_tf = view_transforms(
        args.modality, tok, args.objective, args.caption_views, args.views, args.gscale, args.lscale, args.mask, args.max_len
    )

    def loader(ds, train):
        sampler = prh_data.EpochSampler(len(ds), args.epoch_size) if train and args.epoch_size else None
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            drop_last=train,
            pin_memory=True,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )

    if args.objective == "decode":  # captions as written: the language model's targets keep their case
        train = spt.data.FromTorchDataset(prh_data.Captions(args.data), ["caption"], transform=train_tf)
    elif args.data == "imagenet":  # labels are carried along but never used
        train = spt.data.HFDataset(str(prh_data.IMAGENET / "train"), transform=train_tf, gpu_transform=gpu_tf)
    else:
        train = spt.data.FromTorchDataset(
            prh_data.training_set(args.data, args.modality, args.caption_views, _shard() if args.shard_data else (0, 1)),
            [args.modality, "pair_id", "is_test"],
            transform=train_tf,
            gpu_transform=gpu_tf,
        )
    pairs = prh_data.MSRVTTClips if args.val == "msrvtt" else prh_data.Flickr30k
    val = spt.data.FromTorchDataset(
        pairs(["val", "test"], args.modality, False), [args.modality, "pair_id", "is_test"], transform=val_tf
    )
    return spt.data.DataModule(train=loader(train, True), val=loader(val, False))


def build_networks(args, tok):
    """``(networks, forward function, embedding width)`` of the arm."""
    if args.objective == "ntp":
        return {"backbone": TextEncoder(tok.pad_token_id, causal=True, arch=args.arch)}, forward.next_token, ARCH[args.arch][0]
    if args.objective == "decode":  # --arch sizes the decoder; the frozen encoder's size comes from its checkpoint
        enc = TextEncoder.from_checkpoint(args.encoder, tok.pad_token_id)
        dec = LMDecoder(args.lm, enc.embed_dim) if args.lm else CaptionDecoder(tok.pad_token_id, enc.embed_dim, args.arch)
        if args.init:  # e.g. fine-tune a decoder pretrained on web sentences on human captions
            dec.load_state_dict(load_decoder(args.init).state_dict())
        return {"decoder": dec, "encoder": EvalOnly(enc), "noise": args.noise}, forward.decode, enc.embed_dim
    model = LeJEPA(f"vit_{args.arch}_patch16_224", lamb=args.lamb)
    if args.modality == "text":
        model.backbone = TextEncoder(tok.pad_token_id, arch=args.arch)
    return {"model": model}, forward.lejepa, ARCH[args.arch][0]


def main(argv=None):
    args = parse_args(argv)
    # Spot jobs are preempted every few minutes: also save last.ckpt mid-epoch.
    spt.set(cache_dir=str(args.cache_dir.resolve()), requeue_checkpoint_every_n_steps=250)
    spt.make_it_fast()
    torch.backends.cuda.enable_cudnn_sdp(False)  # some cluster nodes cannot build cuDNN SDPA plans
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    data = build_data(args, tok)
    nets, fwd, emb_dim = build_networks(args, tok)
    total = len(data.train) // (args.devices * args.nodes) * args.epochs if args.steps < 0 else args.steps
    module = spt.Module(
        **nets,
        key=args.modality,
        forward=fwd,
        n_global=args.views[0],
        hparams={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        optim={
            # Lightning cannot clip the gradients of the fused AdamW that make_it_fast selects.
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
                **({"fused": False} if args.clip > 0 else {}),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": min(10 / args.epochs, 0.05),
                "start_factor": 0.01,
                "end_lr": args.lr / 1000,
                "total_steps": total,
            },
            "interval": "step",
        },
    )
    # A config dict (not a pl.Trainer) lets Manager set default_root_dir to the run dir, so that
    # OnlineWriter's relative path is per run.
    trainer = {
        "_target_": "lightning.pytorch.Trainer",
        "devices": args.devices,
        "num_nodes": args.nodes,
        "log_every_n_steps": 10,
        **({"strategy": "ddp", "sync_batchnorm": True} if args.devices * args.nodes > 1 else {}),
        "max_epochs": args.epochs,
        "max_steps": args.steps,
        "check_val_every_n_epoch": args.val_every,
        "num_sanity_val_steps": -1,
        "precision": "bf16-mixed" if torch.cuda.is_available() else "32",
        **({"gradient_clip_val": args.clip} if args.clip > 0 else {}),
        "callbacks": [
            {
                "_target_": "stable_pretraining.callbacks.OnlineWriter",
                "path": "embeddings",
                "during": "validation",
                "names": ["embedding", "proj", "pair_id", "is_test"],
                "save_sanity_check": True,
                "all_gather": False,
            },
            {
                "_target_": "stable_pretraining.callbacks.RankMe",
                "name": "rankme",
                "target": "embedding",
                "queue_length": 2048,
                "target_shape": emb_dim,
            },
        ],
    }
    resume = {"ckpt_path": str(args.resume.resolve()), "weights_only": False} if args.resume else {}
    spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed, **resume)()


if __name__ == "__main__":
    main()
