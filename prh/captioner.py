"""The paper's captioner as one portable bundle: frozen image encoder(s) -> one rotation -> a decoder trained on text only.

    python -m prh.captioner --decoder <decode run> --rotation <caption.py --out>/rotation.pt \
        --image-model dinov2-g+avg,dinov3-l+avg --beam 1 --samples 32 --pairs 230014 --save captioner.pt
    python -m prh.captioner --captioner captioner.pt photo1.jpg photo2.jpg

A bundle holds everything inference needs (the image encoders' names, the rotation, the text JEPA and the decoder's
weights in bfloat16) and loads without the training runs. No network in it was trained on an image-caption pair:
pairs only fitted the rotation (``caption.py``).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .align import Rotation
from .features import DEVICE, embed_images
from .metrics import consensus
from .models import TextEncoder, arch_of_width, build_decoder, load_decoder, tokenizer

#: precision of the vision encoders (SEE) and of the decoder (SAY): bfloat16 on GPUs from Ampere on; on older GPUs (a T4,
#: without fast bfloat16) the decoder runs in float16 and the vision encoders in float32, as DINOv2-g overflows in float16
AMPERE = DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8
SEE = torch.bfloat16 if AMPERE else torch.float32
SAY = torch.bfloat16 if AMPERE else torch.float16 if DEVICE == "cuda" else torch.float32

IMAGE_NAMES = {
    "dinov2-s": "DINOv2 ViT-S/14",
    "dinov2": "DINOv2 ViT-B/14",
    "dinov2-l": "DINOv2 ViT-L/14",
    "dinov2-g": "DINOv2 ViT-g/14",
    "dinov3-l": "DINOv3 ViT-L/16",
    "dinov3-h": "DINOv3 ViT-H+/16",
    "dinov3-7b": "DINOv3 ViT-7B/16",
}
IMAGE_NAMES |= {f"{k}+avg": v for k, v in IMAGE_NAMES.items()}


def photo(image) -> Image.Image:
    """An RGB image as a viewer sees it: upright per its EXIF orientation (phone photos), transparency on white."""
    image = ImageOps.exif_transpose(Image.open(image) if isinstance(image, (str, Path)) else image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
    return image.convert("RGB")


class Captioner:
    """Image(s) -> frozen image encoder -> rotation -> text-only decoder -> caption(s); description -> text JEPA.

    Args:
        decoder: a :class:`prh.models.CaptionDecoder` or :class:`prh.models.LMDecoder`.
        rotation: the map from the image encoder's space to the text JEPA's.
        image_model: the image encoder(s) the rotation was fitted on (``prh.features.embed_images``).
        beam: beam width.
        pairs: number of image-text pairs that fitted the rotation (shown in the app).
        samples: LLM decoders: add this many samples to the ``beam`` best beams and keep their consensus.
        text: the text JEPA the decoder reads, to search photos by description (optional).
    """

    def __init__(
        self, decoder, rotation: Rotation, image_model: str = "dinov2", beam: int = 3, pairs: int = 0, samples: int = 0, text=None
    ):
        self.dec, self.rot, self.image_model, self.beam, self.pairs = decoder.to(DEVICE), rotation, image_model, beam, pairs
        self.samples, self.text = samples, None if text is None else text.to(DEVICE).eval()
        if DEVICE == "cuda" and hasattr(decoder, "lm"):  # half the memory; a saved bundle holds bfloat16 weights anyway
            decoder.lm.to(SAY)

    @classmethod
    def from_runs(cls, decoder: Path, rotation: Path, image_model: str = "dinov2", **kw) -> "Captioner":
        """From a ``train.py --objective decode`` run and a ``caption.py`` rotation; the run names its text JEPA."""
        ckpt = checkpoint(decoder)
        hp = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True).get("hyper_parameters") or {}
        text = TextEncoder.from_checkpoint(hp["encoder"], tokenizer().pad_token_id) if hp.get("encoder") else None
        return cls(load_decoder(ckpt), Rotation.load(rotation), image_model, text=text, **kw)

    @classmethod
    def load(cls, path: str | Path) -> "Captioner":
        """From a bundle written by :meth:`save`."""
        b = torch.load(path, map_location="cpu", weights_only=True)
        dec, rot, text = build_decoder(b["decoder"], b["weights"]), Rotation(**b["rotation"]), None
        if b.get("text"):
            text = TextEncoder(tokenizer().pad_token_id, arch=b["text"]["arch"])
            text.load_state_dict(b["text"]["weights"])
        return cls(dec, rot, b["image_model"], b["beam"], b["pairs"], b.get("samples", 0), text)

    def save(self, path: str | Path) -> None:
        """One file with everything inference needs; decoder weights in bfloat16."""
        weights = {
            k: v.to(torch.bfloat16) if v.is_floating_point() and k != "mu" else v for k, v in self.dec.state_dict().items()
        }
        rotation = {"R": self.rot.R, "mu_a": self.rot.mu_a, "mu_b": self.rot.mu_b}
        text = None if self.text is None else {"arch": arch_of_width(self.text.embed_dim), "weights": self.text.state_dict()}
        bundle = {"decoder": self.dec.config, "weights": weights, "rotation": rotation, "text": text}
        torch.save(
            {**bundle, "image_model": self.image_model, "beam": self.beam, "pairs": self.pairs, "samples": self.samples}, path
        )

    def direction(self, z: torch.Tensor) -> torch.Tensor:
        """Unit vectors of embeddings in the text JEPA's space, centred on its mean: what search compares."""
        return F.normalize(z.double() - self.rot.mu_b, dim=-1)

    def rotate(self, images) -> torch.Tensor:
        """Embeddings ``[N, d]`` of photos (paths or PIL images), rotated into the text JEPA's space."""
        return self.rot(embed_images(self.image_model, [photo(im) for im in images], dtype=SEE).double())

    @torch.no_grad()
    def read(self, texts: list[str]) -> torch.Tensor:
        """Directions ``[N, d]`` of descriptions, by the text JEPA: compare them with :meth:`direction` of photos."""
        if self.text is None:
            raise ValueError("this captioner has no text JEPA; save it again from its decode run")
        return self.direction(self.text.encode(texts).cpu())

    def __call__(self, images, max_len: int = 40) -> list[str]:
        """Captions of photos (paths or PIL images)."""
        return self.caption(self.rotate(images), max_len)

    @torch.no_grad()
    def caption(self, z: torch.Tensor, max_len: int = 40, together: bool = False) -> list[str]:
        """Captions of rotated embeddings ``[N, d]`` (:meth:`rotate`), e.g. the mean over a video's frames.

        With samples, one photo at a time, so that the same photo always gets the same caption; ``together``: all in
        one pass, many times faster for many photos (an app's visitors), each caption then depending on the batch.
        """
        with torch.autocast("cuda", SAY, enabled=DEVICE == "cuda"):
            c = self.dec.condition(z.to(DEVICE))
            if not self.samples:
                return self.dec.generate(c, max_len=max_len, beam=self.beam)
            out, k = [], self.beam + self.samples  # candidates per photo
            for rows in [c] if together else c.split(1):
                with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE == "cuda" else []):
                    torch.manual_seed(0)
                    drafts = self.dec.generate(rows, max_len, self.beam, self.beam, self.samples)
                out += [consensus(drafts[i : i + k]) for i in range(0, len(drafts), k)]
        return out

    def describe(self) -> str:
        """The pipeline in one line."""
        decoder = self.dec.config.get("lm", "a small transformer").split("/")[-1]
        images = " + ".join(IMAGE_NAMES.get(m, m) for m in self.image_model.split(","))
        return f"{images} → one rotation → {decoder} decoder"


def checkpoint(path: Path) -> Path:
    """``path`` itself, or the ``last.ckpt`` of a run directory."""
    path = Path(path)
    return path if path.is_file() else sorted(path.rglob("last.ckpt"))[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captioner", type=Path, help="a bundle written with --save")
    p.add_argument("--decoder", type=Path, help="decode run (train.py --objective decode) or its checkpoint")
    p.add_argument("--rotation", type=Path, help="rotation.pt written by caption.py")
    p.add_argument("--image-model", default="dinov2", help="the image encoder(s) the rotation was fitted on")
    p.add_argument("--pairs", type=int, default=0, help="number of pairs that fitted the rotation (kept in the bundle)")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--samples", type=int, default=0, help="LLM decoders: samples added to the beams, keep their consensus")
    p.add_argument("--save", type=Path, help="write a captioner bundle and exit")
    p.add_argument("images", type=Path, nargs="*", help="photos to caption")
    args = p.parse_args(argv)
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN attention fails on some GPUs in bfloat16
    if args.captioner:
        captioner = Captioner.load(args.captioner)
    elif args.decoder and args.rotation:
        captioner = Captioner.from_runs(
            args.decoder, args.rotation, args.image_model, beam=args.beam, pairs=args.pairs, samples=args.samples
        )
    else:
        p.error("give --captioner, or --decoder and --rotation")
    if args.save:
        return captioner.save(args.save)
    for f, text in zip(args.images, captioner(args.images)):
        print(f"{f.name}: {text}")


if __name__ == "__main__":
    main()
