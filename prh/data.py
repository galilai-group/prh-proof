"""Datasets. Training sets yield ``(image or caption tokens, pair_id, is_test)``; evaluation sets are tables.

Each dataset lives in ``data/<name>`` (a folder or a symbolic link) unless an environment variable says otherwise
(see ``docs/data.md``):

=================  =========================================================================================
``PRH_FLICKR``     Flickr30k with the Karpathy split (``flickr_annotations_30k.csv``, ``flickr30k-images/``)
``PRH_COCO``       COCO Karpathy captions and STS-B (``python -m prh.data`` downloads both into ``data/``)
``PRH_COCO5K``     COCO Karpathy 5k test images (``test_5k_mscoco_2014.csv``, ``images_mscoco_2014_5k_test/``)
``PRH_RECAP``      Recap-DataComp-1B: ``images/*.tar`` shards and ``data/train_data/*.parquet`` texts
``PRH_IMAGENET``   ImageNet-1k in HuggingFace ``datasets`` format
``PRH_MSRVTT``     MSR-VTT: ``videos/*.mp4``, ``annotations/`` (20 captions per clip); ``PRH_FRAMES``: extracted frames
``PRH_ROSETTA``    the benchmark of Schnaus et al. (2026), ``prh.pairfree`` (``python -m prh.data rosetta`` downloads it)
=================  =========================================================================================
"""

import hashlib
import io
import json
import os
import re
import tarfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer

from .models import TOKENIZER

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "data"
FLICKR = Path(os.environ.get("PRH_FLICKR", _DATA / "flickr30k"))
COCO = Path(os.environ.get("PRH_COCO", _DATA))
COCO5K = Path(os.environ.get("PRH_COCO5K", _DATA / "coco5k"))
RECAP = Path(os.environ.get("PRH_RECAP", _DATA / "recap"))
#: text-only paraphrases of the Recap alt-texts, one parquet per Recap file (``python -m prh.data paraphrase``)
RECAP_PARAPHRASE = Path(os.environ.get("PRH_RECAP_PARAPHRASE", _DATA / "recap_paraphrase"))
IMAGENET = Path(os.environ.get("PRH_IMAGENET", _DATA / "imagenet"))
MSRVTT = Path(os.environ.get("PRH_MSRVTT", _DATA / "msrvtt"))
FRAMES = Path(os.environ.get("PRH_FRAMES", _DATA / "msrvtt_frames"))
#: their stored embeddings (``embeddings/<dataset>/<modality>/<model>.pt``), COCO 2014 captions and class prompts
ROSETTA = Path(os.environ.get("PRH_ROSETTA", _DATA / "rosetta"))
#: the commit of github.com/dominik-schnaus/unpaired-rosetta whose code and class prompts we use
ROSETTA_COMMIT = "fdfad84ea488f29702a61a455be71622c4ff5d49"
#: frames kept per MSR-VTT clip; evaluation uses the middle one
N_FRAMES = 8
#: Recap-DataComp texts of one image: its alt-text and two LLaVA-LLaMA-3 recaptions.
RECAP_KEYS = ("org_caption", "re_caption", "re_caption_condition_diverse_topk")
#: ``--data`` kinds of Recap texts: all three, the alt-text alone, or the two recaptions alone.
RECAP_TEXTS = {
    "recaptext": RECAP_KEYS,
    "recaptext-alt": RECAP_KEYS[:1],
    "recaptext-llava": RECAP_KEYS[1:],
    "recaptext-altpara": (RECAP_KEYS[0], "alt_paraphrase"),  # no text written from the image
}


def _tokenize(tok, texts):
    return [torch.tensor(t) for t in tok([t or "" for t in texts], add_special_tokens=False)["input_ids"]]


def pick(captions, train, caption_views):
    """Evaluation: the first caption. Training: every caption (each view draws one) or one for all views.

    ``caption_views`` may be a fraction: each training item then gets every caption with that probability.
    """
    if not train:
        return captions[0]
    if caption_views and torch.rand(()).item() < float(caption_views):
        return captions
    return captions[torch.randint(len(captions), ()).item()]


class EpochSampler(torch.utils.data.Sampler):
    """``num_samples`` random items of ``n`` per epoch, without replacement (``RandomSampler(num_samples=...)``).

    Lightning replaces a ``RandomSampler`` by a ``DistributedSampler`` of the whole dataset under DDP; any other
    sampler it shards across GPUs, so this one keeps the epoch size on any number of GPUs.
    """

    def __init__(self, n: int, num_samples: int):
        self.sampler = torch.utils.data.RandomSampler(range(n), num_samples=num_samples)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class Flickr30k(torch.utils.data.Dataset):
    """Flickr30k images or captions of some Karpathy splits, one item per image, with its pair id.

    Args:
        splits: Karpathy splits to keep, e.g. ``["train"]`` or ``["val", "test"]``.
        modality: ``"image"`` (a PIL image) or ``"text"`` (token ids of its captions, see :func:`pick`).
        train: training item (random or all captions) rather than the first caption.
        caption_views: in training, return all five captions so that each view draws its own.
        root: dataset folder.
    """

    def __init__(self, splits, modality, train, caption_views=False, root=None):
        self.root = Path(root or FLICKR)
        df = pd.read_csv(self.root / "flickr_annotations_30k.csv")
        df = df[df.split.isin(splits)].reset_index(drop=True)
        self.files, self.pair_id, self.is_test = df.filename.tolist(), df.img_id.tolist(), (df.split == "test").tolist()
        self.modality, self.train, self.caption_views = modality, train, caption_views
        if modality == "text":
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            self.captions = [_tokenize(tok, json.loads(r)) for r in df.raw]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.modality == "image":
            x = Image.open(self.root / "flickr30k-images" / self.files[i])
        else:
            x = pick(self.captions[i], self.train, self.caption_views)
        return x, self.pair_id[i], self.is_test[i]


class CocoCaptions(torch.utils.data.Dataset):
    """Text only: the five captions of each of the 113k COCO Karpathy train and restval images."""

    def __init__(self, caption_views=False, root=None):
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        df = pd.concat([pd.read_parquet(Path(root or COCO) / f"coco_{s}.parquet") for s in ("train", "restval")])
        self.captions = [_tokenize(tok, list(r)) for r in df.sentences]
        self.caption_views = caption_views

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        return pick(self.captions[i], True, self.caption_views), -1, False


class RecapText(torch.utils.data.Dataset):
    """Text only, from the first ``n_files`` Recap-DataComp-1B parquet files (about 203k images each).

    ``shard=(rank, world)`` keeps every ``world``-th file from the ``rank``-th: each GPU then samples its own files.

    The texts of an image named by ``keys`` (default: all of :data:`RECAP_KEYS`) are its captions. Arrow buffers
    are shared by the data workers, and tokenization is lazy.
    """

    def __init__(self, n_files, caption_views=False, root=None, shard=(0, 1), keys=RECAP_KEYS):
        import pyarrow as pa
        import pyarrow.parquet as pq

        files = sorted((Path(root or RECAP) / "data" / "train_data").glob("train-*.parquet"))[:n_files]
        files = files[shard[0] :: shard[1]]  # with many GPUs, each holds only its share (about 0.3 GB per file)

        def read(f):  # keys missing from the Recap file come from its row-aligned paraphrase file
            t = pq.read_table(f, columns=[k for k in keys if k in RECAP_KEYS])
            for k in (k for k in keys if k not in RECAP_KEYS):
                t = t.append_column(k, pq.read_table(RECAP_PARAPHRASE / f.name, columns=[k])[k])
            return t.select(list(keys))

        self.cols = pa.concat_tables([read(f) for f in files]).columns
        self.caption_views, self.tok = caption_views, None

    def __len__(self):
        return len(self.cols[0])

    def __getitem__(self, i):
        self.tok = self.tok or AutoTokenizer.from_pretrained(TOKENIZER)
        caps = _tokenize(self.tok, [c[i].as_py() for c in self.cols])
        return pick([c for c in caps if len(c)] or caps[:1], True, self.caption_views), -1, False


def recap_sentences(n_files, root=None):
    """First sentences of the Recap-DataComp recaptions of the first ``n_files`` parquet files (pyarrow strings).

    Sentences about one panel of a collage ("The second image ...") or quoting text in the image are dropped: an
    image embedding cannot carry them.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = sorted((Path(root or RECAP) / "data" / "train_data").glob("train-*.parquet"))[:n_files]
    text = pa.concat_tables([pq.read_table(f, columns=["re_caption"]) for f in files]).column(0).combine_chunks()
    first = pc.struct_field(pc.extract_regex(pc.fill_null(text, ""), r"^(?P<s>[^.!?]*[.!?])"), [0])
    bad = pc.or_(pc.match_substring_regex(first, r"^The \w+ image"), pc.match_substring_regex(first, r"(^|\s)['\"]"))
    return pc.filter(first, pc.invert(pc.fill_null(bad, True)))


class Captions(torch.utils.data.Dataset):
    """Text only, one caption per item, as written: the training texts of the caption decoders.

    ``spec`` joins with ``+``: ``flickr`` (the five captions of each Karpathy training image), ``coco`` (Karpathy
    train and restval captions) and ``recapsent:<n_files>`` (web captions, :func:`recap_sentences`). Items are
    1-tuples, for ``stable_pretraining.data.FromTorchDataset``.
    """

    def __init__(self, spec):
        import pyarrow as pa

        parts = []
        for name in spec.split("+"):
            kind, _, n = name.partition(":")
            if kind == "flickr":
                parts.append(pa.array([c.strip() for caps in flickr_pairs(("train",)).captions for c in caps]))
            elif kind == "coco":
                df = pd.concat([pd.read_parquet(COCO / f"coco_{s}.parquet") for s in ("train", "restval")])
                parts.append(pa.array([c.strip() for caps in df.sentences for c in caps]))
            elif kind == "recapsent":
                parts.append(recap_sentences(int(n)))
            else:
                raise ValueError(f"unknown caption data {name!r}")
        self.texts = pa.chunked_array(parts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return (self.texts[i].as_py(),)


def _index_shard(path):
    """Byte offsets of each sample's jpg and json in a tar shard, from the headers only."""
    members = {}
    with tarfile.open(path) as t:
        for m in t:
            stem, ext = m.name.rsplit(".", 1)
            members.setdefault(stem, {})[ext] = (m.offset_data, m.size)
    return [(*v["jpg"], *v["json"]) for v in members.values() if "jpg" in v and "json" in v]


class RecapImages(torch.utils.data.Dataset):
    """Images (or texts) of the first ``n_shards`` Recap-DataComp-1B tar shards, read in place.

    The byte offsets of all samples are indexed once (with ``workers`` processes) and cached in ``index_dir``.
    """

    def __init__(self, n_shards, modality, caption_views=False, root=None, index_dir=_REPO / "runs", workers=32):
        root = Path(root or RECAP).resolve()
        self.files = sorted((root / "images").glob("recap_datacomp_1b_*.tar"))[:n_shards]
        cache = Path(index_dir) / f"recap_index_{n_shards}_{hashlib.sha1(str(root).encode()).hexdigest()[:8]}.pt"
        if not cache.exists():  # every process (DDP rank) may build it; each writes its own file, then renames
            cache.parent.mkdir(parents=True, exist_ok=True)
            with Pool(workers) as pool:
                rows = [[k, *r] for k, rs in enumerate(pool.map(_index_shard, self.files)) for r in rs]
            tmp = cache.with_suffix(f".{os.getpid()}.tmp")
            torch.save(torch.tensor(rows), tmp)
            tmp.rename(cache)
        self.index, self.modality, self.caption_views, self.fh, self.tok = torch.load(cache), modality, caption_views, {}, None

    def __len__(self):
        return len(self.index)

    def _read(self, shard, offset, size):
        if shard not in self.fh:  # one handle per shard per worker
            self.fh[shard] = open(self.files[shard], "rb")
        self.fh[shard].seek(offset)
        return self.fh[shard].read(size)

    def __getitem__(self, i):
        shard, jo, js, so, ss = self.index[i].tolist()
        if self.modality == "image":
            x = Image.open(io.BytesIO(self._read(shard, jo, js)))
            x.draft("RGB", (448, 448))  # decode large JPEGs at reduced scale; crops are <= 224 px
        else:
            meta = json.loads(self._read(shard, so, ss))
            self.tok = self.tok or AutoTokenizer.from_pretrained(TOKENIZER)
            caps = _tokenize(self.tok, [meta.get(k) for k in RECAP_KEYS])
            x = pick([c for c in caps if len(c)] or caps[:1], True, self.caption_views)
        return x, -1, False


class RecapPairs(RecapImages):
    """Web images with their three texts as written (for fitting a rotation on web pairs, not for training)."""

    def __init__(self, n_shards, root=None):
        super().__init__(n_shards, "image", root=root)

    def __getitem__(self, i):
        shard, jo, js, so, ss = self.index[i].tolist()
        meta = json.loads(self._read(shard, so, ss))
        image = Image.open(io.BytesIO(self._read(shard, jo, js))).convert("RGB")
        return image, [meta.get(k) or "" for k in RECAP_KEYS]


def _msrvtt_split():
    """Video ids of the MSR-VTT splits: train (7,010), val (the other 1,990 of the 9k training clips), test (1,000, JSFUSION)."""
    ann = MSRVTT / "annotations"
    train = pd.read_csv(ann / "MSRVTT_train.7k.csv").video_id.tolist()
    val = sorted(set(pd.read_csv(ann / "MSRVTT_train.9k.csv").video_id) - set(train))
    test = pd.read_csv(ann / "MSRVTT_JSFUSION_test.csv")
    return {"train": train, "val": val, "test": test.video_id.tolist()}, dict(zip(test.video_id, test.sentence))


class MSRVTTClips(torch.utils.data.Dataset):
    """MSR-VTT clips: their frames (:func:`extract_frames`) or the token ids of their 20 captions.

    A clip is one world state observed several times on each side. Training items hold every frame (good image
    views: each view draws its own frame) or one random frame (bad views: crops of one frame), and every caption
    or one, as for Flickr30k. Evaluation items are the middle frame and the first caption (for test clips, the
    JSFUSION sentence).

    Args:
        splits: some of ``"train"``, ``"val"``, ``"test"``.
        modality: ``"image"`` or ``"text"``.
        train: training item rather than the evaluation one.
        caption_views: in training, return every frame or caption so that each view draws its own.
    """

    def __init__(self, splits, modality, train, caption_views=False):
        ids, test_caption = _msrvtt_split()
        self.ids = [v for s in splits for v in ids[s]]
        self.is_test = [s == "test" for s in splits for _ in ids[s]]
        self.modality, self.train, self.caption_views = modality, train, caption_views
        if modality == "text":
            caps = {}
            for c in json.load(open(MSRVTT / "annotations" / "MSRVTT_data.json"))["sentences"]:
                caps.setdefault(c["video_id"], []).append(c["caption"])
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            self.captions = [_tokenize(tok, ([test_caption[v]] if v in test_caption else []) + caps[v]) for v in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = int(self.ids[i][len("video") :])
        if self.modality == "text":
            return pick(self.captions[i], self.train, self.caption_views), pid, self.is_test[i]
        frames = [FRAMES / self.ids[i] / f"{k}.jpg" for k in range(N_FRAMES)]
        if not self.train:
            return Image.open(frames[N_FRAMES // 2]), pid, self.is_test[i]
        frames = [Image.open(f) for f in frames] if self.caption_views else Image.open(frames[torch.randint(N_FRAMES, ()).item()])
        return frames, pid, self.is_test[i]


def _extract(video, out):
    import decord

    reader = decord.VideoReader(str(video))
    (out / video.stem).mkdir(parents=True, exist_ok=True)
    idx = torch.linspace(0, len(reader) - 1, N_FRAMES + 2)[1:-1].round().long().tolist()  # skip the first and last frames
    for k, frame in enumerate(reader.get_batch(idx).asnumpy()):
        Image.fromarray(frame).save(out / video.stem / f"{k}.jpg", quality=95)


def extract_frames(root=None, out=None, workers=16):
    """Save :data:`N_FRAMES` evenly spaced frames of every MSR-VTT clip as ``<out>/<video_id>/<k>.jpg``."""
    out = Path(out or FRAMES)
    videos = sorted((Path(root or MSRVTT) / "videos").glob("*.mp4"))
    with Pool(workers) as pool:
        pool.starmap(_extract, [(v, out) for v in videos if not (out / v.stem / f"{N_FRAMES - 1}.jpg").exists()])


def training_set(spec, modality, caption_views=False, shard=(0, 1)):
    """Concatenation of the ``+``-separated parts of ``spec``.

    Parts: ``flickr``, ``coco`` (text), ``recaptext:<n_files>`` (text; ``recaptext-alt`` and ``recaptext-llava``
    keep only the alt-text or only the two recaptions, :data:`RECAP_TEXTS`), ``recap:<n_shards>`` (image or text),
    ``msrvtt`` (frames or captions of the 7k training clips). ``shard`` splits ``recaptext`` files across GPUs.
    """

    def part(name):
        kind, _, n = name.partition(":")
        if kind in RECAP_TEXTS:
            return RecapText(int(n), caption_views, shard=shard, keys=RECAP_TEXTS[kind])
        if kind == "recap":
            return RecapImages(int(n), modality, caption_views)
        if kind == "coco":
            return CocoCaptions(caption_views)
        if kind == "msrvtt":
            return MSRVTTClips(["train"], modality, True, caption_views)
        assert kind == "flickr", f"unknown training data {name!r}"
        return Flickr30k(["train"], modality, True, caption_views)

    return torch.utils.data.ConcatDataset([part(p) for p in spec.split("+")])


def flickr_pairs(splits=("val", "test")):
    """Flickr30k table (``path``, ``captions``, ``pair_id``, ``is_test``) of some Karpathy splits."""
    df = pd.read_csv(FLICKR / "flickr_annotations_30k.csv")
    df = df[df.split.isin(splits)].reset_index(drop=True)
    return pd.DataFrame(
        {
            "path": [FLICKR / "flickr30k-images" / f for f in df.filename],
            "captions": [json.loads(r) for r in df.raw],
            "pair_id": df.img_id,
            "is_test": df.split == "test",
        }
    )


def msrvtt_pairs(splits=("val", "test")):
    """MSR-VTT table (``path`` of the middle frame, ``captions``, ``pair_id``, ``is_test``) of some splits."""
    ds = MSRVTTClips(splits, "image", False)
    ids, test_caption = _msrvtt_split()
    caps = {}
    for c in json.load(open(MSRVTT / "annotations" / "MSRVTT_data.json"))["sentences"]:
        caps.setdefault(c["video_id"], []).append(c["caption"])
    return pd.DataFrame(
        {
            "path": [FRAMES / v / f"{N_FRAMES // 2}.jpg" for v in ds.ids],
            "captions": [([test_caption[v]] if v in test_caption else []) + caps[v] for v in ds.ids],
            "pair_id": [int(v[len("video") :]) for v in ds.ids],
            "is_test": ds.is_test,
        }
    )


def coco5k_pairs():
    """COCO Karpathy test split, 5,000 images with five captions each; every row is a test pair."""
    df = pd.read_csv(COCO5K / "test_5k_mscoco_2014.csv")
    return pd.DataFrame(
        {
            "path": [COCO5K / "images_mscoco_2014_5k_test" / f for f in df.filename],
            "captions": [json.loads(r) for r in df.raw],
            "pair_id": df.cocoid,
            "is_test": True,
        }
    )


def wit1024():
    """The 1,024 WIT image--caption pairs of the PRH paper (Huh et al., 2024), as a HuggingFace dataset."""
    import datasets

    return datasets.load_dataset("minhuh/prh", revision="wit_1024", split="train")


def stsb():
    """STS-Benchmark test set: ``sentence1``, ``sentence2`` and human similarity ``score``."""
    return pd.read_parquet(COCO / "stsb_test.parquet")


def download_text_data(root=None):
    """Fetch the two small text-only datasets into ``root`` (default ``PRH_COCO``) from HuggingFace.

    COCO Karpathy train and restval captions (``yerevann/coco-karpathy``) and the STS-B test set
    (``sentence-transformers/stsb``, scores in [0, 1]).
    """
    import datasets

    root = Path(root or COCO)
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "restval"):
        datasets.load_dataset("yerevann/coco-karpathy", split=split).to_pandas().to_parquet(root / f"coco_{split}.parquet")
    datasets.load_dataset("sentence-transformers/stsb", split="test").to_pandas().to_parquet(root / "stsb_test.parquet")


def download_rosetta(qwen="qwen"):
    """The zero-pair benchmark of Schnaus et al. (2026) into ``PRH_ROSETTA``, for ``prh.pairfree``.

    Their stored embeddings (DINOv2 ViT-B/14 of COCO 2014, CIFAR-10, CIFAR-100 and ImageNet-100, the labels, and their
    all-mpnet-base-v2 and Qwen3-8B texts: 1.8 GB, and 35 GB more for Qwen3 unless ``python -m prh.data rosetta
    no-qwen``), the COCO 2014 captions that ``evaluate.py pairfree-embed`` embeds our encoders on, and their class
    prompts (``datasets.py`` of their code, read as data by ``prh.pairfree.prompts``).
    """
    import urllib.request
    import zipfile

    from huggingface_hub import snapshot_download

    texts = ("mpnet", "qwen3-*") if qwen != "no-qwen" else ("mpnet",)
    sets = ("coco_train2014", "coco_val2014", "CIFAR-10", "CIFAR-100", "ImageNet-100")
    files = ("labels.pt", "vision/dinov2_vit-b14@224_mean.pt", *(f"language*/{t}.pt" for t in texts))
    patterns = [f"embeddings/{d}/{f}" for d in sets for f in files]
    snapshot_download("schnaus/unpaired-rosetta-embeddings", repo_type="dataset", local_dir=ROSETTA, allow_patterns=patterns)
    code = f"https://raw.githubusercontent.com/dominik-schnaus/unpaired-rosetta/{ROSETTA_COMMIT}/modalities/image_captions"
    urllib.request.urlretrieve(f"{code}/datasets.py", ROSETTA / "datasets.py")
    ann = ROSETTA / "raw/coco2014/annotations"
    if not (ann / "captions_val2014.json").exists():
        ann.mkdir(parents=True, exist_ok=True)
        archive, _ = urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
        with zipfile.ZipFile(archive) as z:
            for split in ("train", "val"):
                (ann / f"captions_{split}2014.json").write_bytes(z.read(f"annotations/captions_{split}2014.json"))


_FLICKR_URL = re.compile(r"(?:staticflickr\.com|flickr\.com)/(?:[^/]+/)*?(\d{5,})_[0-9a-f]+")  # a Flickr photo id in an image URL


def _plain(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


def _overlap_file(path, wit_full, wit_prefix, flickr, coco):
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["url", "org_caption"]).to_pydict()
    wiki, full, prefix, photos, cocos = 0, set(), set(), set(), set()
    for url, caption in zip(t["url"], t["org_caption"]):
        wiki += "wikimedia" in (url or "")
        photo = _FLICKR_URL.search(url or "")
        if photo and int(photo[1]) in flickr:
            photos.add(int(photo[1]))
        if photo and int(photo[1]) in coco:
            cocos.add(coco[int(photo[1])])
        text = _plain(caption)
        full |= {wit_full[text]} if text in wit_full else set()
        prefix |= {wit_prefix[text[:40]]} if text[:40] in wit_prefix else set()
    return wiki, sorted(full), sorted(prefix), sorted(photos), sorted(cocos)


def overlap(out="runs/results/overlap.json", workers=16):
    """Evaluation pairs among the Recap-DataComp training data, per text file (``python -m prh.data overlap``).

    WIT-1024 captions among the alt-texts (lower-cased, letters, digits and spaces kept; matched exactly, or by their
    first 40 characters), Wikimedia-hosted images, and Flickr30k and COCO 2014 photos among the images: Flickr30k file
    names are Flickr photo ids, COCO 2014 annotations give each photo's Flickr URL (``python -m prh.data rosetta``), and
    Flickr-hosted DataComp URLs contain the photo id, so the counts are lower bounds. ``evaluate.py wit --drop <out>``
    rescores without the matched WIT pairs (``--bench coco``: without the matched COCO test photos).
    """
    import functools

    wit = [_plain(t[0]) for t in wit1024()["text"]]
    ann = pd.read_csv(FLICKR / "flickr_annotations_30k.csv")
    split = {int(f.split(".")[0]): s for f, s in zip(ann.filename, ann.split)}
    coco, coco_split = {}, {}  # Flickr photo id -> COCO id, COCO id -> train2014 / val2014
    for part in ("train2014", "val2014"):
        for im in json.load(open(ROSETTA / f"raw/coco2014/annotations/captions_{part}.json"))["images"]:
            photo = _FLICKR_URL.search(im.get("flickr_url", ""))
            if photo:
                coco[int(photo[1])], coco_split[im["id"]] = im["id"], part
    test5k = set(coco5k_pairs().pair_id)
    scan = functools.partial(
        _overlap_file,
        wit_full={t: i for i, t in enumerate(wit) if len(t) >= 20},
        wit_prefix={t[:40]: i for i, t in enumerate(wit) if len(t) >= 40},
        flickr=split,
        coco=coco,
    )
    files = sorted(str(p) for p in (RECAP / "data" / "train_data").glob("train-*.parquet"))
    with Pool(workers) as pool:
        per_file = dict(zip(files, pool.map(scan, files)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(per_file, open(out, "w"))
    for n in (500, 1000, len(files)):  # about 100M, 200M and all 940M images
        rows = [per_file[f] for f in files[:n]]
        full = set().union(*(r[1] for r in rows))
        either = full.union(*(r[2] for r in rows))
        photos = set().union(*(r[3] for r in rows))
        cocos = set().union(*(r[4] for r in rows))
        by_split = pd.Series([split[p] for p in photos], dtype=str).value_counts().to_dict()
        print(
            f"first {n} files: WIT captions matched exactly {len(full)}, exactly or by prefix {len(either)} of 1024; "
            f"Wikimedia images {sum(r[0] for r in rows)}; Flickr30k photos {len(photos)} {by_split}; COCO 2014 photos "
            f"{len(cocos)} ({sum(coco_split[c] == 'val2014' for c in cocos)} val2014, {len(cocos & test5k)} COCO-5k test)",
            flush=True,
        )


@torch.no_grad()
def paraphrase(index, model="Qwen/Qwen2.5-7B-Instruct", batch=384):
    """Good text views without images (``python -m prh.data paraphrase <file index>``).

    A text-only instruct language model rewrites each alt-text of one Recap parquet file; the paraphrases are written
    row-aligned to ``RECAP_PARAPHRASE/<same file name>`` (one column, ``alt_paraphrase``). No image is ever read.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM

    src = sorted((RECAP / "data" / "train_data").glob("train-*.parquet"))[int(index)]
    out = RECAP_PARAPHRASE / src.name
    if out.exists():
        return
    alts = [a or "" for a in pq.read_table(src, columns=["org_caption"])["org_caption"].to_pylist()]
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
    prompt = (
        "Rewrite the following text with different words and sentence structure. Keep exactly the same meaning, "
        "add no information, and reply with the rewritten text only.\n\nText: {}"
    )
    order, para = sorted(range(len(alts)), key=lambda i: len(alts[i])), [""] * len(alts)  # similar lengths per batch
    torch.manual_seed(int(index))
    for k in range(0, len(order), batch):
        rows = order[k : k + batch]
        chats = [[{"role": "user", "content": prompt.format(alts[i][:400])}] for i in rows]
        x = tok.apply_chat_template(chats, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True)
        gen = lm.generate(**x.to("cuda"), max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.95)
        for i, g in zip(rows, tok.batch_decode(gen[:, x["input_ids"].shape[1] :], skip_special_tokens=True)):
            para[i] = g.strip() if alts[i] else ""
    RECAP_PARAPHRASE.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"alt_paraphrase": para}), out.with_suffix(".tmp"))
    out.with_suffix(".tmp").rename(out)


if __name__ == "__main__":  # python -m prh.data [frames | overlap | paraphrase <file index> | rosetta [no-qwen]]
    import sys

    commands = {"frames": extract_frames, "overlap": overlap, "paraphrase": paraphrase, "rosetta": download_rosetta}
    commands.get(sys.argv[1] if sys.argv[1:] else "", download_text_data)(*sys.argv[2:])
