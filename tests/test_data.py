"""Datasets, on tiny synthetic copies of each format."""

import numpy as np
import pytest
import torch
from PIL import Image

from prh import data


def test_flickr_images_and_captions(roots):
    images = data.Flickr30k(["val", "test"], "image", train=False)
    x, pid, is_test = images[3]
    assert len(images) == 4 and isinstance(x, Image.Image) and (pid, is_test) == (111, True)
    text = data.Flickr30k(["train"], "text", train=True, caption_views=True)
    caps, pid, is_test = text[0]
    assert len(text) == 8 and len(caps) == 5 and all(c.dtype == torch.long for c in caps) and not is_test
    first, _, _ = data.Flickr30k(["train"], "text", train=False)[0]
    assert torch.equal(first, caps[0])


def test_coco_captions_are_text_only_training_data(roots):
    ds = data.CocoCaptions(caption_views=True)
    caps, pid, is_test = ds[4]
    assert len(ds) == 5 and len(caps) == 5 and pid == -1 and not is_test


def test_recap_text_uses_the_three_texts_and_skips_empty_ones(roots):
    ds = data.RecapText(1, caption_views=True)
    assert len(ds) == 2 and len(ds[0][0]) == 3 and len(ds[1][0]) == 1


def test_recap_text_keeps_only_the_chosen_texts(roots):
    alt, llava = (data.training_set(f"{k}:1", "text", caption_views=True) for k in ("recaptext-alt", "recaptext-llava"))
    assert [len(alt[i][0]) for i in range(2)] == [1, 1] and [len(llava[i][0]) for i in range(2)] == [2, 1]
    tok = data.AutoTokenizer.from_pretrained(data.TOKENIZER)
    assert tok.decode(alt[0][0][0]) == "alt text" and tok.decode(llava[1][0][0]) == "only this"


def test_recap_text_reads_paraphrases_row_aligned(roots, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = sorted((data.RECAP / "data" / "train_data").glob("train-*.parquet"))[0]
    monkeypatch.setattr(data, "RECAP_PARAPHRASE", tmp_path / "para")
    (tmp_path / "para").mkdir()
    pq.write_table(pa.table({"alt_paraphrase": ["alternative text", "nothing"]}), tmp_path / "para" / src.name)
    ds = data.training_set("recaptext-altpara:1", "text", caption_views=True)
    tok = data.AutoTokenizer.from_pretrained(data.TOKENIZER)
    assert [tok.decode(c) for c in ds[0][0]] == ["alt text", "alternative text"] and len(ds[1][0]) == 1


def test_training_set_concatenates_parts(roots):
    assert len(data.training_set("flickr+coco+recaptext:1", "text", caption_views=True)) == 8 + 5 + 2
    with pytest.raises(AssertionError):
        data.training_set("wikipedia", "text")


def test_evaluation_tables(roots):
    pairs = data.flickr_pairs()
    assert len(pairs) == 4 and pairs.is_test.sum() == 2 and pairs.path.iloc[0].exists()
    assert pairs.captions.iloc[0][0] == "photo 8 caption 0" and len(data.flickr_pairs(("train", "val", "test"))) == 12


def test_pick():
    caps = [torch.tensor([1]), torch.tensor([2])]
    assert data.pick(caps, False, True) is caps[0] and data.pick(caps, True, True) is caps
    assert any(data.pick(caps, True, False) is caps[1] for _ in range(50))
    torch.manual_seed(0)
    good = sum(data.pick(caps, True, 0.3) is caps for _ in range(2000))  # a fraction of good-view items
    assert 500 < good < 700 and data.pick(caps, True, 1.0) is caps and data.pick(caps, True, 0.0) is not caps


def test_msrvtt_splits_and_items(roots):
    test = data.MSRVTTClips(["val", "test"], "image", train=False)
    x, pid, is_test = test[3]
    assert len(test) == 6 and (pid, is_test) == (9, True) and x.getpixel((0, 0))[0] == 20 * (data.N_FRAMES // 2)
    clip, _, _ = data.MSRVTTClips(["train"], "image", train=True, caption_views=True)[0]
    assert isinstance(clip, list) and len(clip) == data.N_FRAMES  # good views: every frame
    one, _, _ = data.MSRVTTClips(["train"], "image", train=True)[0]
    assert isinstance(one, Image.Image)  # bad views: one frame for all views
    text = data.MSRVTTClips(["test"], "text", train=False)
    first, pid, _ = text[0]
    assert pid == 9 and len(data.MSRVTTClips(["train"], "text", train=True, caption_views=True)[0][0]) == 20
    assert data.msrvtt_pairs().captions.iloc[3][0] == "test sentence video9"  # JSFUSION sentence first for test clips


def test_extract_frames_from_a_video(tmp_path):
    pytest.importorskip("decord")
    cv2 = pytest.importorskip("cv2")

    (tmp_path / "videos").mkdir()
    writer = cv2.VideoWriter(str(tmp_path / "videos" / "video0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for t in range(30):
        writer.write(np.full((48, 64, 3), 8 * t, dtype=np.uint8))
    writer.release()
    data.extract_frames(tmp_path, tmp_path / "frames", workers=1)
    shades = [Image.open(tmp_path / "frames" / "video0" / f"{k}.jpg").getpixel((10, 10))[0] for k in range(data.N_FRAMES)]
    assert len(shades) == data.N_FRAMES and shades == sorted(shades) and shades[0] > 0  # evenly spaced, first frame skipped


def test_epoch_sampler_draws_distinct_items_and_is_not_a_random_sampler():
    s = data.EpochSampler(1000, 100)
    idx = list(s)
    assert len(s) == len(idx) == 100 and len(set(idx)) == 100 and max(idx) < 1000
    assert not isinstance(s, torch.utils.data.RandomSampler)  # Lightning would replace it by a whole-data sampler


def test_recap_text_shards_split_the_files_between_gpus(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "data" / "train_data").mkdir(parents=True)
    for f in range(4):  # file f holds f + 1 images
        texts = [f"file {f} image {i}" for i in range(f + 1)]
        pq.write_table(pa.table({k: texts for k in data.RECAP_KEYS}), tmp_path / "data" / "train_data" / f"train-{f:05d}.parquet")
    whole = data.RecapText(4, root=tmp_path)
    parts = [data.RecapText(4, root=tmp_path, shard=(r, 2)) for r in range(2)]
    assert len(whole) == 10 and [len(p) for p in parts] == [1 + 3, 2 + 4]


def test_recap_sentences_keep_one_clean_sentence_per_image(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "data" / "train_data").mkdir(parents=True)
    recaptions = [
        "A dog runs on a beach. The sky is blue.",
        "The second image shows a cat.",
        "A sign with the text 'OPEN' hangs on a door.",
        "A man's red bike leans on a wall!",
        None,
    ]
    pq.write_table(pa.table({"re_caption": recaptions}), tmp_path / "data" / "train_data" / "train-00000.parquet")
    assert data.recap_sentences(1, root=tmp_path).to_pylist() == ["A dog runs on a beach.", "A man's red bike leans on a wall!"]


def test_captions_are_kept_as_written(roots):
    captions = data.Captions("flickr+coco")
    texts = [captions[i][0] for i in range(len(captions))]
    assert len(texts) == 8 * 5 + (3 + 2) * 5  # 8 Flickr training images, 5 COCO images, five captions each
    assert {"photo 0 caption 4", "coco train 0 0", "coco restval 1 4"} <= set(texts)
    with pytest.raises(ValueError):
        data.Captions("flickr+imagenet")


def test_overlap_finds_wit_captions_and_flickr_and_coco_photos_in_a_recap_file(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from prh.data import _overlap_file, _plain

    caption = "A Red Bus, parked at the depot in the snow -- at night (2009)"
    rows = {
        "url": ["https://farm4.staticflickr.com/3001/1000092795_7f4ce2a7ea_z.jpg", "https://upload.wikimedia.org/x.jpg", None],
        "org_caption": ["a photo", caption.upper() + "!", None],
    }
    pq.write_table(pa.table(rows), tmp_path / "train-0.parquet")
    wit = _plain(caption)
    found = _overlap_file(tmp_path / "train-0.parquet", {wit: 7}, {wit[:40]: 7}, {1000092795: "train"}, {1000092795: 391895})
    assert found == (1, [7], [7], [1000092795], [391895])  # the COCO photo is matched by the same Flickr id
