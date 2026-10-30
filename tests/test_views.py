"""Text views (good and bad) and image views."""

import torch
from PIL import Image

from prh.views import CaptionView, Photometric, TextView, image_eval_view, view_transforms


def test_span_view_is_a_masked_contiguous_crop(tok):
    ids = torch.arange(1000, 1020)
    for _ in range(200):
        t = TextView(tok, scale=(0.15, 0.5), mask_ratio=0.3, max_len=32)({"text": ids})["text"]
        n = int(t.ne(tok.pad_token_id).sum())
        span = t[1 : n - 1]
        assert t.shape == (32,) and t[0] == tok.cls_token_id and t[n - 1] == tok.sep_token_id
        assert 1 <= len(span) <= 10 and t[n:].eq(tok.pad_token_id).all()
        kept = span[span.ne(tok.mask_token_id)]
        if len(kept):
            assert kept[-1] - kept[0] <= len(span) - 1 and kept.ge(1000).all() and kept.lt(1020).all()


def test_eval_view_is_identity_up_to_truncation(tok):
    ids = torch.arange(1000, 1100)
    t = TextView(tok, max_len=16)({"text": ids})["text"]
    assert torch.equal(t[1:15], ids[:14]) and t[0] == tok.cls_token_id and t[15] == tok.sep_token_id


def test_good_views_draw_their_own_caption(tok):
    caps = [torch.arange(1000, 1010), torch.arange(2000, 2010), torch.arange(3000, 3010)]
    seen = {int(TextView(tok)({"text": caps})["text"][1]) // 1000 for _ in range(100)}
    assert seen == {1, 2, 3}


def test_multiview_text_transform_gives_independent_views(tok):
    train, _, gpu = view_transforms("text", tok, caption_views=True, views=(2, 6))
    caps = [torch.arange(1000 * i, 1000 * i + 12) for i in range(1, 6)]
    views = train({"text": caps})["views"]
    assert gpu is None and len(views) == 8 and all(v["text"].shape == (64,) for v in views)
    assert len({int(v["text"][1]) // 1000 for v in views}) > 1  # several captions among the views


def test_pair_view_concatenates_two_different_captions(tok):
    caps = [torch.arange(1000, 1005), torch.arange(2000, 2005), torch.arange(3000, 3005)]
    for _ in range(50):
        t = TextView(tok, max_len=32, pair=True)({"text": caps})["text"]
        seps = (t == tok.sep_token_id).nonzero().flatten().tolist()
        assert t[0] == tok.cls_token_id and len(seps) == 2
        assert int(t[1]) // 1000 != int(t[seps[0] + 1]) // 1000 and t[seps[1] + 1 :].eq(tok.pad_token_id).all()


def test_next_token_and_decoder_arms_read_whole_captions(tok):
    train, val, gpu = view_transforms("text", tok, objective="ntp")
    ids = torch.arange(1000, 1030)
    assert gpu is None and torch.equal(train({"text": ids})["text"], val({"text": ids})["text"])
    train, val, gpu = view_transforms("text", tok, objective="decode")  # decoders read captions as written
    ids = torch.tensor(tok("a dog on a beach", add_special_tokens=False)["input_ids"])
    assert gpu is None and torch.equal(train({"caption": "a dog on a beach"})["text"], val({"text": ids})["text"])


def test_image_views_have_two_resolutions_and_survive_cmyk():
    train, val, gpu = view_transforms("image", None, views=(2, 3))
    views = train({"image": Image.new("CMYK", (300, 200))})["views"]
    assert [tuple(v["image"].shape) for v in views] == [(3, 224, 224)] * 2 + [(3, 96, 96)] * 3
    assert views[0]["image"].dtype == torch.uint8 and gpu.n_global == 2
    x = image_eval_view()({"image": Image.new("L", (500, 300))})["image"]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_good_image_views_draw_their_own_frame():
    train, _, _ = view_transforms("image", None, views=(2, 6))
    clip = [Image.new("RGB", (120, 90), (30 * k,) * 3) for k in range(8)]
    views = train({"image": clip})["views"]
    shades = {int(v["image"][0, 45, 60]) // 30 for v in views}
    assert len(shades) > 1  # different views come from different frames
    single = train({"image": clip[3]})["views"]
    assert {int(v["image"][0, 45, 60]) for v in single} == {90}  # bad views: every view crops the same frame


def test_empty_and_single_captions_do_not_crash(tok):
    empty = TextView(tok, scale=(0.15, 0.5), mask_ratio=0.15)({"text": torch.tensor([], dtype=torch.long)})["text"]
    assert empty[:2].tolist() == [tok.cls_token_id, tok.sep_token_id] and (empty[2:] == tok.pad_token_id).all()
    one = TextView(tok, max_len=16, pair=True)({"text": [torch.tensor([2000, 2001])]})["text"]
    assert one[:4].tolist() == [tok.cls_token_id, 2000, 2001, tok.sep_token_id]


def test_photometric_handles_views_without_locals():
    views = [{"image": torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)} for _ in range(2)]
    out = Photometric(n_global=2)({"views": views})["views"]
    assert len(out) == 2 and all(v["image"].dtype == torch.float32 and v["image"].shape == (2, 3, 32, 32) for v in out)


def test_caption_view_keeps_the_written_caption_and_adds_the_encoder_view(tok):
    x = CaptionView(tok)({"caption": "A dog on the Eiffel Tower."})
    ids = torch.tensor(tok("A dog on the Eiffel Tower.", add_special_tokens=False)["input_ids"])
    assert x["caption"] == "A dog on the Eiffel Tower." and torch.equal(x["text"], TextView(tok)({"text": ids})["text"])
