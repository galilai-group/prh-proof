"""Text encoder, caption decoders and their checkpoints."""

import pytest
import torch

from prh.models import ARCH, CaptionDecoder, LMDecoder, TextEncoder, arch_of_width, load_decoder


def test_arch_table_matches_vit_widths():
    assert {a: w for a, (w, _, _) in ARCH.items()} == {"tiny": 192, "small": 384, "base": 768, "large": 1024}
    assert all(arch_of_width(w) == a for a, (w, _, _) in ARCH.items())


def test_causal_flag_controls_attention_to_future_tokens(tok):
    x = torch.randint(1000, 2000, (2, 12))
    y = x.clone()
    y[:, 8:] = torch.randint(1000, 2000, (2, 4))
    for causal in (True, False):
        enc = TextEncoder(tok.pad_token_id, causal=causal, arch="tiny").eval()
        with torch.no_grad():
            assert torch.allclose(enc.hidden(x)[:, :8], enc.hidden(y)[:, :8], atol=1e-5) == causal


def test_pooling_ignores_padding(tok):
    x = torch.randint(1000, 2000, (1, 6))
    padded = torch.cat([x, torch.full((1, 4), tok.pad_token_id)], 1)
    for causal in (True, False):
        enc = TextEncoder(tok.pad_token_id, causal=causal, arch="tiny").eval()
        with torch.no_grad():
            assert torch.allclose(enc(x), enc(padded), atol=1e-5)


def test_text_encoder_from_jepa_and_next_token_checkpoints(tok, tmp_path):
    x = torch.randint(1000, 2000, (3, 10))
    for prefix, causal in (("model.backbone.", False), ("backbone.", True)):
        enc = TextEncoder(tok.pad_token_id, causal=causal, arch="tiny").eval()
        sd = {prefix + k: v for k, v in enc.state_dict().items()} | {"model.projector.0.weight": torch.zeros(1)}
        torch.save({"state_dict": sd}, tmp_path / "last.ckpt")
        loaded = TextEncoder.from_checkpoint(tmp_path / "last.ckpt", tok.pad_token_id)
        assert loaded.causal == causal and loaded.embed_dim == 192
        with torch.no_grad():
            assert torch.allclose(loaded(x), enc(x))


def test_decoder_condition_is_centred_unit_norm_and_noise_sets_the_cosine(tok):
    dec = CaptionDecoder(tok.pad_token_id, cond_dim=256, arch="tiny")
    z = torch.randn(4000, 256) + 5
    dec.mu.copy_(z.mean(0))
    c = dec.condition(z)
    assert torch.allclose(c.norm(dim=1), torch.ones(4000)) and c.mean(0).abs().max() < 0.05
    noisy = dec.condition(z, noise=3.0)
    assert abs((c * noisy).sum(1).mean() - (1 + 3.0**2) ** -0.5) < 0.01


def test_decoder_running_mean():
    dec = CaptionDecoder(0, cond_dim=8, arch="tiny")
    for _ in range(2000):
        dec.update_mean(torch.randn(64, 8) + 3)
    assert torch.allclose(dec.mu, torch.full((8,), 3.0), atol=0.1)


def test_decoder_loss_is_teacher_forced_cross_entropy(tok):
    dec = CaptionDecoder(tok.pad_token_id, cond_dim=32, arch="tiny").eval()
    c = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)
    tokens = torch.tensor([[tok.cls_token_id, 2000, 2001, tok.sep_token_id, tok.pad_token_id]] * 2)
    with torch.no_grad():
        logp = dec.logits(c, tokens[:, :-1]).log_softmax(-1)  # position i predicts token i (prefix at 0)
        manual = -torch.stack([logp[:, i, tokens[0, i]] for i in (1, 2, 3)]).mean()
        assert torch.allclose(dec.loss(c, tokens), manual, atol=1e-5)


def test_decoder_overfits_one_caption_and_generates_it(tok):
    dec = CaptionDecoder(tok.pad_token_id, cond_dim=16, arch="tiny")
    caption = "a dog runs on the beach"
    ids = tok(caption, return_tensors="pt")["input_ids"]
    c = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    for _ in range(150):
        opt.zero_grad()
        dec.loss(c, ids).backward()
        opt.step()
    dec.eval()
    assert dec.generate(c) == [caption] and dec.generate(c, beam=3) == [caption]


def test_beam_search_shapes_and_greedy_special_case(tok):
    dec = CaptionDecoder(tok.pad_token_id, cond_dim=24, arch="tiny").eval()
    c = torch.nn.functional.normalize(torch.randn(4, 24), dim=-1)
    greedy, beam = dec.generate(c, max_len=6), dec.generate(c, max_len=6, beam=4)
    assert len(greedy) == len(beam) == 4 and all(isinstance(s, str) for s in beam)
    words = dec.decoder.bert.embeddings.word_embeddings  # beam 1 is plain greedy decoding
    ids = torch.full((4, 1), tok.cls_token_id)
    with torch.no_grad():
        for _ in range(6):
            h = dec.decoder.bert(inputs_embeds=torch.cat([dec.prefix(c)[:, None], words(ids)], 1)).last_hidden_state[:, -1]
            logits = h @ words.weight.T
            logits[:, [tok.pad_token_id, tok.cls_token_id, tok.mask_token_id, tok.unk_token_id]] = -float("inf")
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], 1)
    expected = [
        tok.decode(
            i[: (i == tok.sep_token_id).nonzero()[0, 0] + 1 if (i == tok.sep_token_id).any() else None], skip_special_tokens=True
        )
        for i in ids
    ]
    assert greedy == expected


def test_decoder_checkpoint_roundtrip(tok, tmp_path):
    dec = CaptionDecoder(tok.pad_token_id, cond_dim=384, arch="tiny").eval()
    dec.mu.normal_()
    torch.save({"state_dict": {"decoder." + k: v for k, v in dec.state_dict().items()}}, tmp_path / "last.ckpt")
    loaded = load_decoder(tmp_path / "last.ckpt")
    c = torch.nn.functional.normalize(torch.randn(2, 384), dim=1)
    assert torch.equal(loaded.mu, dec.mu) and loaded.generate(c, max_len=5) == dec.generate(c, max_len=5)


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    """A two-layer random Qwen2 with the real Qwen2.5 tokenizer, saved like a HuggingFace model."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    path = tmp_path_factory.mktemp("tiny_lm")
    lm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    config = Qwen2Config(
        vocab_size=len(lm_tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(path)
    lm_tok.save_pretrained(path)
    return str(path)


def test_lm_decoder_overfits_one_caption_and_generates_it_in_sentence_case(tok, tiny_lm):
    dec = LMDecoder(tiny_lm, cond_dim=16, n_prefix=2)
    ids = tok("a dog runs on the beach. it is sunny", return_tensors="pt")["input_ids"]  # what the encoder reads
    c = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(100):
        opt.zero_grad()
        dec.loss(c, ids).backward()
        opt.step()
    dec.eval()
    assert dec.generate(c, max_len=16) == dec.generate(c, max_len=16, beam=3) == ["A dog runs on the beach. It is sunny"]


def test_lm_decoder_checkpoint_roundtrip(tiny_lm, tmp_path):
    dec = LMDecoder(tiny_lm, cond_dim=8, n_prefix=3).eval()
    dec.mu.normal_()
    state = {"decoder." + k: v for k, v in dec.state_dict().items()}
    torch.save({"state_dict": state, "hyper_parameters": {"lm": tiny_lm}}, tmp_path / "last.ckpt")
    loaded = load_decoder(tmp_path / "last.ckpt")
    c = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    assert isinstance(loaded, LMDecoder) and loaded.n_prefix == 3 and torch.equal(loaded.mu, dec.mu)
    assert loaded.generate(c, max_len=5) == dec.generate(c, max_len=5)


def test_encode_reads_strings_like_the_evaluation_view(tok):
    from prh.views import TextView

    enc = TextEncoder(tok.pad_token_id, arch="tiny").eval()
    texts = ["a dog runs on the beach", "two people " * 40]  # the second is truncated
    view = TextView(tok)
    ids = torch.stack([view({"text": torch.tensor(tok(t, add_special_tokens=False)["input_ids"])})["text"] for t in texts])
    with torch.no_grad():
        assert torch.allclose(enc.encode(texts), enc(ids), atol=1e-6)


def test_nbest_starts_with_the_best_caption(tok, tiny_lm):
    c = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
    for dec in (
        CaptionDecoder(tok.pad_token_id, cond_dim=16, arch="tiny").eval(),
        LMDecoder(tiny_lm, cond_dim=16, n_prefix=2).eval(),
    ):
        best, nbest = dec.generate(c, max_len=6, beam=4), dec.generate(c, max_len=6, beam=4, nbest=3)
        assert len(nbest) == 9 and nbest[::3] == best
    torch.manual_seed(0)
    both = dec.generate(c, max_len=6, beam=4, nbest=3, samples=2)  # LLM decoders: each row's beams, then its samples
    assert len(both) == 15 and [both[5 * i : 5 * i + 3] for i in range(3)] == [nbest[3 * i : 3 * i + 3] for i in range(3)]


def test_rerank_keeps_the_candidate_closest_to_each_row(tok):
    from prh.models import rerank

    enc = TextEncoder(tok.pad_token_id, arch="tiny").eval()
    cands = ["a cat on a sofa", "a dog on a beach", "two red cars", "a plate of food"]
    z = enc.encode([cands[1], cands[2]]).double()
    assert rerank(cands, z, enc, torch.zeros(z.shape[1], dtype=torch.float64)) == [cands[1], cands[2]]


def test_lm_decoder_learns_the_written_case_when_given_captions(tok, tiny_lm):
    dec = LMDecoder(tiny_lm, cond_dim=16, n_prefix=2)
    ids = tok("a dog in Paris", return_tensors="pt")["input_ids"]
    c = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(100):
        opt.zero_grad()
        dec.loss(c, ids, ["A dog in Paris"]).backward()
        opt.step()
    assert dec.eval().generate(c, max_len=8) == ["A dog in Paris"]
