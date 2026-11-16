# -*- coding: utf-8 -*-
import json
import math
import os

import numpy as np
import pytest
import torch

from octo.data import encode_corpus, get_batch
from octo.model import OctoConfig, OctoModel
from octo.nnet import NumPyMLP, featurize, predict, tokenize, train
from octo.rag import RagIndex
from octo.summarize import summarize
from octo.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer
from octo.agi import safe_calc, tool_convert

SAMPLE = "octo is a small model. it learns from text. bangla: অক্টো ভালো।\nsecond line here."


def write_corpus(tmp_path, text=SAMPLE):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def tiny_model(vocab_size=64, block_size=32):
    cfg = OctoConfig(vocab_size=vocab_size, block_size=block_size,
                     n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    torch.manual_seed(0)
    return OctoModel(cfg).eval()


class TestTokenizers:
    def test_char_roundtrip(self):
        tok = CharTokenizer.from_text(SAMPLE)
        assert tok.decode(tok.encode(SAMPLE)) == SAMPLE

    def test_char_unicode(self):
        tok = CharTokenizer.from_text("বাংলা abc")
        assert tok.decode(tok.encode("বাংলা")) == "বাংলা"

    def test_bpe_roundtrip(self):
        tok = BPETokenizer.train(SAMPLE * 20, vocab_size=120)
        text = "octo learns from text"
        assert tok.decode(tok.encode(text)).split() == text.split()

    def test_bpe_unknown_chars(self):
        tok = BPETokenizer.train(SAMPLE, vocab_size=100)
        ids = tok.encode("zzz qqq")
        assert all(0 <= i < len(tok) for i in ids)

    def test_save_load_roundtrip(self, tmp_path):
        tok = BPETokenizer.train(SAMPLE * 5, vocab_size=90)
        path = str(tmp_path / "tok.json")
        tok.save(path)
        loaded = load_tokenizer(path)
        assert len(loaded) == len(tok)
        assert loaded.encode("octo text") == tok.encode("octo text")

    def test_bpe_compresses(self):
        tok = BPETokenizer.train(SAMPLE * 30, vocab_size=200)
        text = "octo is a small model"
        assert len(tok.encode(text)) < len(text)


class TestModel:
    def test_forward_shapes(self):
        model = tiny_model()
        x = torch.randint(0, 64, (2, 16))
        logits, loss = model(x, x)
        assert logits.shape == (2, 16, 64)
        assert math.isfinite(loss.item())

    def test_generate_shape(self):
        model = tiny_model()
        out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=10, top_k=5)
        assert out.shape == (1, 13)

    def test_generation_deterministic_with_seed(self):
        model = tiny_model()
        torch.manual_seed(7)
        a = model.generate(torch.tensor([[1]]), max_new_tokens=5)
        torch.manual_seed(7)
        b = model.generate(torch.tensor([[1]]), max_new_tokens=5)
        assert torch.equal(a, b)

    def test_param_count(self):
        model = tiny_model()
        assert model.num_params() > 0

    def test_training_reduces_loss(self):
        torch.manual_seed(0)
        model = tiny_model()
        x = torch.randint(0, 64, (4, 32))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        first = model(x, x)[1].item()
        for _ in range(30):
            opt.zero_grad()
            model(x, x)[1].backward()
            opt.step()
        assert model(x, x)[1].item() < first


class TestData:
    def test_get_batch_shapes(self):
        tok = CharTokenizer.from_text(SAMPLE * 10)
        data = encode_corpus(SAMPLE * 10, tok)
        x, y = get_batch(data, block_size=16, batch_size=4, device="cpu")
        assert x.shape == (4, 16) and y.shape == (4, 16)
        assert (y[:, :-1] == x[:, 1:]).all()


class TestRag:
    def test_search_relevance(self, tmp_path):
        write_corpus(tmp_path, "sql injection is prevented by parameterized queries. "
                                "firewalls filter network traffic. cats are cute animals.")
        index = RagIndex.build(roots=(str(tmp_path),))
        hits = index.search("how to prevent sql injection", k=1)
        assert hits and "sql" in hits[0][0].lower()

    def test_empty_query(self, tmp_path):
        write_corpus(tmp_path)
        index = RagIndex.build(roots=(str(tmp_path),))
        assert index.search("zzzzunknownword", k=2) == []


class TestNNet:
    def test_tokenize(self):
        assert tokenize("Remember, this!") == ["remember", "this"]

    def test_train_and_predict(self, tmp_path):
        data = str(tmp_path / "intents.json")
        with open(data, "w", encoding="utf-8") as f:
            json.dump([{"text": "remember this", "intent": "note"},
                       {"text": "write a note", "intent": "note"},
                       {"text": "weather today", "intent": "weather"},
                       {"text": "is it raining", "intent": "weather"}], f)
        blob = train(data, epochs=150)
        assert predict("remember my keys", blob)[0][0] == "note"
        assert predict("weather now please", blob)[0][0] == "weather"

    def test_mlp_shapes(self):
        net = NumPyMLP(n_in=10, n_hidden=4, n_out=3)
        X = np.random.rand(5, 10).astype(np.float32)
        assert net.forward(X).shape == (5, 3)
        y = np.array([0, 1, 2, 1, 0])
        grads = net.backward(X, y)
        assert len(grads) == 4 and all(g.shape == p.shape for g, p in zip(grads, net._params()))


class TestAgiTools:
    def test_safe_calc(self):
        assert safe_calc("(12 + 8) * 4") == 80
        assert safe_calc("2 ** 10") == 1024

    def test_safe_calc_rejects_code(self):
        with pytest.raises(ValueError):
            safe_calc("__import__('os').system('ls')")

    def test_convert(self):
        assert "259200" in tool_convert("seconds in 3 days")
        assert "unknown units" in tool_convert("unicorns in 3 days")


class TestSearch:
    def test_decode_bing_url(self):
        import base64
        from octo.search import _decode_bing_url
        target = "https://owasp.org/www-community/attacks/SQL_Injection"
        b64 = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        href = f"https://www.bing.com/ck/a?!&amp;&amp;u=a1{b64}&ntb=1"
        assert _decode_bing_url(href) == target

    def test_decode_plain_url(self):
        from octo.search import _decode_bing_url
        assert _decode_bing_url("https://example.com/page") == "https://example.com/page"


class TestSummarize:
    def test_summary_is_subset(self):
        summary = summarize(SAMPLE * 5, n=2)
        assert any(s in SAMPLE * 5 for s in summary.split(". ")[:1])

    def test_short_text_fallback(self):
        assert summarize("tiny") == "tiny"


class TestQuantize:
    def test_quantize_smaller_and_loadable(self, tmp_path):
        from octo.quantize import quantize
        cfg = OctoConfig(vocab_size=64, block_size=64, n_layer=2, n_head=4, n_embd=128, dropout=0.0)
        torch.manual_seed(0)
        model = OctoModel(cfg).eval()
        ckpt = str(tmp_path / "octo.pt")
        torch.save({"model_state": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
        tok = CharTokenizer.from_text(SAMPLE)
        tok.save(str(tmp_path / "tokenizer.json"))
        out = quantize(ckpt)
        assert os.path.getsize(out) < os.path.getsize(ckpt)
        from octo.chat import load_model
        qmodel, _ = load_model(out, "cpu")
        logits, _ = qmodel(torch.randint(0, 64, (1, 8)))
        assert logits.shape == (1, 8, 64)
