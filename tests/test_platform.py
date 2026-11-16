# -*- coding: utf-8 -*-
import json
import os
import threading
import urllib.request

import pytest
import torch

from octo.model import OctoConfig, OctoModel

SAMPLE = "octo is a small model. it learns from text."


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    cfg = OctoConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = OctoModel(cfg).eval()
    path = tmp / "octo.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, path)
    from octo.tokenizer import CharTokenizer
    CharTokenizer.from_text(SAMPLE).save(str(tmp / "tokenizer.json"))
    return str(path)


class TestServer:
    def test_health_and_chat(self, tiny_checkpoint):
        from octo import server
        server._state.update({"model": None, "tokenizer": None, "device": "cpu", "history": []})
        httpd = server.make_server(tiny_checkpoint, host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
                assert json.load(r)["status"] == "ok"
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/chat",
                data=json.dumps({"prompt": "hello octo", "tokens": 10}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.load(r)
            assert "response" in body
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
                assert b"Octo" in r.read()
        finally:
            httpd.shutdown()
            server._state["history"] = []

    def test_chat_requires_prompt(self, tiny_checkpoint):
        from octo import server
        server._state.update({"model": None, "tokenizer": None, "device": "cpu", "history": []})
        httpd = server.make_server(tiny_checkpoint, host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/chat",
                                         data=b"{}", headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req, timeout=10)
            assert exc.value.code == 400
        finally:
            httpd.shutdown()
            server._state["history"] = []


class TestCrypto:
    def test_disabled_by_default(self, tmp_path, monkeypatch):
        import storage.crypto as crypto
        monkeypatch.setattr(crypto, "VERIFY_FILE", str(tmp_path / ".crypto_verify"))
        assert not crypto.is_encryption_enabled()

    def test_unlock_roundtrip(self, tmp_path, monkeypatch):
        import storage.crypto as crypto
        monkeypatch.setattr(crypto, "VERIFY_FILE", str(tmp_path / ".crypto_verify"))
        pytest.importorskip("cryptography")
        em = crypto.EncryptionManager()
        assert em.unlock("correct horse")
        assert crypto.is_encryption_enabled()
        token = em.encrypt("secret memory")
        assert em.decrypt(token) == "secret memory"

    def test_wrong_password_rejected(self, tmp_path, monkeypatch):
        import storage.crypto as crypto
        verify = tmp_path / ".crypto_verify"
        monkeypatch.setattr(crypto, "VERIFY_FILE", str(verify))
        pytest.importorskip("cryptography")
        crypto.EncryptionManager().unlock("right-password")
        em = crypto.EncryptionManager()
        assert not em.unlock("wrong-password")


class TestSpeechFallback:
    def test_text_mode_without_stt(self, monkeypatch):
        import speech.engine as engine
        se = engine.SpeechEngine.__new__(engine.SpeechEngine)
        se._tts = None
        se._recognizer = None
        assert not se.is_stt_ready()
        assert not se.is_tts_ready()
        assert not se.is_wakeword_ready()
        assert se.wait_for_wakeword(0) is False
        assert se.listen(1) == ""
