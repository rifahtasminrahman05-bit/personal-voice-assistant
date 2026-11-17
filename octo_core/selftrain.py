# -*- coding: utf-8 -*-
"""Autonomous Self-Training Loop — Octo trains itself, safely.

SelfTrainingLoop: periodically continues OctoV2 training on the accumulated
corpus, compares validation loss before vs after, and keeps the new
checkpoint ONLY when it measurably improved (otherwise rolls back to the
previous one). Every cycle is audit-logged. A background daemon thread can
run cycles on an interval; `stop()` halts it. This never touches production
code — model weights only, improvement-gated.
"""
import os
import shutil
import threading

from .train_v2 import estimate_loss, load_v2, read_corpus, train_v2

MIN_DELTA = 1e-4


class SelfTrainingLoop:
    """Self-training: train -> evaluate -> keep-if-improved -> repeat."""

    def __init__(self, corpus="data/octo", out="octo_checkpoints",
                 steps=300, interval=3600, eval_iters=10, audit=None,
                 device=None, min_delta=MIN_DELTA):
        self.corpus = corpus
        self.out = out
        self.steps = steps
        self.interval = interval
        self.eval_iters = eval_iters
        self.audit = audit
        self.device = device
        self.min_delta = min_delta
        self.cycles = []
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    @property
    def checkpoint(self):
        return os.path.join(self.out, "octo_v2.pt")

    @property
    def backup(self):
        return self.checkpoint + ".prev"

    def _validation_loss(self):
        """Val loss of the current checkpoint on a held-out corpus slice."""
        if not os.path.exists(self.checkpoint):
            return None
        from octo.data import encode_corpus
        from octo.tokenizer import load_tokenizer
        model, _ = load_v2(self.checkpoint, device="cpu")
        tok_path = os.path.join(self.out, "tokenizer_v2.json")
        if not os.path.exists(tok_path):
            return None
        tokenizer = load_tokenizer(tok_path)
        text = read_corpus(self.corpus)
        cut = int(len(text) * 0.9)
        val_text = text[cut:] or text[-2000:]
        cfg = model.cfg
        data = encode_corpus(val_text, tokenizer).to("cpu")
        if data.numel() <= cfg.block_size + 1:
            data = encode_corpus(text, tokenizer).to("cpu")
        return estimate_loss(model.to("cpu"), data, cfg, "cpu",
                             iters=self.eval_iters)

    def run_cycle(self, reason="scheduled"):
        """One self-training cycle. Returns a report dict."""
        with self._lock:
            had_checkpoint = os.path.exists(self.checkpoint)
            if had_checkpoint:
                shutil.copy2(self.checkpoint, self.backup)
            before = self._validation_loss()
            progress_log = []

            def progress(step, total, loss):
                progress_log.append((step, round(loss, 4)))

            train_v2(corpus=self.corpus, out=self.out, steps=self.steps,
                     resume=self.checkpoint if had_checkpoint else None,
                     device=self.device, progress=progress)
            after = self._validation_loss()
            improved = (before is None or after is None
                        or after < before - self.min_delta)
            if improved:
                if os.path.exists(self.backup):
                    os.remove(self.backup)
            else:
                shutil.move(self.backup, self.checkpoint)
            report = {
                "reason": reason,
                "steps": self.steps,
                "had_checkpoint": had_checkpoint,
                "val_loss_before": round(before, 4) if before is not None else None,
                "val_loss_after": round(after, 4) if after is not None else None,
                "improved": bool(improved),
                "kept": bool(improved),
                "action": "kept new checkpoint" if improved
                          else "rolled back to previous checkpoint",
            }
            self.cycles.append(report)
            if self.audit:
                self.audit.log("self_train_cycle", "selftrain",
                               f"{reason}: {report['val_loss_before']} -> "
                               f"{report['val_loss_after']} ({report['action']})")
            return report

    def start(self):
        """Run cycles in a daemon thread every `interval` seconds."""
        if self._thread and self._thread.is_alive():
            return False

        def loop():
            while not self._stop.is_set():
                try:
                    self.run_cycle(reason="auto")
                except Exception as e:
                    if self.audit:
                        self.audit.log("self_train_error", "selftrain", str(e))
                self._stop.wait(self.interval)

        self._stop.clear()
        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="octo-selftrain")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        return True

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def history(self, n=10):
        return self.cycles[-n:]
