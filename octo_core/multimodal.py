# -*- coding: utf-8 -*-
"""Advanced Multimodal Intelligence — optional-dep adapters with graceful fallback.

Vision metadata, OCR, audio transcription and a cross-modal retrieval index.
Heavy backends (Pillow, pytesseract, vosk) are optional; every function
degrades to a useful text-only answer when the dependency is missing.
"""
import datetime
import os
import re


def describe_image(path):
    """Image Understanding Engine (metadata tier): size, mode, aspect, type."""
    if not os.path.exists(path):
        return f"image not found: {path}"
    info = {"path": path, "name": os.path.basename(path),
            "bytes": os.path.getsize(path)}
    try:
        from PIL import Image
        with Image.open(path) as im:
            info.update({"width": im.width, "height": im.height,
                         "mode": im.mode, "format": im.format})
            info["aspect"] = round(im.width / max(im.height, 1), 2)
    except ImportError:
        info["note"] = "Pillow not installed; metadata only"
    return " | ".join(f"{k}={v}" for k, v in info.items())


def ocr_image(path, lang="eng"):
    """OCR & Document Intelligence: pytesseract if available, else guidance."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "OCR unavailable: install pytesseract + tesseract binary"
    if not os.path.exists(path):
        return f"image not found: {path}"
    try:
        text = pytesseract.image_to_string(Image.open(path), lang=lang)
        return text.strip() or "(no text detected)"
    except Exception as e:
        return f"OCR failed: {e}"


def transcribe_audio(path):
    """Real-Time Audio Intelligence (offline tier): vosk if models installed."""
    try:
        import wave
        if not os.path.exists(path):
            return f"audio not found: {path}"
        with wave.open(path, "rb") as w:
            return (f"audio={os.path.basename(path)} channels={w.getnchannels()} "
                    f"rate={w.getframerate()} seconds={w.getnframes() / w.getframerate():.1f} "
                    "(transcription needs a vosk model; see speech/engine.py)")
    except Exception as e:
        return f"audio analysis failed: {e}"


class CrossModalIndex:
    """Cross-Modal Retrieval: tag-based index over text/image/audio entries."""

    def __init__(self):
        self._entries = []

    def add(self, modality, path_or_text, tags=()):
        entry = {"modality": modality, "ref": str(path_or_text)[:200],
                 "tags": {t.lower() for t in tags},
                 "ts": datetime.datetime.now().isoformat(timespec="seconds")}
        self._entries.append(entry)
        return entry

    def search(self, query, modality=None):
        q = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
        scored = []
        for e in self._entries:
            if modality and e["modality"] != modality:
                continue
            score = len(q & e["tags"]) / max(len(q), 1)
            if score:
                scored.append({"ref": e["ref"], "modality": e["modality"],
                               "score": round(score, 2)})
        return sorted(scored, key=lambda s: -s["score"])
