# -*- coding: utf-8 -*-
"""PersonalBrain — minimal command router with Octo LLM integration."""
import os

from core import config

OCTO_CHECKPOINT = os.path.join(config.BASE_DIR, "octo_checkpoints", "octo.pt")


class PersonalBrain:
    def __init__(self):
        self._octo = None
        self._octo_checked = False

    def _load_octo(self):
        if self._octo_checked:
            return self._octo
        self._octo_checked = True
        if not os.path.exists(OCTO_CHECKPOINT):
            return None
        try:
            import torch
            from octo.chat import load_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._octo = (load_model(OCTO_CHECKPOINT, device), device)
        except Exception as e:
            print("octo load failed:", e)
            self._octo = None
        return self._octo

    def ask_octo(self, prompt):
        loaded = self._load_octo()
        if loaded is None:
            return ("Octo এখনো train করা হয়নি। চালান: "
                    "python -m octo.train --corpus data/octo --steps 2000")
        (model, tokenizer), device = loaded
        from octo.chat import complete
        try:
            from octo.rag import context_for_prompt
            context = context_for_prompt(prompt)
            if context:
                prompt = context + "\n\n" + prompt
        except Exception:
            pass
        return complete(model, tokenizer, prompt, 200, 0.8, 40, device).strip()

    def run_agi(self, goal):
        from octo.agi import run_agent
        trace = run_agent(goal, max_steps=6, verbose=False)
        lines = [f"[{name}] {result[:150]}" for name, result in trace]
        return "AGI রিপোর্ট:\n" + "\n".join(lines)

    def process(self, text):
        low = text.lower().strip()
        if low in ("সাহায্য", "help"):
            return ("কমান্ড: অক্টো <প্রশ্ন> — Octo LLM উত্তর | agi <লক্ষ্য> — এজেন্ট মোড | "
                    "মনে রাখো <বিষয়> — নোট | সেলফ রিপোর্ট | bye")
        if low.startswith("agi ") or low.startswith("এজিআই "):
            goal = text.strip().split(" ", 1)[1]
            return self.run_agi(goal)
        if low.startswith("অক্টো") or low.startswith("octo"):
            parts = text.strip().split(" ", 1)
            prompt = parts[1] if len(parts) > 1 else "hello"
            return self.ask_octo(prompt)
        if low.startswith("মনে রাখো"):
            note = text.strip()[len("মনে রাখো"):].strip()
            return f"মনে রাখলাম: {note}" if note else "কী মনে রাখব?"
        if low in ("সেলফ রিপোর্ট", "self report"):
            octo_ok = self._load_octo() is not None
            return f"Assistant: {config.ASSISTANT_NAME} v{config.APP_VERSION} | Octo: {'ready' if octo_ok else 'not trained'}"
        intent = self._guess_intent(text)
        if intent == "note":
            return self.process("মনে রাখো " + text)
        if intent == "llm_agent":
            return self.ask_octo(text)
        if intent == "status":
            return self.process("সেলফ রিপোর্ট")
        if intent == "weather":
            return ("আবহাওয়া ফিচার এখনো যুক্ত হয়নি। শেখার জন্য: "
                    "agi research weather")
        return ("বুঝতে পারিনি। 'সাহায্য' লিখুন, অথবা 'অক্টো <প্রশ্ন>' দিয়ে built-in LLM-কে জিজ্ঞেস করুন।")

    def _guess_intent(self, text):
        try:
            from octo import nnet
            if not hasattr(self, "_nnet"):
                self._nnet = nnet.load()
            return nnet.predict(text, self._nnet)[0][0]
        except Exception:
            pass
        try:
            from octo.classifier import IntentClassifier
            if not hasattr(self, "_clf"):
                self._clf = IntentClassifier.load()
            return self._clf.predict(text)[0][0]
        except Exception:
            return None
