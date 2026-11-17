# -*- coding: utf-8 -*-
"""Octo Chat GUI — Tkinter window for chatting with the built-in LLM.

Run: python gui.py
Requires a trained checkpoint: python -m octo.train --corpus data/octo
"""
import os
import queue
import threading
import tkinter as tk
import tkinter.scrolledtext as scrolledtext

from core import config

CHECKPOINT = os.path.join(config.BASE_DIR, "octo_checkpoints", "octo.pt")


class OctoChatGUI:
    def __init__(self, root):
        self.root = root
        root.title("Octo Chat — offline LLM")
        root.geometry("640x520")

        self.log = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED,
                                             font=("TkDefaultFont", 11))
        self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.entry = tk.Entry(bottom, font=("TkDefaultFont", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = tk.Button(bottom, text="Send", width=10, command=self.send)
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.queue = queue.Queue()
        self.history = []
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        self.append("octo", "লোড হচ্ছে... (প্রথমবার কয়েক সেকেন্ড লাগতে পারে)")
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._poll_queue)

    def append(self, who, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"{who}: {text}\n\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _load_model(self):
        try:
            import torch
            from octo.chat import load_model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.tokenizer = load_model(CHECKPOINT, self.device)
            self.queue.put(("octo", f"প্রস্তুত! (device={self.device}) প্রশ্ন লিখুন।"))
        except Exception as e:
            self.queue.put(("octo", f"মডেল লোড হয়নি: {e}\nট্রেন করুন: python -m octo.train --corpus data/octo"))

    def _poll_queue(self):
        try:
            while True:
                who, text = self.queue.get_nowait()
                self.append(who, text)
                self.send_btn.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def send(self):
        prompt = self.entry.get().strip()
        if not prompt or self.model is None:
            return
        self.entry.delete(0, tk.END)
        self.append("you", prompt)
        self.send_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._generate, args=(prompt,), daemon=True).start()

    def _generate(self, prompt):
        from octo.chat import complete
        try:
            reply = complete(self.model, self.tokenizer, prompt, 200, 0.8, 40,
                             self.device, history=self.history[-3:])
            self.history.append((prompt, reply))
            self.queue.put(("octo", reply))
        except Exception as e:
            self.queue.put(("octo", f"error: {e}"))


def main():
    root = tk.Tk()
    OctoChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
