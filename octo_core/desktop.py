# -*- coding: utf-8 -*-
"""OctoSession — display-independent logic for the Octo desktop app.

Everything the GUI does goes through this class, so the desktop app's
behavior is fully testable without a display or Tkinter.
"""
import os

from .cognitive import DecisionEngine
from .evaluation import AgentMetrics, BenchmarkSuite
from .infrastructure import ResourceManager
from .learning import SkillManager
from .main import run_goal
from .memory import MemoryManager
from .security import AuditLogger, detect_prompt_injection, redact_secrets

CHECKPOINT = os.path.join("octo_checkpoints", "octo.pt")


class OctoSession:
    """Facade over Octo Core for UI layers (Tk, API, tests)."""

    def __init__(self, roots=("data/octo",), checkpoint=CHECKPOINT):
        self.roots = roots
        self.checkpoint = checkpoint
        self.memory = MemoryManager(roots=roots)
        self.audit = AuditLogger()
        self.skills = SkillManager()
        self.suite = BenchmarkSuite(roots=roots)
        self._router = None

    def status(self):
        backend, model = self._router_info()
        return {
            "device": ResourceManager.pick_device(),
            "resources": ResourceManager.report(),
            "checkpoint": os.path.exists(self.checkpoint),
            "model_backend": backend,
            "model_ready": model is not None,
            "knowledge_facts": len(self.memory.graph),
            "skills": len(self.skills._load()),
            "audit_entries": len(self.audit.tail(1000)),
        }

    def _router_info(self):
        if self._router is None:
            from .models import ModelRouter
            self._router = ModelRouter(checkpoint=self.checkpoint)
        return self._router.route("grounded_answer")

    def chat(self, prompt, steps=100):
        """RAG-grounded Octo LLM answer, or guidance when untrained."""
        risk, _ = detect_prompt_injection(prompt)
        if risk == "high":
            return "BLOCKED: এই ইনপুটে prompt-injection প্যাটার্ন ধরা পড়েছে।"
        self.memory.observe(prompt)
        ready = self._router_info()[1]
        if ready is None:
            return ("Octo এখনো train হয়নি। চালান: "
                    "python -m octo.train --corpus data/octo")
        answer = self._router.answer(prompt, steps=steps)
        self.memory.working.push(answer)
        return redact_secrets(answer)

    def run_goal(self, goal, max_steps=8):
        """Full agent pipeline; returns (summary_lines, result_dict)."""
        result = run_goal(goal, max_steps=max_steps, verbose=False)
        if result["blocked"]:
            return [f"BLOCKED (risk={result['risk']}, "
                    f"patterns={len(result['matches'])})"], result
        lines = [f"strategy: {result['strategy']} | "
                 f"success: {result['metrics']['success_rate']}"]
        for tool, output in result["trace"]:
            lines.append(f"[{tool}] {redact_secrets(str(output))[:220]}")
        r = result["reflection"]
        lines.append(f"reflection: {r['steps']} steps, {r['failures']} failures")
        if r["next_strategy"]:
            lines.append(f"next: {r['next_strategy']}")
        return lines, result

    def remember(self, subject, relation, obj):
        fact = self.memory.remember(subject, relation, obj, source="desktop")
        return f"সংরক্ষিত: {subject} {relation} {obj} (source={fact['source']})"

    def recall(self, query, k=3):
        hits = self.memory.recall(query, k=k)
        return [f"({h['origin']}, {h['score']}) {h['text'][:160]}" for h in hits] \
            or ["কিছু পাওয়া যায়নি"]

    def timeline(self, n=15):
        return self.memory.episodic.timeline(n=n) or ["(খালি)"]

    def injection_check(self, text):
        risk, matches = detect_prompt_injection(text)
        return f"risk={risk}, matched patterns={len(matches)}"

    def audit_tail(self, n=15):
        entries = self.audit.tail(n)
        return [f"{e['ts']} [{e['agent']}] {e['event']}: {e['detail'][:90]}"
                for e in entries] or ["(খালি)"]

    def benchmarks(self):
        return self.suite.run()

    def agent_metrics(self, trace):
        return AgentMetrics.from_trace(trace)

    def skill_list(self):
        return [f"{s['trigger']} -> {' > '.join(s['workflow'])}"
                for s in self.skills._load()] or ["(কোনো skill শেখা হয়নি)"]

    def strategy_for(self, goal):
        return DecisionEngine().choose(goal)


class OctoDesktopApp:
    """Tkinter desktop application — tabbed control center for Octo Core.

    Tabs: Chat · Agent · Memory · Security · Benchmarks · Status.
    Long work runs in daemon threads; results flow back through a queue.
    """

    def __init__(self, root, session=None):
        import tkinter as tk
        from tkinter import ttk
        import tkinter.scrolledtext as scrolledtext

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.session = session or OctoSession()
        root.title("🐙 Octo Desktop — LLM + Agent Control Center")
        root.geometry("860x620")

        self.queue = []
        root.after(100, self._poll)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.chat_out = self._tab_with_io("💬 Chat", "প্রশ্ন লিখুন...", self.send_chat)
        self.agent_out = self._tab_with_io("🤖 Agent", "লক্ষ্য লিখুন: research sql injection ...",
                                           self.send_goal)
        self._build_memory_tab()
        self._build_security_tab()
        self._build_bench_tab()
        self._build_status_tab()

    def _tab_with_io(self, title, placeholder, on_send):
        tk, ttk = self.tk, self.ttk
        import tkinter.scrolledtext as scrolledtext
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        out = scrolledtext.ScrolledText(frame, wrap=tk.WORD, state=tk.DISABLED,
                                        font=("TkDefaultFont", 11))
        out.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))
        bottom = ttk.Frame(frame)
        bottom.pack(fill=tk.X, padx=6, pady=(2, 6))
        entry = ttk.Entry(bottom)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.insert(0, placeholder)
        entry.bind("<Return>", lambda e: on_send(entry))
        entry.bind("<FocusIn>", lambda e: entry.delete(0, tk.END)
                   if entry.get() == placeholder else None)
        send = ttk.Button(bottom, text="Run", width=10, command=lambda: on_send(entry))
        send.pack(side=tk.LEFT, padx=(6, 0))
        entry._placeholder = placeholder
        return out

    def _append(self, widget, text):
        widget.configure(state=self.tk.NORMAL)
        widget.insert(self.tk.END, text.rstrip() + "\n\n")
        widget.see(self.tk.END)
        widget.configure(state=self.tk.DISABLED)

    def _read_entry(self, entry):
        text = entry.get().strip()
        ph = getattr(entry, "_placeholder", None)
        if not text or (ph and text == ph):
            return None
        return text

    def send_chat(self, entry):
        text = self._read_entry(entry)
        if not text:
            return
        self._append(self.chat_out, f"you: {text}")
        entry.delete(0, self.tk.END)
        self._run_bg(self.session.chat, text, on_done=self._chat_done)

    def _chat_done(self, answer):
        self._append(self.chat_out, f"octo: {answer}")

    def send_goal(self, entry):
        text = self._read_entry(entry)
        if not text:
            return
        self._append(self.agent_out, f"goal: {text}")
        entry.delete(0, self.tk.END)
        name, strat = self.session.strategy_for(text)
        self._append(self.agent_out, f"strategy: {name} (max {strat['max_steps']} steps)")
        self._run_bg(self.session.run_goal, text, on_done=self._goal_done)

    def _goal_done(self, payload):
        if isinstance(payload, str):
            self._append(self.agent_out, payload)
            return
        lines, result = payload
        for line in lines:
            self._append(self.agent_out, line)

    def _build_memory_tab(self):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🧠 Memory")
        form = ttk.Frame(frame)
        form.pack(fill=tk.X, padx=6, pady=6)
        self.mem_subject = ttk.Entry(form, width=18)
        self.mem_relation = ttk.Entry(form, width=14)
        self.mem_object = ttk.Entry(form, width=24)
        for i, (w, label) in enumerate(((self.mem_subject, "subject"),
                                        (self.mem_relation, "relation"),
                                        (self.mem_object, "object"))):
            w.pack(side=tk.LEFT, padx=2)
        ttk.Button(form, text="Remember", command=self.on_remember).pack(side=tk.LEFT, padx=4)
        query_row = ttk.Frame(frame)
        query_row.pack(fill=tk.X, padx=6)
        self.mem_query = ttk.Entry(query_row)
        self.mem_query.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(query_row, text="Recall", command=self.on_recall).pack(side=tk.LEFT, padx=4)
        ttk.Button(query_row, text="Timeline", command=self.on_timeline).pack(side=tk.LEFT)
        self.mem_out = self._output_box(frame)

    def _build_security_tab(self):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🛡 Security")
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=6, pady=6)
        self.sec_input = ttk.Entry(row)
        self.sec_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Inject check", command=self.on_inject).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Audit log", command=self.on_audit).pack(side=tk.LEFT)
        self.sec_out = self._output_box(frame)

    def _build_bench_tab(self):
        ttk = self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Benchmarks")
        ttk.Button(frame, text="Run benchmark suite", command=self.on_bench).pack(pady=6)
        self.bench_out = self._output_box(frame)

    def _build_status_tab(self):
        ttk = self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙ Status")
        ttk.Button(frame, text="Refresh status", command=self.on_status).pack(pady=6)
        self.status_out = self._output_box(frame)
        self.on_status()

    def _output_box(self, parent):
        tk = self.tk
        import tkinter.scrolledtext as scrolledtext
        out = scrolledtext.ScrolledText(parent, wrap=tk.WORD, state=tk.DISABLED,
                                        font=("TkFixedFont", 10), height=18)
        out.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        return out

    def on_remember(self):
        s, r, o = (self.mem_subject.get().strip(), self.mem_relation.get().strip(),
                   self.mem_object.get().strip())
        if s and r and o:
            self._append(self.mem_out, self.session.remember(s, r, o))
            for w in (self.mem_subject, self.mem_relation, self.mem_object):
                w.delete(0, self.tk.END)

    def on_recall(self):
        q = self.mem_query.get().strip()
        if q:
            for line in self.session.recall(q):
                self._append(self.mem_out, line)

    def on_timeline(self):
        for line in self.session.timeline():
            self._append(self.mem_out, line)

    def on_inject(self):
        text = self.sec_input.get().strip()
        if text:
            self._append(self.sec_out, self.session.injection_check(text))

    def on_audit(self):
        for line in self.session.audit_tail():
            self._append(self.sec_out, line)

    def on_bench(self):
        self._append(self.bench_out, "চলছে...")
        self._run_bg(self.session.benchmarks, on_done=self._bench_done)

    def _bench_done(self, report):
        for name, res in report.items():
            self._append(self.bench_out, f"{name}: {res}")

    def on_status(self):
        st = self.session.status()
        lines = [f"device: {st['device']}",
                 f"checkpoint: {'✓' if st['checkpoint'] else '✗ (train করুন)'}",
                 f"model backend: {st['model_backend']} "
                 f"({'ready' if st['model_ready'] else 'not trained'})",
                 f"knowledge facts: {st['knowledge_facts']}",
                 f"skills: {st['skills']}",
                 f"audit entries: {st['audit_entries']}",
                 f"resources: {st['resources']}"]
        self.status_out.configure(state=self.tk.NORMAL)
        self.status_out.delete("1.0", self.tk.END)
        self.status_out.configure(state=self.tk.DISABLED)
        for line in lines:
            self._append(self.status_out, line)

    def _run_bg(self, fn, *args, on_done=None):
        import threading
        def worker():
            try:
                result = fn(*args)
                self.queue.append((on_done, result))
            except Exception as e:
                self.queue.append((on_done, f"error: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        while self.queue:
            on_done, result = self.queue.pop(0)
            if on_done:
                on_done(result)
        self.root.after(100, self._poll)


def main():
    import tkinter as tk
    root = tk.Tk()
    OctoDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
