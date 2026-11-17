# -*- coding: utf-8 -*-
"""OctoSession — display-independent logic for the Octo desktop app.

Everything the GUI does goes through this class, so the desktop app's
behavior is fully testable without a display or Tkinter.
"""
import os

from .benchhistory import BenchmarkHistory
from .cognitive import DecisionEngine
from .evaluation import AgentMetrics, BenchmarkSuite, HallucinationDetector
from .evolution import RedTeamSuite, capability_manifest
from .infrastructure import ResourceManager
from .jobs import JobScheduler
from .learning import SkillManager
from .main import run_goal
from .memory import MemoryManager
from .security import AuditLogger, detect_prompt_injection, redact_secrets
from .settings import Settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(BASE_DIR, "octo_checkpoints", "octo.pt")


class OctoSession:
    """Facade over Octo Core for UI layers (Tk, API, tests)."""

    def __init__(self, roots=("data/octo",), checkpoint=CHECKPOINT):
        self.roots = roots
        self.checkpoint = checkpoint
        self.settings = Settings()
        self.memory = MemoryManager(roots=roots)
        self.audit = AuditLogger()
        self.skills = SkillManager()
        self.suite = BenchmarkSuite(roots=roots)
        self.history = BenchmarkHistory()
        self.detector = HallucinationDetector(roots=roots)
        self.redteam = RedTeamSuite()
        self._router = None
        self._jobs = None
        self._selftrain_loop = None

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
            "quantized": self.settings.get("quantized"),
        }

    def _router_info(self):
        if self._router is None:
            from .models import ModelRouter
            self._router = ModelRouter(checkpoint=self.checkpoint,
                                       quantized=self.settings.get("quantized"))
        return self._router.route("grounded_answer")

    def chat(self, prompt, steps=None):
        """RAG-grounded Octo LLM answer (settings-driven), or guidance."""
        risk, _ = detect_prompt_injection(prompt)
        if risk == "high":
            return "BLOCKED: এই ইনপুটে prompt-injection প্যাটার্ন ধরা পড়েছে।"
        self.memory.observe(prompt)
        ready = self._router_info()[1]
        if ready is None:
            return ("Octo এখনো train হয়নি। চালান: "
                    "python -m octo.train --corpus data/octo")
        answer = self._router.answer(prompt, steps=steps or self.settings.get("steps"))
        self.memory.working.push(answer)
        return redact_secrets(answer)

    def chat_checked(self, prompt):
        """Chat + hallucination guard: returns (answer, confidence_badge)."""
        answer = self.chat(prompt)
        if answer.startswith("BLOCKED") or answer.startswith("Octo এখনো"):
            return answer, ""
        check = self.detector.check(answer)
        badge = (f"grounding: {check['verdict']} "
                 f"({check['grounded_ratio']:.0%})")
        return answer, badge

    def run_goal(self, goal, max_steps=None, approve=None):
        """Full agent pipeline; returns (summary_lines, result_dict)."""
        result = run_goal(goal, max_steps=max_steps or self.settings.get("max_steps"),
                          approve=approve, verbose=False)
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

    def run_goal_parallel(self, goal):
        """Parallel multi-agent execution; returns (lines, reflection)."""
        from .agents import AgentOrchestrator
        orch = AgentOrchestrator(approve=None)
        trace, reflection = orch.execute_parallel(goal, verbose=False)
        lines = [f"[{name}] {redact_secrets(out)[:200]}" for name, out in trace]
        lines.append(f"parallel reflection: {reflection['steps']} steps, "
                     f"{reflection['failures']} failures")
        return lines, reflection

    def train(self, corpus, steps, progress=None):
        """OctoV2 training from the desktop UI (runs in a background thread)."""
        from .train_v2 import train_v2
        try:
            return train_v2(corpus=corpus, steps=int(steps), progress=progress)
        except SystemExit as e:
            raise ValueError(f"training failed: {e}")

    def benchmarks_recorded(self):
        report = self.suite.run()
        self.history.record(report)
        trend = self.history.trend(("math", "score"))
        return report, trend

    def redteam_run(self):
        return self.redteam.run(lambda text: detect_prompt_injection(text)[0] != "none")

    def manifest(self):
        return capability_manifest(session=self)

    def jobs(self):
        if self._jobs is None:
            from .agents import AgentOrchestrator
            self._jobs = JobScheduler(AgentOrchestrator(approve=None),
                                      audit=self.audit)
        return self._jobs

    def remember(self, subject, relation, obj):
        fact = self.memory.remember(subject, relation, obj, source="desktop")
        return f"সংরক্ষিত: {subject} {relation} {obj} (source={fact['source']})"

    def recall(self, query, k=3):
        hits = self.memory.recall(query, k=k)
        return [f"({h['origin']}, {h['score']}) {h['text'][:160]}" for h in hits] \
            or ["কিছু পাওয়া যায়নি"]

    def timeline(self, n=15):
        return self.memory.episodic.timeline(n=n) or ["(খালি)"]

    def export_memory(self, path):
        state = self.memory.export_json(path)
        return f"exported {len(state['facts'])} facts -> {path}"

    def import_memory(self, path):
        n = self.memory.import_json(path)
        return f"imported {n} facts from {path}"

    def injection_check(self, text):
        risk, matches = detect_prompt_injection(text)
        return f"risk={risk}, matched patterns={len(matches)}"

    def audit_tail(self, n=15):
        entries = self.audit.tail(n)
        return [f"{e['ts']} [{e['agent']}] {e['event']}: {e['detail'][:90]}"
                for e in entries] or ["(খালি)"]

    def set_setting(self, key, value):
        self.settings.set(key, value)
        if key == "quantized":
            self._router = None
        return f"{key} = {value}"

    def skill_list(self):
        return [f"{s['trigger']} -> {' > '.join(s['workflow'])}"
                for s in self.skills._load()] or ["(কোনো skill শেখা হয়নি)"]

    def strategy_for(self, goal):
        return DecisionEngine().choose(goal)

    def predict_trend(self, horizon=3):
        """Predictive Intelligence Engine: forecast the benchmark trend."""
        from .predictive import PredictiveEngine
        return PredictiveEngine().predict_outcome(self.history.entries(),
                                                  horizon=horizon)

    def simulate_future(self, decisions, steps=3):
        """Future State Simulator: rank decision branches into futures."""
        from .predictive import FutureStateSimulator
        return FutureStateSimulator().simulate({"roots": list(self.roots)},
                                               decisions, steps=steps)

    def knowledge_gaps(self):
        """Knowledge Gap Detector: what the graph does not know yet."""
        from .knowledge import KnowledgeGapDetector
        report = KnowledgeGapDetector(self.memory.graph).gaps()
        report["questions"] = KnowledgeGapDetector(self.memory.graph).questions()
        return report

    def attack_surface(self, root="."):
        """Attack Surface Intelligence: scan an authorized code tree."""
        from .cybersec import AttackSurfaceAnalyzer
        report = AttackSurfaceAnalyzer(root).scan()
        self.audit.log("attack_surface_scan", "desktop",
                       f"root={root} risk={report['risk_score']} "
                       f"({report['verdict']})")
        return report

    def optimize_architecture(self):
        """Cognitive Architecture Simulator: grid-search best capability mix."""
        from .metacog import CognitiveArchitectureSimulator
        tasks = [
            {"name": "chat", "needs": {"attention": 0.8, "memory": 0.4}},
            {"name": "act", "needs": {"tools": 0.9, "reflection": 0.3}},
            {"name": "research", "needs": {"memory": 0.6, "reflection": 0.5}},
        ]
        return CognitiveArchitectureSimulator().optimize(tasks)

    def _selftrain(self):
        if self._selftrain_loop is None:
            from .selftrain import SelfTrainingLoop
            self._selftrain_loop = SelfTrainingLoop(
                corpus="data/octo", out=os.path.dirname(self.checkpoint),
                audit=self.audit)
        return self._selftrain_loop

    def selftrain_cycle(self, steps=None):
        """Autonomous Self-Training: one improvement-gated training cycle."""
        loop = self._selftrain()
        if steps:
            loop.steps = int(steps)
        return loop.run_cycle(reason="manual")

    def selftrain_toggle(self, on):
        """Start/stop the background self-training daemon."""
        loop = self._selftrain()
        return loop.start() if on else loop.stop()

    def selftrain_status(self):
        loop = self._selftrain()
        return {"running": loop.running,
                "interval": loop.interval,
                "checkpoint": os.path.exists(loop.checkpoint),
                "cycles": len(loop.cycles),
                "last": loop.cycles[-1] if loop.cycles else None}

    def voice_listen(self):
        """STT via the shared SpeechEngine; graceful text when unavailable."""
        try:
            from speech.engine import SpeechEngine
            engine = SpeechEngine()
            if not engine.is_stt_ready():
                return "(STT প্রস্তুত নয় — vosk model ও sounddevice দরকার)"
            text = engine.listen(timeout=4)
            return text or "(কিছু শোনা যায়নি)"
        except Exception as e:
            return f"(voice unavailable: {e})"

    def voice_speak(self, text):
        try:
            from speech.engine import SpeechEngine
            SpeechEngine().speak(text)
            return "spoken"
        except Exception as e:
            return f"(TTS unavailable: {e})"


class OctoDesktopApp:
    """Tkinter desktop application — tabbed control center for Octo Core.

    Tabs: Chat · Agent · Memory · Security · Benchmarks · Lab · Train ·
    Settings · Status. Long work runs in daemon threads; results flow back
    through a queue; sensitive operations raise a human-approval dialog.
    """

    def __init__(self, root, session=None):
        import tkinter as tk
        from tkinter import ttk
        import tkinter.scrolledtext as scrolledtext

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.session = session or OctoSession()
        root.title("🐙 Octo Desktop — LLM + Agent Control Center")
        root.geometry("900x660")

        self.queue = []
        self.approvals = []
        root.after(100, self._poll)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.chat_out = self._tab_with_io("💬 Chat", "প্রশ্ন লিখুন...", self.send_chat)
        self.agent_out = self._tab_with_io("🤖 Agent", "লক্ষ্য লিখুন: research sql injection ...",
                                           self.send_goal)
        self._build_memory_tab()
        self._build_security_tab()
        self._build_bench_tab()
        self._build_lab_tab()
        self._build_train_tab()
        self._build_settings_tab()
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
        self._run_bg(self.session.chat_checked, text, on_done=self._chat_done)

    def _chat_done(self, payload):
        answer, badge = payload
        if badge:
            self._append(self.chat_out, f"octo: {answer}\n[{badge}]")
        else:
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
        from tkinter import filedialog
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🧠 Memory")
        form = ttk.Frame(frame)
        form.pack(fill=tk.X, padx=6, pady=6)
        self.mem_subject = ttk.Entry(form, width=18)
        self.mem_relation = ttk.Entry(form, width=14)
        self.mem_object = ttk.Entry(form, width=24)
        for w in (self.mem_subject, self.mem_relation, self.mem_object):
            w.pack(side=tk.LEFT, padx=2)
        ttk.Button(form, text="Remember", command=self.on_remember).pack(side=tk.LEFT, padx=4)
        query_row = ttk.Frame(frame)
        query_row.pack(fill=tk.X, padx=6)
        self.mem_query = ttk.Entry(query_row)
        self.mem_query.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(query_row, text="Recall", command=self.on_recall).pack(side=tk.LEFT, padx=4)
        ttk.Button(query_row, text="Timeline", command=self.on_timeline).pack(side=tk.LEFT)
        ttk.Button(query_row, text="Export", command=lambda: self.on_export(filedialog)).pack(side=tk.LEFT, padx=4)
        ttk.Button(query_row, text="Import", command=lambda: self.on_import(filedialog)).pack(side=tk.LEFT)
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
        ttk.Button(row, text="Audit log", command=self.on_audit).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Red-team", command=self.on_redteam).pack(side=tk.LEFT)
        self.sec_out = self._output_box(frame)

    def _build_bench_tab(self):
        ttk = self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Benchmarks")
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=6)
        ttk.Button(row, text="Run + record", command=self.on_bench).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Show trend", command=self.on_trend).pack(side=tk.LEFT)
        self.bench_out = self._output_box(frame)

    def _build_lab_tab(self):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🧪 Lab")
        scan_row = ttk.Frame(frame)
        scan_row.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(scan_row, text="scan root:").pack(side=tk.LEFT)
        self.lab_root = ttk.Entry(scan_row)
        self.lab_root.insert(0, ".")
        self.lab_root.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(scan_row, text="Attack surface",
                   command=self.on_lab_attack_surface).pack(side=tk.LEFT)
        quick_row = ttk.Frame(frame)
        quick_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(quick_row, text="Trend forecast",
                   command=self.on_lab_trend).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(quick_row, text="Knowledge gaps",
                   command=self.on_lab_gaps).pack(side=tk.LEFT, padx=4)
        ttk.Button(quick_row, text="Optimize architecture",
                   command=self.on_lab_optimize).pack(side=tk.LEFT)
        train_row = ttk.Frame(frame)
        train_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(train_row, text="self-train steps:").pack(side=tk.LEFT)
        self.lab_steps = ttk.Entry(train_row, width=8)
        self.lab_steps.insert(0, "300")
        self.lab_steps.pack(side=tk.LEFT, padx=4)
        ttk.Button(train_row, text="Self-train cycle",
                   command=self.on_lab_selftrain).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_row, text="Auto-train on/off",
                   command=self.on_lab_autotrain).pack(side=tk.LEFT)
        self._autotrain_on = False
        future_row = ttk.Frame(frame)
        future_row.pack(fill=tk.X, padx=6, pady=2)
        self.lab_future = ttk.Entry(future_row)
        self.lab_future.insert(0, "safe path, 0.5, 0.1, 0.1; bold path, 0.9, 0.6, 0.4")
        self.lab_future.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(future_row, text="Simulate futures",
                   command=self.on_lab_future).pack(side=tk.LEFT)
        self.lab_out = self._output_box(frame)

    def _build_train_tab(self):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🏋 Train (OctoV2)")
        form = ttk.Frame(frame)
        form.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(form, text="corpus:").pack(side=tk.LEFT)
        self.train_corpus = ttk.Entry(form)
        self.train_corpus.insert(0, "data/octo")
        self.train_corpus.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(form, text="steps:").pack(side=tk.LEFT)
        self.train_steps = ttk.Entry(form, width=8)
        self.train_steps.insert(0, "500")
        self.train_steps.pack(side=tk.LEFT, padx=4)
        ttk.Button(form, text="Train", command=self.on_train).pack(side=tk.LEFT)
        self.train_out = self._output_box(frame)

    def _build_settings_tab(self):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙ Settings")
        form = ttk.Frame(frame)
        form.pack(fill=tk.X, padx=6, pady=8)
        fields = (("temperature", 0.1), ("top_k", 1), ("steps", 1),
                  ("max_steps", 1))
        self.setting_vars = {}
        for i, (key, step) in enumerate(fields):
            ttk.Label(form, text=key).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = self.tk.StringVar(value=str(self.session.settings.get(key)))
            self.setting_vars[key] = var
            ttk.Spinbox(form, from_=0, to=10000, increment=step,
                        textvariable=var, width=10).grid(row=i, column=1, sticky=tk.W)
        ttk.Label(form, text="language (bn/en)").grid(row=len(fields), column=0, sticky=tk.W, pady=2)
        lang = self.tk.StringVar(value=self.session.settings.get("language"))
        self.setting_vars["language"] = lang
        ttk.Combobox(form, textvariable=lang, values=("bn", "en"), width=8,
                     state="readonly").grid(row=len(fields), column=1, sticky=tk.W)
        ttk.Label(form, text="quantized runtime").grid(row=len(fields) + 1, column=0, sticky=tk.W, pady=2)
        quant = self.tk.BooleanVar(value=bool(self.session.settings.get("quantized")))
        self.setting_vars["quantized"] = quant
        ttk.Checkbutton(form, variable=quant).grid(row=len(fields) + 1, column=1, sticky=tk.W)
        ttk.Button(form, text="Save settings", command=self.on_save_settings).grid(
            row=len(fields) + 2, column=0, columnspan=2, pady=8, sticky=tk.W)
        self.settings_out = self._output_box(frame)

    def _build_status_tab(self):
        ttk = self.ttk
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ℹ Status")
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=6)
        ttk.Button(row, text="Refresh status", command=self.on_status).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Capability manifest", command=self.on_manifest).pack(side=tk.LEFT)
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

    def on_export(self, filedialog):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            initialfile="octo_memory.json")
        if path:
            self._append(self.mem_out, self.session.export_memory(path))

    def on_import(self, filedialog):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self._append(self.mem_out, self.session.import_memory(path))

    def on_inject(self):
        text = self.sec_input.get().strip()
        if text:
            self._append(self.sec_out, self.session.injection_check(text))

    def on_audit(self):
        for line in self.session.audit_tail():
            self._append(self.sec_out, line)

    def on_redteam(self):
        self._append(self.sec_out, "red-team চলছে...")
        self._run_bg(self.session.redteam_run, on_done=self._redteam_done)

    def _redteam_done(self, report):
        self._append(self.sec_out, f"red-team: {report['blocked']}/{report['cases']} "
                     f"blocked (pass {report['pass_rate']:.0%})")

    def on_bench(self):
        self._append(self.bench_out, "চলছে...")
        self._run_bg(self.session.benchmarks_recorded, on_done=self._bench_done)

    def _bench_done(self, payload):
        report, trend = payload
        for name, res in report.items():
            self._append(self.bench_out, f"{name}: {res}")
        self._append(self.bench_out, f"trend: {trend['direction']} "
                     f"({len(trend['history'])} runs)")

    def on_trend(self):
        trend = self.session.history.trend(("math", "score"))
        for ts, value in trend["history"]:
            self._append(self.bench_out, f"{ts}  math={value}")
        self._append(self.bench_out, f"direction: {trend['direction']}")

    def on_lab_attack_surface(self):
        root = self.lab_root.get().strip() or "."
        self._append(self.lab_out, f"attack-surface scan চলছে: {root} ...")
        self._run_bg(self.session.attack_surface, root,
                     on_done=self._lab_attack_done)

    def _lab_attack_done(self, report):
        self._append(self.lab_out,
                     f"files={report['files_scanned']} risk={report['risk_score']} "
                     f"({report['verdict']}) | high={report['counts'].get('high', 0)} "
                     f"medium={report['counts'].get('medium', 0)} "
                     f"low={report['counts'].get('low', 0)}")
        for f in report["findings"][:10]:
            self._append(self.lab_out,
                         f"[{f['severity']}] {f['file']}:{f['line']} — {f['finding']}")
        if len(report["findings"]) > 10:
            self._append(self.lab_out,
                         f"... আরও {len(report['findings']) - 10} টি finding")

    def on_lab_trend(self):
        out = self.session.predict_trend()
        self._append(self.lab_out,
                     f"forecast: {out['forecast']} | slope={out['slope']} "
                     f"| direction={out['direction']} | confidence={out['confidence']}")

    def on_lab_gaps(self):
        report = self.session.knowledge_gaps()
        self._append(self.lab_out,
                     f"facts={report['total_facts']} coverage={report['coverage']} | "
                     f"sparse={len(report['sparse'])} "
                     f"conflicts={len(report['contradictions'])} "
                     f"missing={len(report['missing_relations'])}")
        for q in report["questions"]:
            self._append(self.lab_out, f"? {q}")

    def on_lab_optimize(self):
        out = self.session.optimize_architecture()
        arch = ", ".join(f"{k}={v}" for k, v in out["best_architecture"].items())
        self._append(self.lab_out, f"best architecture: {arch} (score={out['score']})")

    def on_lab_selftrain(self):
        try:
            steps = int(self.lab_steps.get())
        except ValueError:
            steps = 300
        self._append(self.lab_out, f"self-train cycle চলছে ({steps} steps)...")
        self._run_bg(self.session.selftrain_cycle, steps,
                     on_done=self._lab_selftrain_done)

    def _lab_selftrain_done(self, report):
        self._append(self.lab_out,
                     f"self-train: loss {report['val_loss_before']} -> "
                     f"{report['val_loss_after']} | {report['action']}")

    def on_lab_autotrain(self):
        self._autotrain_on = not self._autotrain_on
        started = self.session.selftrain_toggle(self._autotrain_on)
        state = "চালু" if self._autotrain_on and started else "বন্ধ"
        self._append(self.lab_out, f"auto-train {state} "
                     f"(প্রতি {self.session.selftrain_status()['interval']} সেকেন্ডে)")

    def on_lab_future(self):
        decisions = []
        for part in self.lab_future.get().split(";"):
            bits = [b.strip() for b in part.split(",")]
            if len(bits) != 4:
                continue
            try:
                decisions.append({"name": bits[0], "gain": float(bits[1]),
                                  "risk": float(bits[2]), "cost": float(bits[3])})
            except ValueError:
                continue
        if not decisions:
            self._append(self.lab_out,
                         "ফরম্যাট: name, gain, risk, cost; name2, ...")
            return
        self._run_bg(self.session.simulate_future, decisions,
                     on_done=self._lab_future_done)

    def _lab_future_done(self, result):
        for f in result["futures"]:
            self._append(self.lab_out,
                         f"{f['decision']}: EV={f['expected_value']} "
                         f"p10={f['p10']} p90={f['p90']} "
                         f"loss-risk={f['risk_of_loss']}")
        self._append(self.lab_out, f"recommended: {result['recommended']}")

    def on_train(self):
        corpus = self.train_corpus.get().strip() or "data/octo"
        try:
            steps = int(self.train_steps.get())
        except ValueError:
            steps = 500
        self._append(self.train_out, f"training শুরু: {corpus} ({steps} steps)...")
        self._run_bg(self.session.train, corpus, steps,
                     on_done=self._train_done, progress_cb=self._train_progress)

    def _train_progress(self, step, total, loss):
        self.queue.append((self._train_progress_done,
                           f"step {step}/{total} | val loss {loss:.4f}"))

    def _train_progress_done(self, line):
        self._append(self.train_out, line)

    def _train_done(self, path):
        self._append(self.train_out, f"সম্পন্ন: {path}")

    def on_save_settings(self):
        for key, var in self.setting_vars.items():
            value = var.get()
            if isinstance(value, bool):
                self.session.set_setting(key, value)
                continue
            if key in ("language",):
                self.session.set_setting(key, value)
                continue
            try:
                self.session.set_setting(key, float(value) if "." in str(value) else int(value))
            except (TypeError, ValueError):
                pass
        self._append(self.settings_out, "settings সেভ হয়েছে ✓")

    def on_status(self):
        st = self.session.status()
        lines = [f"os: {st['resources'].get('os', '?')} | device: {st['device']}",
                 f"checkpoint: {'✓' if st['checkpoint'] else '✗ (train করুন)'}",
                 f"model backend: {st['model_backend']} "
                 f"({'ready' if st['model_ready'] else 'not trained'})",
                 f"quantized runtime: {'on' if st['quantized'] else 'off'}",
                 f"knowledge facts: {st['knowledge_facts']}",
                 f"skills: {st['skills']}",
                 f"audit entries: {st['audit_entries']}",
                 f"resources: {st['resources']}"]
        self.status_out.configure(state=self.tk.NORMAL)
        self.status_out.delete("1.0", self.tk.END)
        self.status_out.configure(state=self.tk.DISABLED)
        for line in lines:
            self._append(self.status_out, line)

    def on_manifest(self):
        caps = self.session.manifest()
        self._append(self.status_out, f"capabilities ({len(caps['capabilities'])}):")
        for c in caps["capabilities"]:
            self._append(self.status_out, f"  + {c['detail']}")
        self._append(self.status_out, "limitations:")
        for l in caps["limitations"]:
            self._append(self.status_out, f"  - {l}")

    def _request_approval(self, action, description):
        """Human-in-the-loop from a background thread: queue the request and
        wait; the UI thread shows the dialog in _poll()."""
        import threading
        event = threading.Event()
        holder = {}
        self.approvals.append((action, description, holder, event))
        event.wait(timeout=120)
        return holder.get("approved", False)

    def _run_bg(self, fn, *args, on_done=None, progress_cb=None):
        import threading
        if progress_cb is not None and on_done is not None:
            def work():
                try:
                    result = fn(*args, progress=progress_cb)
                except TypeError:
                    result = fn(*args)
                except Exception as e:
                    result = f"error: {e}"
                self.queue.append((on_done, result))
        else:
            def work():
                try:
                    result = fn(*args)
                except Exception as e:
                    result = f"error: {e}"
                self.queue.append((on_done, result))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        while self.approvals:
            action, description, holder, event = self.approvals.pop(0)
            from tkinter import messagebox
            holder["approved"] = messagebox.askyesno(
                "Octo — approval needed",
                f"Sensitive action: {action}\n\n{description[:300]}\n\nAllow?")
            event.set()
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
