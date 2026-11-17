# Personal Voice Assistant (Brain) v5

Offline-first personal CLI assistant. No Ollama / paid LLM API required.
Voice optional: Vosk STT + pyttsx3 TTS + OpenWakeWord.

## Setup

```bash
git clone https://github.com/rifahtasminrahman05-bit/personal-voice-assistant.git
cd personal-voice-assistant
pip install -r requirements.txt
```

Download a Vosk model once from https://alphacephei.com/vosk/models  
Unzip to: `models/vosk-model-small-en-us-0.15/`

## Run (CLI only)

```bash
python main.py              # terminal chat
python main.py --voice      # microphone + STT
python main.py --diag       # system + self diagnostic
python train_model.py --stats
python train_model.py --tune

## Octo (built-in small LLM)

Offline GPT-style Transformer LLM written from scratch (PyTorch), focused on coding and
cybersecurity research (defensive / CTF style). Train it on any text file or directory:

```bash
python -m octo.train --corpus data/octo --steps 2000
python -m octo.chat --checkpoint octo_checkpoints/octo.pt --prompt "hello" --tokens 200
python -m octo.chat --checkpoint octo_checkpoints/octo.pt   # interactive chat
python -m octo.train --corpus data/octo --tokenizer bpe --vocab-size 512 --steps 5000  # BPE tokenizer
python -m octo.chat --checkpoint octo_checkpoints/octo.pt --rag --prompt "how to prevent sql injection"
```

RAG — retrieve relevant corpus passages to ground answers (also automatic in `অক্টো <প্রশ্ন>`):

```bash
python -m octo.rag --query "how to prevent sql injection"
```

Research mode — collect fresh text from Wikipedia or any URL, then learn it:

```bash
python -m octo.research --topics "cryptography,firewall,phishing" --lang en
python -m octo.research --url https://docs.python.org/3/tutorial/index.html
python -m octo.research --topic "firewall" --retrain 1000   # collect + retrain
```

Deep research — one command: Bing + Wikipedia -> save -> summary:

```bash
python -m octo.research --deep "sql injection prevention"
python -m octo.agi --goal "deep research on phishing attacks"
```

Resume training from a checkpoint (vocab changes are handled). Reuse the saved tokenizer
to keep vocab stable, and oversample focus files to weight a topic (e.g. programming):

```bash
python -m octo.train --corpus data/octo --steps 5000 --resume octo_checkpoints/octo.pt \
    --tokenizer-file octo_checkpoints/tokenizer.json --oversample "programming.txt:3,coding.txt:2"
```

AGI agent mode — goal-driven tool loop (research, RAG search, unit conversion,
safe calculator, persistent memory, Octo answers):

```bash
python -m octo.agi --goal "how many seconds in 3 days"
python -m octo.agi --goal "research sql injection and answer how to prevent it"
```

In the assistant CLI: `agi <লক্ষ্য>`. Agent notes persist in `data/octo/agi_memory.txt`.

Deep learning extras:

```bash
python -m octo.train --corpus data/octo --warmup 100 --grad-accum 2 --bf16   # warmup + accumulation + bf16
python -m octo.embeddings --query "how to prevent sql injection"             # semantic search (Octo embeddings)
python -m octo.classifier --train --data data/octo/intents.json              # intent head (transfer learning)
python -m octo.classifier --predict "remember my password"
```

Training saves `octo-best.pt` (lowest val loss) alongside `octo.pt`. The intent head also
routes unmatched assistant inputs (e.g. "remember this thing" -> note). Embedding and
classifier quality scale with how long Octo has been trained.

Pure-NumPy neural network — hand-written MLP (manual backprop + Adam), no PyTorch needed.
Fast intent detection; tried first by the assistant before the torch head:

```bash
python -m octo.nnet --train --data data/octo/intents.json
python -m octo.nnet --predict "আজ বৃষ্টি হবে কি"
```

Extractive summarizer for research documents and RAG results:

```bash
python -m octo.summarize --file data/octo/research/agi-firewall.txt --sentences 3
python -m octo.summarize --query "sql injection" --sentences 3
```

Chat memory — the interactive chat remembers the last N exchanges as context:

```bash
python -m octo.chat --checkpoint octo_checkpoints/octo.pt --memory 3   # default
python -m octo.chat --checkpoint octo_checkpoints/octo.pt --memory 0   # stateless
```

int8 quantization — ~4x smaller Linear weights for faster CPU inference:

```bash
python -m octo.quantize --checkpoint octo_checkpoints/octo-best.pt
python -m octo.chat --checkpoint octo_checkpoints/octo-int8.pt
```

Web search (Bing) — beyond Wikipedia, feeds AGI and research:

```bash
python -m octo.search --query "sql injection prevention owasp" --fetch 1
python -m octo.agi --goal "web search for sql injection prevention"
```

Octo-Pro — LoRA fine-tune of a pretrained LLM (Qwen2.5-0.5B) on your own Q&A data.
GPU recommended; needs `pip install transformers peft datasets accelerate`:

```bash
python -m octo.pro --train --data data/octo/pro_train.jsonl
python -m octo.pro --prompt "How do I prevent SQL injection?"
```

SecOps toolkit — educational, defensive/CTF-style security tools (no attack tooling):

```bash
python -m octo.secops hashid 8d969eef...        # identify hash type
python -m octo.secops passcheck "P@ssw0rd"    # password strength audit
python -m octo.secops genpass --length 20       # strong password
python -m octo.secops decode "aGVsbG8="         # base64/hex decode
python -m octo.secops caesar "Khoor Zruog"      # Caesar cracking (CTF)
python -m octo.secops logscan data/auth.log     # brute-force / scan detection
python -m octo.secops cve "log4j"               # CVE lookup via web
```

AGI can use them too: `agi check hash <hex>`, `agi passcheck "..."`, `agi logscan file.log`.
Ethical-hacking methodology corpus: `data/octo/ethhacking.txt` (pentest phases, tools
overview, hardening, ATT&CK, red/blue/purple — conceptual and defensive).

GUI chat (Tkinter):

```bash
python gui.py
```

HTTP API — chat with Octo from any app (loopback only, stdlib):

```bash
python -m octo.server                # http://127.0.0.1:8600 (web chat at /)
curl -s http://127.0.0.1:8600/health
curl -s -X POST http://127.0.0.1:8600/chat -d '{"prompt": "hello", "tokens": 100}'
```

Voice mode — STT via Vosk, TTS via pyttsx3, graceful text-mode fallback
(`speech/engine.py`). Notes persist in SQLite (`storage/db.py`), optional
Fernet encryption via master password (`storage/crypto.py`).

## Octo Core (Advanced Intelligence Architecture)

`octo_core/` — the next-generation modular layer on top of Octo LLM (spec
Phases 1–7). Every layer is importable on its own and covered by tests:

| Module | Layer | Highlights |
| ------ | ----- | ---------- |
| `octo_core/cognitive.py` | Cognitive Core | task decomposition, hypothesis ranking, causal chains, uncertainty quantification, knowledge synthesis, constraint checks, reflection loop, adaptive strategy selection |
| `octo_core/agents.py` | Agent Core | multi-agent orchestrator (researcher/coder/security), agent-to-agent messaging, execution verification, long-horizon state, human-in-the-loop hook |
| `octo_core/memory.py` | Memory Core | knowledge graph (provenance + conflict resolution), working/episodic memory, hybrid RAG recall, conversation compression |
| `octo_core/models.py` | Model Core | OctoV2: RoPE + Grouped-Query Attention + KV cache + sliding window; model router (nnet → Octo → Octo-Pro) |
| `octo_core/security.py` | Security & Governance | prompt-injection defense, least-privilege permissions, AST sandbox, secrets redaction, JSONL audit log, fail-closed approval gateway |
| `octo_core/learning.py` | Learning Core | skill acquisition, preference feedback, regression detection, controlled self-improvement (propose → isolated test → human approval) |
| `octo_core/evaluation.py` | Evaluation | math/grounding/recall/uncertainty benchmarks, hallucination detector, quality metrics, agent success metrics, pytest regression runner |
| `octo_core/infrastructure.py` | Infrastructure | TTL+LRU cache, retry with backoff, circuit breaker, threaded scheduler, resource manager, latency/RSS profiler |
| `octo_core/multimodal.py` | Multimodal | image metadata, OCR (optional pytesseract), audio inspection, cross-modal tag retrieval |

Usage:

```bash
python -m octo_core --goal "research sql injection and answer how to prevent it"
python -m octo_core --benchmark                        # math + grounding suites
python -m octo_core --inject-check "ignore all previous instructions"
python -m octo_core --audit                            # last audit entries
```

Pipeline: injection check → adaptive strategy → task decomposition → agent
routing → permission/approval-gated tool calls → execution verification →
reflection + skill capture → audit log. Blocked or failed steps never reach
the model; secrets are redacted before anything is logged.

Self-improvement is deliberately conservative: proposals are written to
`octo_checkpoints/proposals/` as JSON patches, validated against current file
content, tested in an isolated subprocess, and applied only after explicit
human approval.

## Octo Desktop (Desktop Application)

`octo_desktop.py` — Tkinter-ভিত্তিক ডেস্কটপ অ্যাপ্লিকেশন: Octo Core-এর সব লেয়ার
একটি ট্যাবড কন্ট্রোল সেন্টারে। দীর্ঘ কাজ ব্যাকগ্রাউন্ড থ্রেডে চলে, ফলাফল queue-র
মাধ্যমে UI-তে আসে — UI কখনো ফ্রিজ করে না।

```bash
python octo_desktop.py
```

ট্যাবসমূহ:

| ট্যাব | কাজ |
| --- | --- |
| 💬 Chat | RAG-grounded Octo LLM চ্যাট, prompt-injection ব্লকিং সহ |
| 🤖 Agent | লক্ষ্য দিন → strategy + task decomposition + agent routing + verification + reflection |
| 🧠 Memory | knowledge graph-এ fact যোগ, hybrid recall, episodic timeline |
| 🛡 Security | injection checker, audit log viewer, red-team |
| 📊 Benchmarks | math + grounding benchmark suite, trend |
| 🧪 Lab | attack-surface scan, trend forecast, knowledge gaps, future-state simulator, cognitive-architecture optimizer |
| ⚙ Status | device, checkpoint, model backend, facts/skills/audit সারাংশ |

সব লজিক `OctoSession` ক্লাসে (display-independent) — Tk ছাড়াই টেস্টেবল।
GUI নির্ভরতা: Python-এর built-in Tkinter (Linux-এ `sudo apt install python3-tk`)।

### macOS ও Windows-এ চালানো

**macOS:**

```bash
# 1. Python 3 (python.org installer — Tkinter সহ) অথবা:
brew install python-tk
# 2. নির্ভরতা
pip3 install torch numpy
# 3. চালান (দুইভাবে)
python3 octo_desktop.py          # টার্মিনাল থেকে
open octo_desktop.command        # অথবা Finder-এ ডাবল-ক্লিক
```

Apple Silicon-এ Octo Core স্বয়ংক্রিয়ভাবে **MPS** (GPU acceleration) ব্যবহার করে
(দেখুন ⚙ Status ট্যাব)। প্রথমবার `octo_desktop.command` খুললে Gatekeeper আটকালে
right-click → Open চাপুন।

**Windows:**

```powershell
# 1. Python 3 ইনস্টল করুন python.org থেকে ("tcl/tk and IDLE" চেকবক্স সহ)
# 2. নির্ভরতা
pip install torch numpy
# 3. চালান (দুইভাবে)
python octo_desktop.py           # টার্মিনাল থেকে
octo_desktop.bat                 # অথবা Explorer-এ ডাবল-ক্লিক
```

উভয় প্ল্যাটফর্মে checkpoint ও data পাথ রিপো রুটে anchor করা — যেখান থেকেই
চালান না কেন, সব ফাইল সঠিক জায়গায় থাকবে।

## Phase 8–12: Advanced Capabilities

### OctoV2 training pipeline (Phase 9)

```bash
python -m octo_core.train_v2 --corpus data/octo --steps 2000
```

RoPE + GQA + KV-cache মডেল প্রশিক্ষণ, progress callback সহ (desktop Train ট্যাব
এটাই ব্যবহার করে)। Speculative decoding (`octo_core.models.speculative_generate`),
int8 quantized runtime (Settings → quantized on, `octo.quantize` দিয়ে তৈরি
`octo-int8.pt` লোড হয়), এবং Bengali-aware BPE pre-tokenization যোগ হয়েছে।

### Agent upgrades (Phase 10)

- **Sandboxed code execution** — `octo_core/sandbox_code.py`: subprocess + timeout +
  POSIX memory/CPU limits; coding agent-এর `code` টুল এটাই ব্যবহার করে
- **Parallel multi-agent** — `orchestrator.execute_parallel(goal)`: স্বাধীন subtask
  একসাথে চলে
- **Research cross-check** — `octo_core.researchx.cross_check(query)`: multi-source
  agreement + citation list
- **Scheduled jobs** — `octo_core/jobs.py`: cron-style recurring goals, `jobs.json`-এ persist

### Trust & scale (Phase 11)

- **Benchmark dashboard** — `octo_core/benchhistory.py`: প্রতি run record + trend
  (improving/declining); desktop Benchmarks ট্যাবে "Run + record" ও "Show trend"
- **Hallucination guard** — desktop chat-এ প্রতিটি উত্তরে grounding badge
- **Cross-platform CI** — GitHub Actions ubuntu/windows/macos matrix + benchmark smoke job
- **Memory export/import** — Memory ট্যাব থেকে JSON backup/restore
- **Encrypted episodic memory** — `EncryptedEpisodicMemory` (Fernet থাকলে)

### Controlled self-evolution (Phase 12)

`octo_core/evolution.py`:

- **ContinualLearner** — নতুন ডেটা → isolated candidate checkpoint → regression
  check → human approval → promote (production কখনো auto-touch হয় না)
- **distill()** — teacher→student knowledge distillation (KL + CE)
- **RedTeamSuite** — adversarial prompt bank vs injection guard (CI-তেও চলে)
- **capability_manifest()** — structured self-report of capabilities + limitations

### Desktop packaging (Phase 8)

```bash
pip install pyinstaller
python build_desktop.py     # dist/OctoDesktop.app (macOS) / dist/OctoDesktop (Windows/Linux)
```

Desktop-এ নতুন ট্যাব: 🏋 Train (OctoV2), ⚙ Settings (temperature/top_k/language/quantized —
`settings.json`-এ persist), benchmark trend, red-team button, memory export/import,
এবং sensitive কাজে human-approval dialog।

## Next-Generation Intelligence (15 features)

`octo_core/`-এ ৮টি নতুন মডিউল — ৮ ক্যাটাগরিতে ১৫টি উন্নত ফিচার:

| # | Feature | Module / API |
|---|---------|--------------|
| 1 | Predictive Intelligence Engine | `octo_core.predictive.PredictiveEngine.forecast()` — লিনিয়ার-রিগ্রেশন ট্রেন্ড + residual confidence |
| 2 | Future State Simulator | `FutureStateSimulator.simulate()` — মন্টে-কার্লো সিদ্ধান্ত শাখা, ranked futures, p10/p90, risk_of_loss |
| 3 | Scientific Hypothesis Engine | `octo_core.scientific.ScientificHypothesisEngine` — correlation থেকে অনুমান + held-out যাচাই |
| 4 | Autonomous Discovery Laboratory | `DiscoveryLab.run_cycle()` — নিয়ন্ত্রিত পরীক্ষা লুপ (sandboxed runner সহ) + logbook |
| 5 | Digital Twin Engine | `octo_core.digital.DigitalTwin` — sync / drift / project (dynamics function) |
| 6 | Embodied Intelligence Interface | `EmbodiedInterface` + `LoopbackRobot` — register/connect/act_and_observe/sense_all |
| 7 | Knowledge Gap Detector | `octo_core.knowledge.KnowledgeGapDetector` — sparse/contradiction/missing relations + research questions |
| 8 | Ontology Learning Engine | `OntologyLearner` — is-a / part-of / used-for extraction, hierarchy, summary |
| 9 | Codebase Understanding Engine | `octo_core.software.CodebaseUnderstanding` — AST scan, dependency graph, entry candidates |
| 10 | Automated Test Generation | `TestGenerator` — public functions + edge cases (None/empty/zero/negative/large) → `tests/generated/` |
| 11 | Attack Surface Intelligence | `octo_core.cybersec.AttackSurfaceAnalyzer` — dangerous-call scan, risk score, verdict (defensive) |
| 12 | Incident Timeline Reconstruction | `IncidentTimeline.reconstruct()` — log → recon→compromise→escalation→exfiltration→cleanup |
| 13 | Test-Time Compute Scaling | `octo_core.modelint.TestTimeCompute` — difficulty routing + self-consistency voting |
| 14 | Ensemble Model Intelligence | `Ensemble` — multi-model consensus, agreement, weighted pick |
| 15 | Cognitive Architecture Simulator | `octo_core.metacog.CognitiveArchitectureSimulator` — simulate/compare/optimize capability vectors |

## Advanced Intelligence Expansion (60 features)

`octo_core/`-এ ১০টি নতুন মডিউল — ১০ ক্যাটাগরিতে ৬০টি ফিচার (প্রতিটি আলাদা ক্লাস,
stdlib heuristic, সম্পূর্ণ টেস্টযোগ্য):

| ক্যাটাগরি | মডিউল | ফিচার |
|---|---|---|
| 1. Advanced Cognitive Intelligence | `octo_core/cogx.py` | (1) Cognitive Load Manager, (2) Adaptive Reasoning Engine, (3) Causal Reasoning Engine, (4) Counterfactual Reasoning, (5) Metacognitive Controller, (6) Uncertainty Awareness Engine |
| 2. Advanced Autonomous Agent | `octo_core/swarm.py` | (7) Multi-Agent Collaboration Network, (8) Agent Negotiation Engine, (9) Dynamic Agent Creation, (10) Agent Reputation System, (11) Autonomous Workflow Composer, (12) Long-Term Goal Manager |
| 3. Advanced Memory & Knowledge | `octo_core/memx.py` | (13) Associative Memory Network, (14) Memory Consolidation Engine, (15) Memory Decay Manager, (16) Knowledge Evolution Tracker, (17) Semantic Knowledge Compression, (18) Cross-Domain Knowledge Transfer |
| 4. Self-Learning & Model Evolution | `octo_core/evolvex.py` | (19) Continual Learning Engine, (20) Experience Replay System, (21) Adaptive Curriculum Generator, (22) Skill Acquisition Engine, (23) Model Adaptation Controller, (24) Self-Improvement Evaluation Loop (approval-gated) |
| 5. Advanced Multimodal Intelligence | `octo_core/perception.py` | (25) Real-Time Scene Understanding, (26) Visual Object Tracking, (27) Spatial Reasoning Engine, (28) Gesture Recognition System, (29) Audio Scene Understanding, (30) Multimodal Context Fusion |
| 6. Advanced Software Engineering | `octo_core/devx.py` | (31) Autonomous Debugging Engine, (32) Architecture Design Intelligence, (33) Code Optimization Engine, (34) Dependency Risk Analyzer, (35) Automated Documentation Engine, (36) Software Migration Intelligence |
| 7. Advanced Cybersecurity Intelligence | `octo_core/secopsx.py` | (37) Threat Intelligence Correlation, (38) Security Posture Analyzer, (39) Configuration Risk Detector, (40) Vulnerability Prioritization Engine, (41) Zero-Trust Policy Intelligence, (42) Incident Response Planner, (43) Cloud Security Intelligence, (44) Supply Chain Security Analyzer |
| 8. Advanced System Intelligence | `octo_core/sysint.py` | (45) Dynamic Resource Allocation, (46) Intelligent Task Scheduler, (47) Predictive Failure Detection, (48) Self-Healing Infrastructure (approval-gated), (49) Distributed Agent Execution, (50) Intelligent Cache Manager |
| 9. Advanced Human-AI Interaction | `octo_core/humanai.py` | (51) Personalized Interaction Engine, (52) Intent Prediction System, (53) Context-Aware Dialogue Manager, (54) Explainable Decision Engine, (55) Collaborative Problem-Solving Engine, (56) Adaptive Teaching Engine |
| 10. Experimental Frontier Intelligence | `octo_core/frontier.py` | (57) Cognitive Architecture Evolution Engine, (58) Artificial Society Simulator, (59) Emergent Behavior Analyzer, (60) Autonomous Research Planner |

`capability_manifest()` এখন সবগুলো নতুন মডিউলও রিপোর্ট করে (Status ট্যাব)।

### নিজে-নিজে ট্রেন (Autonomous Self-Training)

`octo_core/selftrain.py` — OctoV2 নিজে নিজে ট্রেন হয়, নিরাপদভাবে:

- প্রতিটি cycle: বর্তমান checkpoint-এর validation loss মাপা → কর্পাসে ট্রেন চালিয়ে যাওয়া
  → আবার loss মাপা → **উন্নতি হলেই শুধু নতুন checkpoint রাখা**, নাহলে আগেরটায় rollback
- প্রতিটি cycle audit log-এ যায়; production কোড কখনো স্পর্ষ হয় না (শুধু model weights)
- ব্যাকগ্রাউন্ড daemon: `loop.start()` — প্রতি `interval` সেকেন্ডে এক cycle; `stop()` দিয়ে থামে

```bash
# এক cycle এখনই:
python -c "from octo_core.selftrain import SelfTrainingLoop; \
  SelfTrainingLoop(steps=500).run_cycle('manual')"
```

Desktop 🧪 Lab ট্যাবে: **Self-train cycle** বাটন (এক cycle) ও **Auto-train on/off**
(ব্যাকগ্রাউন্ড daemon); `OctoSession.selftrain_cycle() / selftrain_toggle() /
selftrain_status()`.

Desktop `OctoSession`-এ convenience API: `predict_trend()`, `simulate_future()`,
`knowledge_gaps()`, `attack_surface()`, `optimize_architecture()`; `capability_manifest()` এখন ১৫টি নতুন
ফিচারও রিপোর্ট করে (Status ট্যাব)।

## Components Deep Dive (Cognitive Agent Architecture)

Octo-র সম্পূর্ণ cognitive agent architecture — ১৩টি component area, সব লোকাল ও offline:

| Component | Module | যা আছে |
| --- | --- | --- |
| Reasoning Engine | `octo_core/reasoning.py` | Chain-of-Thought, Tree-of-Thoughts (beam + pruning), Reflection loop, Confidence calibration (meta-cognition), Analogical reasoning (case mapping), Abductive reasoning (best-explanation ranking) |
| Memory System | `memory.py` + `memx.py` | STM (7±2 WorkingMemory), LTM (SemanticMemory), Episodic, Procedural (SkillManager), Consolidation, ranked retrieval, Controlled forgetting (decay) |
| Planning & Goals | `octo_core/planner.py` | GoalManager (priority/deadline/metrics + decomposition), ActionPlanner (sequential/parallel + critical path), ExecutionMonitor (dynamic replanning), Multi-criteria decisions, Contingency fallbacks, Long-horizon scheduling |
| World Model | `octo_core/worldmodel.py` | Entities, typed/weighted relations, properties/states, temporal model + projection, causal graph (transitive inference), spatial awareness, query/pattern-match/deduction engine |
| Perception & Understanding | `humanai.py` + `sysint.py` | Intent recognition, Sentiment/emotion detection (Bangla সহ), anomaly/failure detection, context awareness |
| Action & Tool System | `octo_core/actions.py` | ToolRegistry (discovery), ActionEngine (least-privilege + safety checks + approval gateway), Guarded desktop automation (default off), External local tools, Feedback loop, Action history + audit |
| Learning & Adaptation | `evolvex.py` + `learning.py` | Experience replay, Preference learning, Skill acquisition, Self-evaluation (approval-gated), Curriculum, Online fine-tuning (selftrain) |
| Personality & Identity | `octo_core/identity.py` | Core identity, Emotional model, Relationship memory, Consistency engine, Persona modulation |
| Communication & Interface | `octo_core/identity.py` | Personality-consistent NLG, Conversation management, Clarification strategies, Explanation generation, Multi-turn coherence |
| Ethics & Value Alignment | `octo_core/identity.py` | Value system, Harm assessment, Transparency, Boundary enforcement, User value learning |
| Resource & Performance | `sysint.py` | Compute awareness, Priority scheduling, Idle optimization, Graceful degradation (full → reduced → core-only) |
| Privacy & Security | `security.py` + `actions.py` | 100% offline, local-only storage, no telemetry, safe-mode guardrails, self-healing, input sanitization, sandboxed execution, least privilege, action confirmation, audit logging |
| Data Persistence | `octo_core/agentstate.py` | `agi_state.json` (conversation + goals + runtime), `agent.save()` auto-persist, encrypted local backups, versioned snapshots + rollback |

## 🧠 Cognitive Core — মস্তিষ্ক ও সিদ্ধান্ত গ্রহণ

`octo_core/cognitive_core.py` — চারটি embodied component-কে একটি perceive → decide → act লুপে যুক্ত করে:

| # | Component | ক্লাস | বিবরণ |
| --- | --- | --- | --- |
| 1 | Voice Interface | `VoiceInterface` | কথা বলা ও শোনা — `speech/engine.py` ব্যবহার করে; হার্ডওয়্যার না থাকলে text-only মোডে নামে |
| 2 | Vision System | `VisionSystem` | ক্যামেরা ও স্ক্রিন বোঝা — image/OCR/screen analysis, ক্যামেরা ক্যাপচার (opencv থাকলে) |
| 3 | Automation Engine | `AutomationEngine` | কম্পিউটার, স্মার্ট হোম ও ডিভাইস নিয়ন্ত্রণ — প্রতিটি কমান্ড permission-checked + lock/power/delete-এ approval বাধ্যতামূলক |
| 4 | Security Core | `SecurityCore` | অনুমতি, নিরাপত্তা ও কার্যক্রম পর্যবেক্ষণ — grant/revoke, activity monitor, anomaly সনাক্তকরণ |

`CognitiveCore.decide()` ইনপুট রাউট করে (automation / vision / voice / reason), `cycle()` সম্পূর্ণ রাউন্ড চালায় ও audit-logged। Desktop 🧪 Lab ট্যাবে **Cognitive Core** বাটন; `OctoSession.core_status() / core_cycle()`।

Desktop 🧪 Lab ট্যাবে নতুন বাটন: **Reason (CoT)**, **Ethics check**, **Sentiment**,
**Snapshot state**, **Rollback state**; `OctoSession.reason() / ethics_check() /
state_snapshot() / state_rollback()`.

Tests (331 tests across `tests/`: cognitive, security,
memory, OctoV2 model + speculative, agents + parallel, learning, evaluation,
infrastructure, multimodal, main pipeline, desktop session/app,
cross-platform, train_v2, sandbox code, jobs, bench history, settings,
memory export, evolution/red-team/distill/continual):

CI runs the test suite on every push/PR (`.github/workflows/ci.yml`).

Bangla text is supported. Research data lands in `data/octo/research/` and checkpoints in
`octo_checkpoints/` (both gitignored). For ethics: Octo is a learning and defensive-research
tool, not an attack tool — corpus content stays conceptual (CTF, OWASP, forensics).
```

Wake word (if enabled): `hey_mycroft` model / say **hey brain** in software list.

## Commands

- Memory: `মনে রাখো ...` | `সাহায্য` | `সেলফ রিপোর্ট`
- Work: `নোট` | `টাস্ক` | `রিমাইন্ডার` | `জার্নাল`
- Web: `আবহাওয়া` | `নিউজ` | `উইকি` | `গবেষণা করো`
- Voice control: `ভয়েস কন্ট্রোল` | `ক্রোম খোল` | `স্ক্রিনশট`
- Agent: `এলএলএম এজেন্ট ...` | `এজেন্ট গবেষণা ...`
- Octo LLM: `অক্টো <প্রশ্ন>` — built-in offline LLM উত্তর (train করা থাকলে)
- Learn: `শেখাও টেক্সট | intent` | `ভুল, এটা জোক ছিল`
- Exit: `bye` / `বন্ধ`

Personal runtime data (`data/*.db`, logs, memory) is gitignored.
