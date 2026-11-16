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
| 🛡 Security | injection checker, audit log viewer |
| 📊 Benchmarks | math + grounding benchmark suite |
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

Tests (144 unit tests in `tests/test_octo_core.py`: cognitive, security,
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
