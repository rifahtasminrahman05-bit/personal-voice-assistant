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

Resume training from a checkpoint (vocab changes are handled):

```bash
python -m octo.train --corpus data/octo --steps 5000 --resume octo_checkpoints/octo.pt
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
