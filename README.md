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

Offline GPT-style Transformer LLM written from scratch (PyTorch). Train it on any text file:

```bash
python -m octo.train --corpus data/octo/corpus.txt --steps 2000
python -m octo.chat --checkpoint octo_checkpoints/octo.pt --prompt "hello" --tokens 200
python -m octo.chat --checkpoint octo_checkpoints/octo.pt   # interactive chat
```

Bangla text is supported. Checkpoints go to `octo_checkpoints/` (gitignored).
```

Wake word (if enabled): `hey_mycroft` model / say **hey brain** in software list.

## Commands

- Memory: `মনে রাখো ...` | `সাহায্য` | `সেলফ রিপোর্ট`
- Work: `নোট` | `টাস্ক` | `রিমাইন্ডার` | `জার্নাল`
- Web: `আবহাওয়া` | `নিউজ` | `উইকি` | `গবেষণা করো`
- Voice control: `ভয়েস কন্ট্রোল` | `ক্রোম খোল` | `স্ক্রিনশট`
- Agent: `এলএলএম এজেন্ট ...` | `এজেন্ট গবেষণা ...`
- Learn: `শেখাও টেক্সট | intent` | `ভুল, এটা জোক ছিল`
- Exit: `bye` / `বন্ধ`

Personal runtime data (`data/*.db`, logs, memory) is gitignored.
