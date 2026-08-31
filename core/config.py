# -*- coding: utf-8 -*-
"""Configuration for Personal Voice Assistant (v5 Complete)"""
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15")
MEMORY_FILE = os.path.join(BASE_DIR, "data", "memory.json")
SAMPLE_RATE = 16000
CHANNELS = 1
TTS_RATE = 165
TTS_VOLUME = 0.95
WAKE_WORDS = ["hey brain", "ok brain", "brain", "assistant", "হ্যালো ব্রেইন", "ব্রেইন"]
ASSISTANT_NAME = "Brain"
APP_VERSION = "5.0"
RESPONSE_LANG = "bn"
LISTEN_TIMEOUT = 12.0
STT_BLOCKSIZE = 4000
STT_SILENCE_SEC = 1.1
STT_MIN_SPEECH_SEC = 0.25
STT_ENERGY_THRESHOLD = 280
STT_SHOW_PARTIAL = True
STT_USE_COMMAND_GRAMMAR = False
STT_ALT_MODELS = [
    os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15"),
    os.path.join(BASE_DIR, "models", "vosk-model-en-us-0.22"),
    os.path.join(BASE_DIR, "models", "vosk-model-small-bn"),
]
ENABLE_LOGGING = True
USE_OPENWAKEWORD = True
OPENWAKEWORD_MODELS = ["hey_mycroft"]
OPENWAKEWORD_THRESHOLD = 0.5
POST_WAKE_LISTEN = 10.0
NEURAL_LAYER_SIZES = [128, 96, 64, 32]
NEURAL_LEARNING_RATE = 0.05
NEURAL_L2 = 5e-4
NEURAL_MOMENTUM = 0.9
NEURAL_EPOCHS_FULL = 150
NEURAL_EPOCHS_ONLINE = 40
NEURAL_MAX_EXAMPLES = 800
NEURAL_MAX_FEATURES = 400
NEURAL_CUSTOM_DATA = os.path.join(BASE_DIR, "data", "custom_training.json")
LOCAL_LLM_URL = ""
LOCAL_LLM_MODEL = "local"
