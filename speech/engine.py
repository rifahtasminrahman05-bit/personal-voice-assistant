# -*- coding: utf-8 -*-
"""Speech engine — STT (Vosk) + TTS (pyttsx3), graceful text-mode fallback.

All components are optional: without audio libraries or models the engine
runs in text mode and the assistant still works.
"""
import os
import queue
import threading

from core import config


class SpeechEngine:
    def __init__(self):
        self._tts = None
        self._tts_error = None
        self._recognizer = None
        self._stt_error = None
        self._stream = None
        self._init_tts()
        self._init_stt()

    def _init_tts(self):
        try:
            import pyttsx3
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate", config.TTS_RATE)
            self._tts.setProperty("volume", config.TTS_VOLUME)
        except Exception as e:
            self._tts = None
            self._tts_error = str(e)

    def _init_stt(self):
        if not os.path.exists(config.MODEL_PATH):
            self._stt_error = f"vosk model not found: {config.MODEL_PATH}"
            return
        try:
            from vosk import KaldiRecognizer, SetLogLevel
            import sounddevice as sd
            SetLogLevel(-1)
            self._recognizer = KaldiRecognizer(16000, 0)
            self._recognizer.SetWords(False)
            self._sd = sd
            self._stt_error = None
        except Exception as e:
            self._recognizer = None
            self._stt_error = str(e)

    def is_stt_ready(self):
        return self._recognizer is not None

    def is_tts_ready(self):
        return self._tts is not None

    def is_wakeword_ready(self):
        return False

    def wait_for_wakeword(self, timeout=0):
        return False

    def listen(self, timeout=None):
        """Record from the microphone and return recognized text ('' on silence)."""
        if self._recognizer is None:
            return ""
        timeout = timeout or config.LISTEN_TIMEOUT
        import numpy as np
        result = []

        def callback(indata, frames, time_info, status):
            result.append(bytes(indata))

        self._recognizer.Reset()
        with self._sd.RawInputStream(samplerate=16000, blocksize=config.STT_BLOCKSIZE,
                                     dtype="int16", channels=1, callback=callback):
            total = 0.0
            while total < timeout:
                self._sd.sleep(100)
                total += 0.1
                if result and self._recognizer.AcceptWaveform(b"".join(result)):
                    break
        if not result:
            return ""
        text = self._recognizer.FinalResult().get("text", "")
        return text.strip()

    def speak(self, text):
        if self._tts is not None:
            try:
                self._tts.say(str(text))
                self._tts.runAndWait()
                return
            except Exception:
                self._tts = None
        print(f"[assistant] {text}")
