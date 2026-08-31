# -*- coding: utf-8 -*-
"""Personal Voice Assistant — CLI only. Optional voice: python main.py --voice"""
import sys, os, time, getpass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import config
from brain.core import PersonalBrain
from speech.engine import SpeechEngine
from storage.crypto import EncryptionManager, is_encryption_enabled

def print_banner(speech):
    print("\n=== Brain v5 CLI ===")
    print(f"Assistant : {config.ASSISTANT_NAME} v{getattr(config, 'APP_VERSION', '5.0')}")
    print(f"STT: {'OK' if speech.is_stt_ready() else 'text mode'}  TTS: {'OK' if speech.is_tts_ready() else 'off'}")
    print(f"Encrypt: {'ON' if is_encryption_enabled() else 'OFF'}")

def handle_encryption_startup(speech):
    em = EncryptionManager()
    if is_encryption_enabled():
        print("Encryption on.")
        for _ in range(3):
            try:
                pwd = getpass.getpass("Master password: ")
            except Exception:
                pwd = input("Master password: ")
            if em.unlock(pwd):
                print("Unlocked.")
                return em
        return em
    return em

def conversation_loop(brain, speech, voice_mode):
    print("Type a command. Exit: bye / Ctrl+C\n")
    while True:
        try:
            if voice_mode and speech.is_stt_ready():
                if speech.is_wakeword_ready() and getattr(config, 'USE_OPENWAKEWORD', True):
                    if not speech.wait_for_wakeword(timeout=0):
                        continue
                    speech.speak('হ্যাঁ?' if config.RESPONSE_LANG=='bn' else 'Yes?')
                user_text = speech.listen(timeout=getattr(config, 'POST_WAKE_LISTEN', config.LISTEN_TIMEOUT))
            else:
                try:
                    user_text = input('👤 আপনি: ').strip()
                except EOFError:
                    break
            if not user_text:
                continue
            low = user_text.lower().strip()
            if low in ('bye','exit','quit','বন্ধ','বন্ধ করো','বিদায়'):
                speech.speak('বিদায়!' if config.RESPONSE_LANG=='bn' else 'Goodbye!')
                break
            response = brain.process(user_text)
            if response == 'EXIT':
                speech.speak('বিদায়!' if config.RESPONSE_LANG=='bn' else 'Goodbye!')
                break
            speech.speak(response)
        except KeyboardInterrupt:
            print('\nbye')
            break
        except Exception as e:
            print('warn', e)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--voice', action='store_true')
    ap.add_argument('--no-speak', action='store_true')
    ap.add_argument('--diag', action='store_true')
    args = ap.parse_args()
    speech = SpeechEngine()
    if args.no_speak:
        speech.tts_engine = None
    print_banner(speech)
    crypto = handle_encryption_startup(speech)
    brain = PersonalBrain()
    brain.crypto = crypto
    brain.speech_ref = speech
    if args.diag:
        from features.extras import run_diagnostics
        print(run_diagnostics(brain, speech))
        return
    voice_mode = bool(args.voice and speech.is_stt_ready())
    print('CLI mode. Commands: সাহায্য | আবহাওয়া | মনে রাখো ...\n')
    speech.speak(f'হ্যালো! আমি {config.ASSISTANT_NAME}.')
    conversation_loop(brain, speech, voice_mode)
    if brain.crypto:
        brain.crypto.lock()

if __name__ == '__main__':
    main()
