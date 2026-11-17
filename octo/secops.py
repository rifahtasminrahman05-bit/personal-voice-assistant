# -*- coding: utf-8 -*-
"""Octo SecOps — educational security toolkit (defensive / CTF style).

Tools: hash identification, password strength audit, strong password
generation, base64/hex decoding, Caesar cipher cracking, log analysis,
CVE lookup via web search. No active attack tooling.

Usage:
    python -m octo.secops hashid 8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92
    python -m octo.secops passcheck "P@ssw0rd123"
    python -m octo.secops genpass --length 20
    python -m octo.secops decode "aGVsbG8gd29ybGQ="
    python -m octo.secops caesar "Khoor Zruog"
    python -m octo.secops logscan data/auth.log
    python -m octo.secops cve "log4j"
"""
import argparse
import base64
import binascii
import math
import os
import re
import secrets
import string
from collections import Counter

HASH_PATTERNS = [
    ("md5", r"^[a-f0-9]{32}$"),
    ("sha1", r"^[a-f0-9]{40}$"),
    ("sha224", r"^[a-f0-9]{56}$"),
    ("sha256", r"^[a-f0-9]{64}$"),
    ("sha384", r"^[a-f0-9]{96}$"),
    ("sha512", r"^[a-f0-9]{128}$"),
    ("bcrypt", r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$"),
    ("argon2", r"^\$argon2(id|i|d)\$"),
    ("md5-crypt", r"^\$1\$[./A-Za-z0-9]{8}\$"),
    ("sha512-crypt", r"^\$6\$[./A-Za-z0-9]{8,16}\$"),
    ("ntlm", r"^[A-Fa-f0-9]{32}$"),
    ("crc32", r"^[a-f0-9]{8}$"),
]

COMMON_PASSWORDS = {"password", "123456", "12345678", "qwerty", "abc123", "password1",
                    "iloveyou", "admin", "welcome", "monkey", "letmein", "dragon"}


def identify_hash(value):
    value = value.strip()
    hits = [name for name, pattern in HASH_PATTERNS if re.match(pattern, value)]
    return hits or ["unknown format"]


def password_strength(password):
    """Return (score 0-4, feedback list). Defensive audit, offline."""
    feedback = []
    score = 0
    n = len(password)
    if n >= 12:
        score += 2
    elif n >= 8:
        score += 1
    else:
        feedback.append("খুব ছোট — কমপক্ষে ১২ অক্ষর")
    classes = sum(bool(re.search(p, password)) for p in
                  (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    score += max(0, classes - 1)
    if classes < 3:
        feedback.append("ছোট হাতের + বড় হাতের + সংখ্যা + চিহ্ন — সব মেশান")
    if password.lower() in COMMON_PASSWORDS:
        score = 0
        feedback.append("এটি সবচেয়ে পরিচিত পাসওয়ার্ড তালিকায় আছে")
    charset = 26 * (bool(re.search(r"[a-z]", password)) + bool(re.search(r"[A-Z]", password))) \
        + 10 * bool(re.search(r"\d", password)) + 33 * bool(re.search(r"[^A-Za-z0-9]", password))
    entropy = n * math.log2(charset or 1)
    if entropy < 40:
        feedback.append(f"entropy মাত্র {entropy:.0f} bits — brute-force সহজ")
    elif entropy >= 70:
        score = min(4, score + 1)
    labels = ["খুব দুর্বল", "দুর্বল", "মাঝারি", "শক্ত", "খুব শক্ত"]
    return score, labels[score], feedback, entropy


def gen_password(length=20):
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in pw) and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw) and any(c in "!@#$%^&*-_=+" for c in pw)):
            return pw


def decode_text(value):
    value = value.strip()
    attempts = []
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
        if decoded.isprintable():
            attempts.append(("base64", decoded))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        pass
    hexish = value.replace(" ", "").replace(":", "")
    if re.fullmatch(r"(0x)?[0-9a-fA-F]+", hexish) and len(hexish) % 2 == 0:
        try:
            decoded = bytes.fromhex(hexish.replace("0x", "")).decode("utf-8")
            if decoded.isprintable():
                attempts.append(("hex", decoded))
        except (ValueError, UnicodeDecodeError):
            pass
    return attempts or [("none", "no base64/hex encoding detected")]


def caesar_shifts(text):
    out = []
    for shift in range(1, 26):
        moved = []
        for ch in text:
            if "a" <= ch <= "z":
                moved.append(chr((ord(ch) - 97 + shift) % 26 + 97))
            elif "A" <= ch <= "Z":
                moved.append(chr((ord(ch) - 65 + shift) % 26 + 65))
            else:
                moved.append(ch)
        out.append((shift, "".join(moved)))
    return out


ENGLISH_WORDS = {"hello", "world", "the", "and", "is", "are", "you", "this", "that",
                 "with", "for", "have", "not", "was", "are", "key", "secret", "attack",
                 "password", "test", "message", "flag", "code", "text"}


def caesar_crack(text, top=3):
    """Rank all Caesar shifts: English-word matches first, then letter frequency."""
    freq = {"e": 12.7, "t": 9.1, "a": 8.2, "o": 7.5, "i": 7.0, "n": 6.7, "s": 6.3,
            "h": 6.1, "r": 6.0, "d": 4.3, "l": 4.0, "u": 2.8}
    scored = []
    for shift, candidate in caesar_shifts(text):
        letters = [c for c in candidate.lower() if c.isalpha()]
        if not letters:
            continue
        words = set(re.findall(r"[a-z]+", candidate.lower()))
        word_hits = len(words & ENGLISH_WORDS)
        counts = Counter(letters)
        total = len(letters)
        freq_score = -sum(abs(freq.get(c, 1.0) / 100 - counts[c] / total) for c in freq)
        scored.append((shift, (word_hits, freq_score), candidate))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(shift, text_) for shift, _, text_ in scored[:top]]


def analyze_log(path):
    """Defensive log analysis: failed logins, top source IPs, scan-like 404 bursts."""
    if not os.path.exists(path):
        raise SystemExit(f"log file not found: {path}")
    failed = Counter()
    not_found = Counter()
    total = 0
    ip_pattern = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
    for line in open(path, encoding="utf-8", errors="replace"):
        total += 1
        ips = ip_pattern.findall(line)
        low = line.lower()
        if "failed password" in low or "authentication failure" in low or " 401 " in line:
            for ip in ips:
                failed[ip] += 1
        if " 404 " in line:
            for ip in ips:
                not_found[ip] += 1
    report = [f"lines: {total}"]
    if failed:
        report.append("failed logins:")
        report += [f"  {ip}: {count}x" for ip, count in failed.most_common(5)]
        hot = [ip for ip, c in failed.items() if c >= 10]
        if hot:
            report.append(f"  ⚠ brute-force pattern from: {', '.join(hot)}")
    if not_found:
        report.append("404s (possible scanning):")
        report += [f"  {ip}: {count}x" for ip, count in not_found.most_common(3)]
    if not failed and not not_found:
        report.append("no failed logins or 404 bursts detected")
    return "\n".join(report)


def cve_lookup(query):
    from .search import search_and_read
    body, sources = search_and_read(f"CVE {query} vulnerability", fetch=1)
    return body[:1200] or "no results", sources


def main():
    ap = argparse.ArgumentParser(description="Octo SecOps — educational security toolkit.")
    sub = ap.add_subparsers(dest="tool", required=True)
    p1 = sub.add_parser("hashid"); p1.add_argument("value")
    p2 = sub.add_parser("passcheck"); p2.add_argument("value")
    p3 = sub.add_parser("genpass"); p3.add_argument("--length", type=int, default=20)
    p4 = sub.add_parser("decode"); p4.add_argument("value")
    p5 = sub.add_parser("caesar"); p5.add_argument("value"); p5.add_argument("--all", action="store_true")
    p6 = sub.add_parser("logscan"); p6.add_argument("path")
    p7 = sub.add_parser("cve"); p7.add_argument("query")
    args = ap.parse_args()

    if args.tool == "hashid":
        print("possible types:", ", ".join(identify_hash(args.value)))
    elif args.tool == "passcheck":
        score, label, feedback, entropy = password_strength(args.value)
        print(f"strength: {score}/4 ({label}) | entropy: {entropy:.0f} bits")
        for msg in feedback:
            print(" -", msg)
    elif args.tool == "genpass":
        print(gen_password(args.length))
    elif args.tool == "decode":
        for kind, text in decode_text(args.value):
            print(f"{kind}: {text}")
    elif args.tool == "caesar":
        if args.all:
            for shift, text in caesar_shifts(args.value):
                print(f"{shift:2d}: {text}")
        else:
            for shift, text in caesar_crack(args.value):
                print(f"shift {shift:2d}: {text}")
    elif args.tool == "logscan":
        print(analyze_log(args.path))
    elif args.tool == "cve":
        body, sources = cve_lookup(args.query)
        print(body)
        for src in sources:
            print("source:", src)


if __name__ == "__main__":
    main()
