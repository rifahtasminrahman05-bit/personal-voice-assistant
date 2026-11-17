# -*- coding: utf-8 -*-
import hashlib

import pytest

from octo.secops import (analyze_log, caesar_crack, decode_text, gen_password,
                         identify_hash, password_strength)


class TestHashId:
    def test_sha256(self):
        digest = hashlib.sha256(b"octo").hexdigest()
        assert "sha256" in identify_hash(digest)

    def test_md5(self):
        digest = hashlib.md5(b"octo").hexdigest()
        assert "md5" in identify_hash(digest)

    def test_unknown(self):
        assert identify_hash("not-a-hash") == ["unknown format"]


class TestPasswordStrength:
    def test_weak(self):
        score, label, _, _ = password_strength("123456")
        assert score <= 1 and "পরিচিত" in label or score == 0

    def test_strong(self):
        score, label, _, entropy = password_strength("xK9$mQ2#vL8pN4@w")
        assert score >= 3 and entropy > 70

    def test_feedback_given(self):
        _, _, feedback, _ = password_strength("abc")
        assert feedback


class TestGenPassword:
    def test_length_and_classes(self):
        pw = gen_password(20)
        assert len(pw) == 20
        assert any(c.isupper() for c in pw) and any(c.isdigit() for c in pw)

    def test_random(self):
        assert gen_password(20) != gen_password(20)


class TestDecode:
    def test_base64(self):
        kind, text = decode_text("aGVsbG8gd29ybGQ=")[0]
        assert kind == "base64" and text == "hello world"

    def test_hex(self):
        kind, text = decode_text("48656c6c6f")[0]
        assert kind == "hex" and text == "Hello"

    def test_plain(self):
        assert decode_text("just plain text")[0][0] == "none"


class TestCaesar:
    def test_crack(self):
        shifts = caesar_crack("Khoor Zruog")
        assert any("hello world" in text.lower() for _, text in shifts)


class TestLogScan:
    def test_detects_brute_force(self, tmp_path):
        log = tmp_path / "auth.log"
        lines = []
        for _ in range(15):
            lines.append("Nov 16 10:00:00 host sshd[1]: Failed password for root from 10.0.0.9 port 22 ssh2")
        lines.append("Nov 16 10:01:00 host sshd[1]: Accepted password for admin from 10.0.0.2 port 22 ssh2")
        log.write_text("\n".join(lines), encoding="utf-8")
        report = analyze_log(str(log))
        assert "10.0.0.9" in report and "brute-force" in report

    def test_clean_log(self, tmp_path):
        log = tmp_path / "clean.log"
        log.write_text("Nov 16 10:00:00 host systemd: started\n", encoding="utf-8")
        assert "no failed logins" in analyze_log(str(log))
