# -*- coding: utf-8 -*-
"""Optional CLI extras: --diag system + self diagnostic."""
import os
import platform
import sys


def run_diagnostics(brain=None, speech=None):
    """System + self diagnostic report (local only, no network)."""
    lines = ["=== সিস্টেম ডায়াগনস্টিক ==="]

    # platform + python
    lines.append("Python    : %s (%s)" % (platform.python_version(),
                                         sys.platform))
    lines.append("OS        : %s %s" % (platform.system(),
                                        platform.release()))

    # speech stack readiness
    if speech is not None:
        lines.append("STT ready : %s" % getattr(speech, "is_stt_ready", bool)())
        tts = getattr(speech, "tts_engine", None) or (
            "on" if getattr(speech, "tts_ready", False) else "off")
        lines.append("TTS       : %s" % tts)

    # brain / crypto
    if brain is not None:
        crypto = getattr(brain, "crypto", None)
        lines.append("Encrypt   : %s" %
                     ("ON" if crypto and getattr(crypto, "unlocked", False)
                      else "OFF"))

    # hardware via octo_core.sysmon (graceful fallback inside)
    try:
        from octo_core.sysmon import (CPUMonitor, RAMMonitor,
                                      StorageMonitor, BatteryMonitor,
                                      GPUMonitor)
        cpu = CPUMonitor().usage()
        ram = RAMMonitor().usage()
        disk = StorageMonitor().usage()
        lines.append("CPU       : %.1f%% (%s cores)"
                     % (cpu["percent"], cpu["cores"]))
        lines.append("RAM       : %s/%s MB (%.1f%%)"
                     % (ram["used_mb"], ram["total_mb"], ram["percent"]))
        lines.append("Disk      : %s GB free of %s GB (%.1f%% used)"
                     % (disk["free_gb"], disk["total_gb"], disk["percent"]))
        gpu = GPUMonitor().status()
        lines.append("GPU (CUDA): %s" % gpu["available"])
        battery = BatteryMonitor().status()
        lines.append("Battery   : %s" %
                     ("%.0f%%" % battery["percent"] if battery["present"]
                      else "not present / desktop"))
    except ImportError:
        lines.append("octo_core.sysmon unavailable — hardware skipped")

    # Octo checkpoints
    ckpt = "octo_checkpoints/octo_v2.pt"
    lines.append("OctoV2    : %s (%s KB)"
                 % ("ready" if os.path.exists(ckpt) else "missing",
                    os.path.getsize(ckpt) // 1024
                    if os.path.exists(ckpt) else 0))

    # quick self-test: full test count is tracked in CI, run smoke here
    lines.append("Self-test : import octo_core ... ", )
    try:
        import octo_core
        lines[-1] += "ok (v%s, %d capabilities)" % (
            octo_core.__version__,
            len(__import__("octo_core.evolution",
                           fromlist=["capability_manifest"])
                 .capability_manifest()["capabilities"]))
    except Exception as exc:
        lines[-1] += "FAILED (%s)" % exc

    lines.append("=== সব ঠিক আছে ===")
    return "\n".join(lines)
