# -*- coding: utf-8 -*-
"""Tests for the 185-feature expansion (tests 137-186).

Covers the genuinely-new modules: voiceint, smarthome, desktopctl,
visionx, assistant, sysmon, surveillance, researchint, robotics,
rifah and integrations.
"""
import json
import time

import pytest

from octo_core import voiceint, smarthome, desktopctl, visionx
from octo_core import assistant, sysmon, surveillance, researchint
from octo_core import robotics, rifah, integrations


# ---------------------------------------------------------------- voiceint
class TestVoiceIntelligence:
    def test_137_voice_authentication(self):
        auth = voiceint.VoiceAuthenticator()
        auth.enroll("rifah", [0.2, 0.8] * 8)
        assert auth.verify("rifah", [0.2, 0.8] * 8)["ok"]
        assert not auth.verify("rifah", [0.8, 0.2] * 8)["ok"]
        assert not auth.verify("ghost", [0.2] * 16)["ok"]

    def test_138_speaker_recognition(self):
        rec = voiceint.SpeakerRecognizer()
        rec.add_speaker("ami", [0.1, 0.2, 0.3])
        rec.add_speaker("tumi", [0.9, 0.8, 0.7])
        assert rec.identify([0.1, 0.2, 0.3])["speaker"] == "ami"

    def test_139_voice_emotion(self):
        det = voiceint.VoiceEmotionDetector()
        result = det.detect(pitch_hz=230, energy=0.9, speech_rate=2.4)
        assert result["emotion"] == "excited"
        assert det.detect(pitch_hz=100, energy=0.2,
                          speech_rate=0.5)["emotion"] == "sad"

    def test_140_vad_and_noise_gate(self):
        vad = voiceint.VoiceActivityDetector()
        out = vad.detect([0.5, 0.5, 0.001, 0.001, 0.4])
        assert out["segments"] == [(0, 1), (4, 4)]
        gate = voiceint.NoiseCanceller(floor_db=-40)
        reduced = gate.reduce([-20, -60, -10])
        assert reduced["frames_gated"] == 1
        assert reduced["levels"] == [-20, -40, -10]

    def test_141_multilingual_voice(self):
        mv = voiceint.MultilingualVoice()
        assert mv.detect_language("hello world") == "en"
        assert mv.detect_language("আমি ভালো আছি") == "bn"
        assert mv.voice_for("bn").startswith("bn-")

    def test_142_conversational_voice_memory(self):
        mem = voiceint.ConversationalVoiceMemory()
        mem.add("ami", "hello there")
        mem.add("tumi", "hi back")
        assert mem.by_speaker("ami")[0]["text"] == "hello there"
        assert mem.context()


# ---------------------------------------------------------------- smarthome
class TestSmartHome:
    def make_hub(self):
        hub = smarthome.HomeHub()
        hub.add_device("light1", "light")
        hub.add_device("fan1", "fan")
        return hub

    def test_143_device_control(self):
        hub = self.make_hub()
        assert hub.control("light1", "on")["ok"]
        assert hub.control("light1", "off")["ok"]
        assert not hub.control("ghost", "on")["ok"]

    def test_144_lock_requires_approval(self):
        hub = self.make_hub()
        hub.add_device("lock1", "lock", risky=True)
        assert not hub.control("lock1", "lock")["ok"]
        hub.approval = lambda what: True
        assert hub.control("lock1", "lock")["ok"]

    def test_145_all_off_and_schedule(self):
        hub = self.make_hub()
        hub.control("light1", "on")
        assert "light1" in hub.all_off()["turned_off"]
        hub.schedule("light1", "2020-01-01T00:00", "on")
        due = hub.due("2021-01-01T00:00")
        assert due and due[0]["device"] == "light1"

    def test_146_motion_occupancy_temperature(self):
        hub = self.make_hub()
        hub.motion_event("sensor1", True, area="bedroom")
        assert hub.security_events
        assert hub.occupancy("bedroom", True)["present"]
        assert hub.temperature("temp1", 45.0)["alert"]
        assert not hub.temperature("temp1", 25.0)["alert"]

    def test_147_energy_and_cctv(self, tmp_path):
        hub = self.make_hub()
        hub.energy("light1", 5.0)
        hub.energy("fan1", 1.0)
        assert hub.energy("light1", 2.0)["top_consumer"] == "light1"
        snap = hub.cctv_snapshot("cam1", str(tmp_path / "frame.png"))
        assert snap["camera"] == "cam1"
        assert "devices" in hub.status()


# ---------------------------------------------------------------- desktopctl
class TestDesktopControl:
    def test_148_file_ops_and_organize(self, tmp_path):
        ctl = desktopctl.DesktopController(approval=lambda a, d: True)
        src = tmp_path / "a.txt"
        src.write_text("x")
        assert ctl.file_op("copy", str(src), str(tmp_path / "b.txt"))["ok"]
        (tmp_path / "c.jpg").write_text("y")
        plan = ctl.organize_folder(str(tmp_path))
        assert len(plan["moved"]) >= 1

    def test_149_delete_requires_approval(self):
        ctl = desktopctl.DesktopController(approval=lambda a, d: False)
        assert not ctl.file_op("delete", "/tmp/nothing")["ok"]

    def test_150_launch_gated(self):
        denied = desktopctl.DesktopController(
            approval=lambda a, d: False).launch_app("octo")["reason"]
        assert denied == "not approved"
        allowed = desktopctl.DesktopController(
            approval=lambda a, d: True).launch_app("octo-no-such-bin")
        assert allowed["reason"] != "not approved"

    def test_151_troubleshoot_and_list_apps(self):
        ctl = desktopctl.DesktopController()
        assert ctl.troubleshoot("my laptop is slow")["steps"]
        assert isinstance(ctl.list_apps(), list)

    def test_152_install_package_gated(self):
        ctl = desktopctl.DesktopController(approval=lambda a, d: False)
        assert not ctl.install_package("requests")["ok"]


# ---------------------------------------------------------------- visionx
class TestVisionX:
    def test_153_face_detection(self):
        det = visionx.FaceDetector()
        result = det.detect([{"x": 0, "y": 0, "w": 50, "h": 50,
                              "skin_ratio": 0.6},
                             {"x": 5, "y": 5, "w": 5, "h": 5,
                              "skin_ratio": 0.9}])
        assert result["count"] == 1

    def test_154_person_recognition(self):
        rec = visionx.PersonRecognizer()
        rec.enroll("rifah", [0.3] * 12)
        assert rec.recognize([0.3] * 12)["known"]
        assert not rec.recognize([0.0] * 12)["known"]

    def test_155_expression_and_document(self):
        expr = visionx.ExpressionAnalyzer()
        assert expr.analyze(False, 0.1, 0.9)["expression"] == "happy"
        doc = visionx.DocumentRecognizer()
        out = doc.classify("Invoice #123 total due 45.00")
        assert out["type"] == "invoice"

    def test_156_visual_anomaly_and_video(self):
        anom = visionx.VisualAnomalyDetector()
        anom.learn_baseline([0.1] * 8)
        assert anom.detect([0.9] * 8)["anomaly"]
        va = visionx.VideoAnalyzer()
        out = va.analyze([
            {"ts": 0, "motion": 0.1, "objects": ["dog"]},
            {"ts": 1, "motion": 0.9, "objects": ["dog", "cat"]},
            {"ts": 2, "motion": 0.2, "objects": []},
        ])
        assert out["frames"] == 3 and len(out["events"]) == 3


# ---------------------------------------------------------------- assistant
class TestPersonalAssistant:
    def make_pa(self, tmp_path):
        return assistant.PersonalAssistant(
            store=str(tmp_path / "assistant.json"))

    def test_157_calendar_and_briefing(self, tmp_path):
        pa = self.make_pa(tmp_path)
        pa.calendar_add("standup", "2099-01-01T09:00")
        assert pa.calendar_upcoming()[0]["title"] == "standup"

    def test_158_reminders_due(self, tmp_path):
        pa = self.make_pa(tmp_path)
        pa.remind("call back", "2000-01-01T00:00:00")
        due = pa.due_reminders()
        assert due and due[0]["done"]

    def test_159_tasks_priority_sorted(self, tmp_path):
        pa = self.make_pa(tmp_path)
        pa.task_add("low", priority="low")
        pa.task_add("high", priority="high")
        assert pa.task_list()[0]["text"] == "high"
        assert pa.task_complete("high")

    def test_160_meeting_conflict(self, tmp_path):
        pa = self.make_pa(tmp_path)
        pa.schedule_meeting("a", "2099-01-01T10:00")
        conflict = pa.schedule_meeting("b", "2099-01-01T10:00")
        assert conflict["conflict"]

    def test_161_email_drafts_only(self, tmp_path):
        pa = self.make_pa(tmp_path)
        pa.email_draft("x@y", "hi", "body")
        assert pa.email_outbox()[0]["status"] == "draft"

    def test_162_weather_offline_cache(self, tmp_path):
        pa = self.make_pa(tmp_path)
        miss = pa.weather("dhaka")
        assert not miss["ok"] and "offline" in miss["error"]
        pa.weather("dhaka", condition="sunny", temp_c=31)
        assert pa.weather("dhaka")["temp_c"] == 31

    def test_163_news_notes_contacts(self, tmp_path):
        pa = self.make_pa(tmp_path)
        text = ("Octo is a local AI platform built for research. "
                "It runs fully offline on the user machine. "
                "The model trains locally and keeps data private. "
                "Tests verify every feature works as expected.")
        assert pa.news_digest(text)["summary"]
        pa.note_add("grid idea", tags=["idea"])
        assert pa.note_search("grid")
        pa.contact_add("rifah", phone="123")
        assert pa.contact_find("rifah")

    def test_164_travel_time_goals(self, tmp_path):
        pa = self.make_pa(tmp_path)
        plan = pa.travel_plan("cox", 3, interests=["beach", "food"])
        assert len(plan["itinerary"]) == 3
        shares = pa.time_plan([{"label": "code", "hours": 6},
                               {"label": "rest", "hours": 2}])
        assert abs(sum(b["share"] for b in shares["blocks"]) - 1.0) < 0.01
        pa.goal_add("ship octo")
        assert pa.goal_progress("ship octo", 0.5)["progress"] == 0.5

    def test_165_assistant_persists(self, tmp_path):
        store = str(tmp_path / "assistant.json")
        pa = assistant.PersonalAssistant(store=store)
        pa.task_add("persist me")
        pa2 = assistant.PersonalAssistant(store=store)
        assert any(t["text"] == "persist me" for t in pa2.task_list())


# ---------------------------------------------------------------- sysmon
class TestSystemIntelligence:
    def test_166_cpu_ram_disk_monitors(self):
        assert 0 <= sysmon.CPUMonitor().usage()["percent"] <= 100
        ram = sysmon.RAMMonitor().usage()
        assert ram["total_mb"] >= 0
        disk = sysmon.StorageMonitor().usage()
        assert 0 <= disk["percent"] <= 100

    def test_167_gpu_battery_temperature(self):
        assert "backend" in sysmon.GPUMonitor().status()
        assert "present" in sysmon.BatteryMonitor().status()
        assert "available" in sysmon.TemperatureMonitor().read()

    def test_168_process_top(self):
        top = sysmon.ProcessMonitor().top(3)
        assert isinstance(top, list) and len(top) <= 3

    def test_169_optimizer_and_allocator(self):
        rec = sysmon.PerformanceOptimizer().recommend(95, 95, 95)
        assert rec["tips"] and not rec["healthy"]
        alloc = sysmon.ResourceAllocator().allocate(10, 10)
        assert alloc["workers"] >= 1

    def test_170_crash_detection_and_recovery(self):
        det = sysmon.CrashDetector(threshold=3)
        for _ in range(3):
            det.record("svc", ok=False)
        assert det.crashed("svc")
        mgr = sysmon.RecoveryManager()
        mgr.register("svc", lambda: True)
        assert mgr.recover("svc")["ok"]

    def test_171_health_dashboard_snapshot(self):
        snap = sysmon.HealthDashboard().snapshot()
        assert "cpu" in snap and "healthy" in snap


# ---------------------------------------------------------------- surveillance
class TestSurveillance:
    def test_172_intrusion_detection(self):
        det = surveillance.IntrusionDetector(max_failures=3)
        for _ in range(3):
            result = det.record_failure("door")
        assert result["intrusion"]
        assert det.check_time(3)["unusual_hour"]

    def test_173_unauthorized_access(self):
        det = surveillance.UnauthorizedAccessDetector()
        det.grant("rifah", "vault")
        assert det.check("rifah", "vault")["allowed"]
        assert det.check("stranger", "vault")["violation"]

    def test_174_face_access_control(self):
        ctl = surveillance.FaceAccessController()
        ctl.enroll("rifah", [0.4] * 10)
        assert ctl.request_entry([0.4] * 10)["granted"]
        assert not ctl.request_entry([0.0] * 10)["granted"]

    def test_175_suspicious_activity(self):
        det = surveillance.SuspiciousActivityDetector()
        result = det.inspect("please delete everything now")
        assert "destructive" in result["tags"]

    def test_176_network_monitor_and_malware(self, tmp_path):
        net = surveillance.NetworkSecurityMonitor()
        flagged = net.audit_connections([{"remote": "x", "port": 4444}])
        assert not flagged["clean"]
        target = tmp_path / "eicar.txt"
        target.write_bytes(b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR")
        scan = surveillance.MalwareScanner().scan_file(str(target))
        assert not scan["clean"]

    def test_177_malware_scan_dir(self, tmp_path):
        scanner = surveillance.MalwareScanner()
        (tmp_path / "ok.txt").write_text("hello")
        result = scanner.scan_dir(str(tmp_path))
        assert result["scanned"] >= 1 and not result["threats"]

    def test_178_emergency_alerts_and_camera(self):
        alerts = surveillance.EmergencyAlerts()
        alerts.raise_alert("breach", level="critical")
        assert alerts.critical()
        cam = surveillance.CameraAnalyzer()
        cam.analyze_frame([0.1, 0.9] * 4)
        assert cam.analyze_frame([0.9, 0.1] * 4)["motion"]


# ---------------------------------------------------------------- researchint
class TestResearchIntelligence:
    def test_179_website_analyzer(self):
        html = "<h1>Title</h1><a href='http://x'>l</a><p>some words here</p>"
        out = researchint.WebsiteAnalyzer().analyze(html)
        assert out["headings"] and out["external_links"]

    def test_180_fact_checker(self):
        fc = researchint.FactChecker()
        assert fc.check("octo runs offline",
                        ["octo runs offline locally"])["verdict"] \
            == "supported"
        assert fc.check("octo runs offline",
                        ["octo does not run offline"])["verdict"] \
            in ("disputed", "unsupported")

    def test_181_news_monitor_trending(self):
        nm = researchint.NewsMonitor()
        nm.add("a", "transformer models are evolving quickly")
        nm.add("b", "the transformer architecture improves reasoning")
        assert nm.by_topic("transformer")
        assert nm.trending(1)[0][0] == "transformer"

    def test_182_document_analyzer(self):
        doc = researchint.DocumentAnalyzer().analyze(
            "Contact research@octo.ai for version 1.4 details.\n\nMore text.")
        assert doc["emails"] and doc["paragraphs"] == 2

    def test_183_report_generator(self):
        out = researchint.ReportGenerator().generate(
            "Findings", [("Summary", "all good"), ("Next", "ship it")])
        assert out["sections"] == 2 and out["report"].startswith("# ")

    def test_184_source_verifier(self):
        sv = researchint.SourceVerifier()
        good = sv.verify({"name": "paper", "note": "peer-reviewed",
                          "author": "a", "date": "2024",
                          "corroborations": 2})
        assert good["trust"] == "high"
        assert sv.verify({"name": "blog"})["trust"] == "low"


# ---------------------------------------------------------------- robotics
class TestRobotics:
    def make_bot(self):
        return robotics.RobotInterface(grid_size=6)

    def test_185_sensor_bus(self):
        bus = robotics.SensorBus()
        bus.publish("temp", 30.0)
        assert bus.latest("temp") == 30.0
        assert bus.average("temp") == 30.0

    def test_186_motor_control_and_estop(self):
        bot = self.make_bot()
        assert bot.motors.command("left", 0.5)["ok"]
        bot.emergency_stop()
        assert bot.motors.estop
        assert not bot.motors.command("left", 0.5)["ok"]
        bot.reset_estop()
        assert bot.motors.command("left", 0.5)["ok"]

    def test_187_obstacle_detection_and_path(self):
        nav = robotics.Navigator(grid_size=5)
        nav.add_obstacle(1, 0)
        assert nav.obstacle_ahead("E")["blocked"]
        plan = nav.plan((2, 0))
        assert plan["ok"] and (1, 0) not in plan["path"]

    def test_188_mapping_and_autonomous_move(self):
        bot = self.make_bot()
        result = bot.run_autonomous((2, 2))
        assert result["ok"] and result["moves"] == 4
        assert bot.navigator.map()["coverage"] > 0

    def test_189_manipulation_state_machine(self):
        m = robotics.Manipulator()
        assert m.act("reach")["ok"]
        assert m.act("grasp", obj="box")["ok"]
        assert m.act("release")["ok"] and m.holding is None

    def test_190_robot_status(self):
        bot = self.make_bot()
        status = bot.status()
        assert status["name"] == "octo-bot" and status["healthy"]


# ---------------------------------------------------------------- rifah
class TestRifah:
    def test_191_holographic_projection_rotation(self):
        holo = rifah.HolographicInterface()
        holo.project("cube", [(0, 0, 0), (1, 0, 0), (1, 1, 0)])
        out = holo.rotate("cube", axis="z", degrees=90)
        assert out["ok"] and out["vertices"] == 3

    def test_192_ar_and_hud(self):
        ar = rifah.ARInterface().overlay({"w": 640}, ["status"])
        assert ar["anchored"] == 1
        hud = rifah.HUD()
        hud.update("power", "82%", priority=0)
        hud.update("nav", "north", priority=2)
        assert hud.render()["hud"][0]["panel"] == "power"

    def test_193_suit_diagnostics_and_armor(self):
        suit = rifah.SuitDiagnostics()
        suit.wear("power", 80)
        suit.impact("plate_0", 50)
        report = suit.report()
        assert "power" in report["critical"]
        assert "plate_0" in report["breached"]

    def test_194_flight_telemetry(self):
        ft = rifah.FlightTelemetry()
        ft.record(100, 40, 0)
        ft.record(200, 45, 5)
        out = ft.analyze()
        assert out["ok"] and out["climb_m"] == 100

    def test_195_tactical_and_target_tracking(self):
        tac = rifah.TacticalAnalyzer()
        contacts = [{"distance": 10, "hostile": True, "id": f"t{i}"}
                    for i in range(3)]
        pic = tac.tactical_picture(contacts, ["smoke detected"])
        assert pic["recommended"] == "evade"
        assert pic["environmental"] == ["smoke"]
        tracker = rifah.TargetTracker()
        update = tracker.update(contacts)
        assert update["locked"].startswith("t")

    def test_196_emergency_response(self):
        out = rifah.EmergencyResponseCoordinator().coordinate("fire in lab")
        assert out["protocol"] == "fire" and out["coordinated"]

    def test_197_hardware_diagnostics_maintenance(self):
        hd = rifah.HardwareDiagnostics()
        hd.register("thruster", 950, 1000, health=35)
        diag = hd.diagnose()
        assert "thruster" in diag["attention"]
        plan = hd.predictive_maintenance(horizon_hours=200)
        assert plan["schedule"][0]["component"] == "thruster"


# ---------------------------------------------------------------- integrations
class TestIntegrations:
    def test_198_message_bus(self):
        bus = integrations.LocalMessageBus()
        seen = []
        bus.subscribe("watch", lambda m: seen.append(m))
        out = bus.publish("test.topic", {"a": 1})
        assert "watch" in out["delivered_to"] and seen

    def test_199_smartphone_and_watch(self):
        bus = integrations.LocalMessageBus()
        phone = integrations.SmartphoneIntegration(bus)
        assert not phone.relay_notification("mail", "hi")["ok"]
        phone.pair("pixel")
        relayed = phone.relay_notification("mail", "hi")
        assert "watch" not in relayed["delivered_to"]
        watch = integrations.SmartwatchIntegration(bus)
        assert watch.ingest(72, 1000)["heart_rate"] == 72

    def test_200_iot_local_only(self):
        iot = integrations.IoTIntegration()
        iot.register("bulb1", "light", room="lab")
        assert iot.send("bulb1", "on")["topic"] == "iot.command"
        assert not iot.send("ghost", "on")["ok"]
        iot.devices["bulb1"]["online"] = False
        assert not iot.send("bulb1", "on")["ok"]

    def test_201_api_integration_local_only(self):
        api = integrations.APIIntegration()
        with pytest.raises(ValueError):
            api.register("cloud", lambda: None, local=False)
        api.register("local_fn", lambda x: x * 2)
        assert api.call("local_fn", 21)["result"] == 42

    def test_202_local_vault_encrypted(self, tmp_path):
        vault = integrations.LocalVault(
            path=str(tmp_path / "vault.bin"),
            key_path=str(tmp_path / "vault.key"))
        vault.put("secret", {"k": "v"})
        raw = (tmp_path / "vault.bin").read_bytes()
        assert b"secret" not in raw
        again = integrations.LocalVault(
            path=str(tmp_path / "vault.bin"),
            key_path=str(tmp_path / "vault.key"))
        assert again.get("secret") == {"k": "v"}

    def test_203_cross_device_sync(self, tmp_path):
        vault = integrations.LocalVault(
            path=str(tmp_path / "vault.bin"),
            key_path=str(tmp_path / "vault.key"))
        sync = integrations.CrossDeviceSync(vault)
        sync.push("mac", {"tab": 1})
        assert sync.pull("mac") == {"tab": 1}
        assert sync.conflicts()["devices"] == ["mac"]

    def test_204_remote_access_guard(self):
        guard = integrations.RemoteAccessGuard()
        assert not guard.open_session("u")["ok"]
        assert not guard.enable("bad", "good")["ok"]
        assert guard.enable("good", "good")["ok"]
        assert guard.open_session("u")["ok"]
        assert guard.disable()["sessions_closed"]

    def test_205_secure_messenger_encrypted(self):
        msgr = integrations.SecureMessenger()
        msgr.send("lab", "octo", "secret plan")
        raw = msgr.threads["lab"][0]
        assert b"secret plan" not in raw
        assert msgr.read("lab")[0]["text"] == "secret plan"

    def test_206_notification_center(self):
        nc = integrations.NotificationCenter()
        assert nc.notify("alert", priority=0)["ok"]
        assert not nc.notify("alert")["ok"]
        assert nc.pending()[0]["priority"] == 0
        assert nc.dismiss("alert")["dismissed"] == 1

    def test_207_multi_platform_profiles(self):
        for system in ("linux", "darwin", "nt"):
            profile = integrations.MultiPlatformSupport().profile(system)
            assert profile["ui"] == "tkinter"


class TestCryptoFallback:
    def test_208_xor_cipher_fallback(self):
        cipher = integrations._XorCipher(b"octo-test-key")
        token = cipher.encrypt(b"secret vault data")
        assert b"secret vault data" not in token
        assert cipher.decrypt(token) == b"secret vault data"

    def test_209_vault_without_cryptography(self, tmp_path, monkeypatch):
        import octo_core.integrations as integ
        monkeypatch.setattr(integ, "_HAVE_FERNET", False)
        vault = integ.LocalVault(path=str(tmp_path / "v.bin"),
                                 key_path=str(tmp_path / "v.key"))
        assert isinstance(vault._fernet, integ._XorCipher)
        vault.put("k", {"a": 1})
        raw = (tmp_path / "v.bin").read_bytes()
        assert b'"a": 1' not in raw
        again = integ.LocalVault(path=str(tmp_path / "v.bin"),
                                 key_path=str(tmp_path / "v.key"))
        assert again.get("k") == {"a": 1}


# --------------------------------------------------- OCTO AI master system
class TestMasterSystem:
    def make_orch(self, tmp_path):
        from octo_core.master import AgentRegistry, MasterOrchestrator
        reg = AgentRegistry()
        reg.register("coding", handler=lambda task, payload:
                     {"wrote": task, "keywords": "def main()"})
        reg.register("research", handler=lambda task, payload:
                     {"sources_checked": 3, "topic": task})
        orch = MasterOrchestrator(registry=reg)
        return orch

    def test_210_master_prompt_and_registry(self, tmp_path, monkeypatch):
        from octo_core.master import AgentRegistry, MASTER_SYSTEM_PROMPT
        assert "Think. Plan. Execute. Verify. Deliver." in MASTER_SYSTEM_PROMPT
        monkeypatch.chdir(tmp_path)
        reg = AgentRegistry()
        assert len(reg.CATALOG) == 20
        assert not reg.is_available("coding")  # nothing capable by default
        reg.register("coding", handler=lambda t, p: t)
        assert reg.is_available("coding")
        assert reg.available("security") == []  # division filter
        with pytest.raises(KeyError):
            reg.register("fabricated-agent")

    def test_211_intent_requirements_decomposition(self, tmp_path,
                                                   monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = self.make_orch(tmp_path)
        assert orch.analyze_intent("debug the payment bug")["intent"] \
            == "coding"
        req = orch.analyze_requirements("refactor in python, keep it offline")
        assert req["language"] == "python" and "offline" in req["constraints"]
        steps = orch.decompose("first scan the host. then analyze services.")
        assert steps[0] == "first scan the host"
        assert steps[1] == "analyze services"  # connectors stripped

    def test_212_full_pipeline_delivery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = self.make_orch(tmp_path)
        out = orch.run("write python code for the parser. then research parsing.")
        assert set(out) >= {"intent", "requirements", "decomposition",
                            "agent_assignment", "execution",
                            "verification", "delivery", "project"}
        assert out["delivery"]["status"] in ("COMPLETED", "BLOCKED")
        project = orch.projects.projects[out["project"]]
        assert project["requirements"] == out["requirements"]

    def test_213_no_fabricated_agents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = self.make_orch(tmp_path)
        out = orch.run("rotate the backup tapes on mars")  # unassignable
        assert out["failures"]
        assert all(f["error"] == "no registered agent can handle this"
                   for f in out["failures"])
        assert out["delivery"]["status"] == "BLOCKED"

    def test_214_level_3_agent_needs_approval(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from octo_core.master import AgentRegistry
        reg = AgentRegistry()
        called = []
        reg.register("business", handler=lambda t, p: called.append(t),
                     level=3)
        out = reg.run("business", "send invoice")
        assert out["status"] == "REQUIRES_APPROVAL" and not called
        out = reg.run("business", "send invoice",
                      approve=lambda a, d: True)
        assert out["ok"] and called
        out2 = reg.run("business", "send invoice")
        assert not out2["ok"]  # still gated next time

    def test_215_registry_run_unregistered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from octo_core.master import AgentRegistry
        reg = AgentRegistry()
        out = reg.run("pentest", "attack target")
        assert out["status"] == "not_registered"

    def test_216_permission_levels(self):
        from octo_core.master import PermissionLevels
        assert PermissionLevels.required_level("read the log") == 0
        assert PermissionLevels.required_level("save the note") == 1
        assert PermissionLevels.required_level("run the test") == 2
        assert PermissionLevels.required_level("deploy to prod") == 4
        assert PermissionLevels.required_level("rm -rf data") == 4
        check = PermissionLevels.check("deploy to prod", granted_level=2)
        assert not check["allowed"] and check["requires"]
        allowed = PermissionLevels.check("deploy to prod", granted_level=2,
                                         approve=lambda a, d: True)
        assert allowed["allowed"] and allowed["confirmed"]
        assert PermissionLevels.confirmations_for(4)

    def test_217_quality_gate_and_correction_loop(self):
        from octo_core.master import QualityGate
        gate = QualityGate()
        bad = {c: False for c in gate.CHECKS}
        assert not gate.evaluate(bad)["pass"]
        good = {c: True for c in gate.CHECKS}
        assert gate.evaluate(good, expect_keywords=("octo",))["pass"] is False
        result = QualityGate.correction_loop(
            lambda r: r["n"] >= 2,
            lambda attempt: {"n": attempt + 1})
        assert result["ok"] and result["attempts"] == 2
        failed = QualityGate.correction_loop(
            lambda r: False, lambda a: {}, retries=2)
        assert not failed["ok"] and failed["attempts"] == 2

    def test_218_model_routing_policy(self):
        from octo_core.master import ModelRoutingPolicy
        router = ModelRoutingPolicy()
        router.register("octo-fast", ("coding",), cost=0.2, latency=0.2)
        router.register("octo-reason", ("reasoning", "coding"), cost=0.8,
                        latency=0.8)
        router.register("cloud-vision", ("vision",), local=False)
        assert router.route({"capability": "reasoning"})["model"] \
            == "octo-reason"
        coding = router.route({"capability": "coding", "latency": "fast"})
        assert coding["model"] == "octo-fast"
        private = router.route({"capability": "vision", "private": True})
        assert private["available"] is False  # never assumes a private cloud model
        offline = router.route({"capability": "voice"})
        assert offline["model"] is None and not offline["available"]

    def test_219_error_recovery_loop(self, tmp_path, monkeypatch):
        from octo_core.master import ErrorRecoveryLoop
        monkeypatch.chdir(tmp_path)
        loop = ErrorRecoveryLoop(max_retries=2)
        assert loop.classify("connection reset by peer") == "network"
        assert loop.classify("request timeout after 30s") == "timeout"
        report = loop.handle("timeout", retry=lambda: "ok",
                             verify=lambda out: out == "ok")
        assert report["recovered"] and report["verified"]
        destructive = loop.handle("rm -rf /tmp/target failed",
                                  retry=lambda: "reran")
        assert not destructive["retried"]
        assert destructive["reason"].startswith("destructive")

    def test_220_idempotency_keys(self, tmp_path, monkeypatch):
        from octo_core.master import ErrorRecoveryLoop
        monkeypatch.chdir(tmp_path)
        loop = ErrorRecoveryLoop()
        calls = []
        first = loop.idempotent("deploy-42", lambda: calls.append(1) or "done")
        again = loop.idempotent("deploy-42", lambda: calls.append(1) or "done")
        assert first["result"] == again["result"] == "done"
        assert first["duplicate"] is False and again["duplicate"] is True
        assert len(calls) == 1  # ran exactly once

    def test_221_project_lifecycle(self, tmp_path, monkeypatch):
        from octo_core.master import ProjectManager
        monkeypatch.chdir(tmp_path)
        pm = ProjectManager(store=str(tmp_path / "projects.json"))
        pid = pm.create("ship octo", requirements=["offline"], agents=["coding"])
        assert pm.transition(pid, "IN_PROGRESS")["ok"]
        assert not pm.transition(pid, "COMPLETED")["ok"]  # invalid jump
        pm.add_task(pid, "write code", assigned="coding")
        pm.add_task(pid, "write tests", assigned="testing")
        pm.complete_task(pid, "write code")
        pm.milestone(pid, "alpha")
        assert pm.projects[pid]["progress"] == 0.5
        resume = pm.resume(pid)
        assert resume["pending"] == ["write tests"]
        pm.complete_task(pid, "write tests")
        pm.transition(pid, "UNDER_VERIFICATION")
        assert pm.transition(pid, "COMPLETED")["ok"]
        stored = ProjectManager(store=str(tmp_path / "projects.json"))
        assert stored.projects[pid]["status"] == "COMPLETED"

    def test_222_security_report_builder(self):
        from octo_core.master import SecurityReportBuilder
        builder = SecurityReportBuilder()
        empty = builder.build()
        assert not empty["complete"] and "scope" in empty["missing"]
        builder.set("scope", "lab only").set("methodology", "OWASP") \
            .set("findings", "2 low").set("risk_level", "low") \
            .set("remediation", "patch").set("evidence", "logs") \
            .set("impact", "minimal").set("verification_results", "retested")
        out = builder.build()
        assert out["complete"]
        assert "# Security Assessment" in builder.to_markdown()

    def test_223_conflict_resolution_majority(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = self.make_orch(tmp_path)
        resolved = orch.resolve_conflicts({
            "agentA": {"answer": "x"}, "agentB": {"answer": "x"},
            "agentC": {"answer": "y"},
        })
        assert resolved["answer"]["value"] == "x"  # majority
        assert sorted(resolved["answer"]["sources"]) == \
            ["agentA", "agentB", "agentC"]

    def test_224_session_master_integration(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "notes.txt").write_text(
            "octo master orchestrator coordinates specialist agents with "
            "verification and permission levels built in.", encoding="utf-8")
        from octo_core.desktop import OctoSession
        s = OctoSession(roots=(str(corpus),),
                        checkpoint="/nonexistent/octo.pt")
        status = s.master_status()
        assert status["agents_catalogued"] == 20
        assert status["agents_registered"] == []
        s.master_enable("research", "coding")
        assert "research" in s.master_status()["agents_registered"]
        out = s.master_run("research the topic. then code the fix.")
        assert "delivery" in out

    def test_225_agents_with_generic_handler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from octo_core.master import AgentRegistry
        reg = AgentRegistry()
        reg.register("planning", handler=lambda t, p: {"steps": [t]})
        out = reg.run("planning", "plan the sprint", payload="")
        assert out["ok"] and out["result"] == {"steps": ["plan the sprint"]}
        assert reg.agents["planning"]["status"] == "idle"
        reg.register("coding", handler=lambda t, p: 1 / 0)
        bad = reg.run("coding", "boom")
        assert not bad["ok"] and bad["status"] == "FAILED"
        assert reg.agents["coding"]["status"] == "error"
