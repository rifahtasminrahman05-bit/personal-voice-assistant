# -*- coding: utf-8 -*-
import os
import textwrap

import pytest

from octo_core.cogx import (AdaptiveReasoningEngine, CognitiveLoadManager,
                            CounterfactualReasoning, CausalReasoningEngine,
                            MetacognitiveController, UncertaintyAwareness)
from octo_core.swarm import (CollaborationNetwork, DynamicAgentFactory,
                             LongTermGoalManager, NegotiationEngine,
                             ReputationSystem, WorkflowComposer)
from octo_core.memx import (AssociativeMemoryNetwork, CrossDomainTransfer,
                            KnowledgeEvolutionTracker, MemoryConsolidationEngine,
                            MemoryDecayManager, SemanticCompressor)
from octo_core.evolvex import (AdaptiveCurriculum, ContinualLearningEngine,
                               ExperienceReplaySystem, ModelAdaptationController,
                               SelfImprovementEvalLoop, SkillAcquisitionEngine)
from octo_core.perception import (AudioSceneAnalyzer, GestureRecognizer,
                                  MultimodalContextFusion, ObjectTracker,
                                  SceneUnderstandingEngine,
                                  SpatialReasoningEngine)
from octo_core.devx import (ArchitectureDesigner, AutonomousDebugger,
                            CodeOptimizer, DependencyRiskAnalyzer,
                            DocumentationEngine, MigrationIntelligence)
from octo_core.secopsx import (CloudSecurityAnalyzer, ConfigRiskDetector,
                               IncidentResponsePlanner, SecurityPostureAnalyzer,
                               SupplyChainAnalyzer, ThreatIntelCorrelator,
                               VulnerabilityPrioritizer, ZeroTrustAdvisor)
from octo_core.sysint import (DynamicResourceAllocator, DistributedExecutor,
                              FailurePredictor, PriorityScheduler,
                              SelfHealingManager, SmartCacheManager)
from octo_core.humanai import (AdaptiveTeachingEngine, CollaborativeProblemSolver,
                               DialogueManager, ExplainableDecisionEngine,
                               IntentPredictor, PersonalizedInteractionEngine)
from octo_core.frontier import (ArtificialSocietySimulator,
                                AutonomousResearchPlanner,
                                CognitiveArchitectureEvolutionEngine,
                                EmergenceAnalyzer)


class TestCognitiveX:
    def test_01_load_manager_allocates_by_complexity(self):
        mgr = CognitiveLoadManager()
        easy = mgr.allocate("what is 2+2?")
        hard = mgr.allocate(
            "prove and derive why this multi-step architecture design is "
            "optimal compared to the alternatives under the given constraints")
        assert easy["level"] in ("low", "medium")
        assert hard["level"] == "high"
        assert hard["steps"] > easy["steps"]

    def test_02_adaptive_reasoning_selects_strategy(self):
        engine = AdaptiveReasoningEngine()
        assert engine.select_strategy("calculate the sum of squares")["strategy"] == "stepwise-arithmetic"
        assert engine.select_strategy("fix this bug in the code")["strategy"] == "decompose-and-test"
        assert engine.select_strategy("hello")["strategy"] == "chain-of-thought"
        assert len(engine.reason("research this topic")["steps"]) == 3

    def test_03_causal_reasoning_ranks_causes(self):
        engine = CausalEngine = CausalReasoningEngine()
        episodes = [
            ["deploy", "config drift", "outage"],
            ["deploy", "config drift", "outage"],
            ["deploy", "cache flush", "ok"],
            ["patch", "outage"],
        ]
        report = engine.infer(episodes, "outage")
        assert report["causes"][0]["cause"] == "config drift"
        assert engine.is_plausible_cause("deploy", "outage", episodes[0])

    def test_04_counterfactual_finds_better_option(self):
        out = CounterfactualReasoning().analyze(
            {"name": "chosen", "value": 5},
            [{"name": "alt-a", "value": 8}, {"name": "alt-b", "value": 2}],
            baseline_value=5)
        assert out["regret"] == 3
        assert out["would_have_chosen"] == "alt-a"
        assert out["scenarios"][0]["decision"] == "alt-a"

    def test_05_metacognitive_monitor_flags_gaps(self):
        ctrl = MetacognitiveController()
        bad = ctrl.monitor([{"step": "claim", "has_evidence": False, "verifies": False}])
        assert bad["verdict"] == "needs-review"
        assert ctrl.should_escalate(bad)
        good = ctrl.monitor([{"step": "claim", "has_evidence": True, "verifies": True}])
        assert good["verdict"] == "sound" and not ctrl.should_escalate(good)

    def test_06_uncertainty_awareness(self):
        ua = UncertaintyAwareness()
        grounded = ua.quantify("parameterized queries prevent sql injection",
                               evidence="sql injection is prevented by parameterized queries")
        hedged = ua.quantify("maybe this might possibly work, unclear")
        assert grounded["uncertainty"] < hedged["uncertainty"]
        assert hedged["hedges"] >= 3
        assert hedged["label"] in ("medium", "high")


class TestSwarm:
    def test_07_collaboration_network_routing(self):
        net = CollaborationNetwork()
        net.register("alice", ("security",))
        net.register("bob", ("coding",))
        net.post("bob", "security", "found eval() in parser")
        assert net.inbox("alice") and not net.inbox("bob")
        assert net.collaborators("security") == ["alice"]

    def test_08_negotiation_weighted_consensus(self):
        engine = NegotiationEngine()
        proposals = [{"agent": "a", "plan": "rewrite", "score": 0.9},
                     {"agent": "b", "plan": "rewrite", "score": 0.8},
                     {"agent": "c", "plan": "patch", "score": 0.7}]
        out = engine.negotiate(proposals, weights={"a": 3})
        assert out["agreed"] == "rewrite" and out["consensus"] is True
        resolved = engine.resolve_conflict(
            [{"agent": "x", "claim": "p1", "confidence": 0.4},
             {"agent": "y", "claim": "p2", "confidence": 0.9}])
        assert resolved["claim"] == "p2"

    def test_09_dynamic_agent_creation(self):
        spec = DynamicAgentFactory().create("audit the security of this module")
        assert spec["role"] == "security-analyst"
        assert "scan" in spec["tools"]
        registry = {}
        DynamicAgentFactory().create("write a test for the bug", registry)
        assert len(registry) == 1

    def test_10_reputation_system(self):
        rep = ReputationSystem()
        for _ in range(5):
            rep.record("good", True)
        for _ in range(5):
            rep.record("bad", False)
        assert rep.score("good") > rep.score("bad")
        assert rep.rank()[0][0] == "good"
        assert rep.select(["good", "bad"]) == "good"

    def test_11_workflow_composer(self):
        wf = WorkflowComposer().compose("research the latest exploit reports")
        assert wf["steps"][0] == "search sources"
        assert len(WorkflowComposer().compose("hello")["steps"]) == 4

    def test_12_long_term_goal_manager(self):
        mgr = LongTermGoalManager()
        mgr.add("g1", "ship octo", ["design", "implement", "test"])
        mgr.complete("g1", "design")
        mgr.complete("g1", "implement")
        assert mgr.progress("g1")["percent"] == 0.67
        assert mgr.next_action("g1") == "test"
        mgr.complete("g1", "test")
        assert mgr.next_action("g1") is None


class TestMemoryX:
    def test_13_associative_memory_spread(self):
        net = AssociativeMemoryNetwork()
        net.store("octo", "the assistant")
        net.store("transformer", "neural net")
        net.store("attention", "mechanism")
        net.link("octo", "transformer")
        net.link("transformer", "attention")
        hits = net.associate("octo", depth=2)
        assert hits[0][0] == "transformer"
        assert any(k == "attention" for k, _, _ in hits)

    def test_14_memory_consolidation(self):
        eng = MemoryConsolidationEngine(threshold=3)
        assert eng.touch("fact") == "short-term"
        eng.touch("fact")
        assert eng.touch("fact") == "consolidated"
        assert eng.recall("fact")["store"] == "long-term"

    def test_15_memory_decay_forgets(self):
        mgr = MemoryDecayManager(rate=0.5, floor=0.1)
        mgr.learn("strong", 1.0)
        mgr.learn("weak", 0.15)
        forgotten = mgr.decay(cycles=1)
        assert "weak" in forgotten and "strong" not in forgotten
        mgr.access("strong")
        assert mgr.scores["strong"] == pytest.approx(0.7)

    def test_16_knowledge_evolution_tracker(self):
        tracker = KnowledgeEvolutionTracker()
        tracker.record("dns", "uses port 53", ts="2024-01-01")
        tracker.record("dns", "uses port 53 over tcp too", ts="2024-06-01")
        assert tracker.changed("dns")
        assert tracker.current("dns").startswith("uses port 53 over")
        assert len(tracker.evolution("dns")) == 2
        assert not tracker.changed("missing")

    def test_17_semantic_compression(self):
        text = ("The transformer model uses attention. It processes tokens in "
                "parallel. Attention weights decide which tokens matter most. "
                "Training uses backpropagation over large corpora.")
        out = SemanticCompressor().compress(text, keep=2)
        assert out["ratio"] < 0.8
        assert "attention" in out["summary"].lower()

    def test_18_cross_domain_transfer(self):
        xfer = CrossDomainTransfer()
        xfer.learn("biology", [("neuron", "transmits", "signal"),
                               ("heart", "pumps", "blood")])
        out = xfer.transfer("biology", "ai", {"neuron": "perceptron",
                                              "signal": "activation"})
        assert ("perceptron", "transmits", "activation") in out
        assert ("perceptron", "transmits", "activation") in xfer.domains["ai"]
        assert xfer.shared_relations("biology", "ai") == ["pumps", "transmits"] \
            or "transmits" in xfer.shared_relations("biology", "ai")


class TestEvoX:
    def test_19_continual_learning_rehearsal(self):
        eng = ContinualLearningEngine(seed=1)
        eng.add_experience("old-1")
        out = eng.incorporate(["new-1", "new-2", "new-3", "new-4"], replay_ratio=0.25)
        assert out["new"] == 4 and out["replayed"] == 1
        assert out["trained_on"] == 5

    def test_20_experience_replay_prioritized(self):
        replay = ExperienceReplaySystem()
        replay.add("meh", reward=0.0)
        replay.add("great", reward=10.0)
        assert replay.best() == "great"
        assert len(replay.sample(k=2)) == 2
        assert replay.sample(k=0) == []

    def test_21_adaptive_curriculum(self):
        cur = AdaptiveCurriculum()
        for score in (0.9, 0.95, 0.85):
            cur.record(score)
        out = cur.next_task({"easy": ["e"], "medium": ["m"], "hard": ["h1", "h2"]})
        assert out["level"] == "hard" and out["task"] == "h1"
        cur2 = AdaptiveCurriculum()
        assert cur2.next_task({"easy": ["e"], "medium": ["m"]})["level"] == "easy"

    def test_22_skill_acquisition(self):
        eng = SkillAcquisitionEngine()
        assert eng.acquire("run benchmark", ["load suite", "record"]) == "run benchmark"
        assert eng.acquire("bad skill", ["x"], validated=False) is None
        assert eng.match("please run benchmark now")["workflow"] == ["load suite", "record"]
        assert eng.match("unrelated") is None

    def test_23_model_adaptation_controller(self):
        ctrl = ModelAdaptationController()
        out = ctrl.adapt({"too_random": True, "too_slow": True})
        assert out["temperature"] == pytest.approx(0.6)
        assert out["steps"] == 96
        out2 = ctrl.adapt({"low_quality": True})
        assert out2["steps"] > 96

    def test_24_self_improvement_eval_loop(self):
        loop = SelfImprovementEvalLoop()
        out = loop.evaluate([0.5, 0.6], [0.6, 0.7], approved=False)
        assert out["improved"] and not out["approved"]
        assert loop.evaluate([0.5], [0.6], approved=True)["approved"]
        assert not loop.evaluate([0.7], [0.6], approved=True)["approved"]
        assert not loop.evaluate([0.5], [], approved=True)["approved"]


class TestPerception:
    def test_25_scene_understanding(self):
        out = SceneUnderstandingEngine().understand(
            {"objects": [{"label": "car"}, {"label": "car"}, {"label": "person"}]})
        assert out["dominant"] == "car"
        assert out["labels"] == {"car": 2, "person": 1}
        assert SceneUnderstandingEngine().understand({"objects": []})["description"] == "empty scene"

    def test_26_object_tracking_ids(self):
        tracker = ObjectTracker(max_distance=15)
        f1 = tracker.update([{"label": "car", "x": 10, "y": 10}])
        f2 = tracker.update([{"label": "car", "x": 12, "y": 11}])
        assert f1[0]["id"] == f2[0]["id"]
        new = tracker.update([{"label": "car", "x": 200, "y": 200}])
        assert new[0]["id"] != f1[0]["id"]
        assert len(tracker.active()) == 2

    def test_27_spatial_reasoning(self):
        eng = SpatialReasoningEngine()
        out = eng.relations({"x": 0, "y": 0}, {"x": 10, "y": 1})
        assert "right of" in out["relations"]
        assert eng.relations({"x": 0, "y": 0}, {"x": 0, "y": 0})["relations"] == ["near"]
        assert eng.path_length([{"x": 0, "y": 0}, {"x": 3, "y": 4}]) == 5.0

    def test_28_gesture_recognition(self):
        rec = GestureRecognizer()
        assert rec.classify([{"x": 5, "y": 5}])["gesture"] == "static"
        assert rec.classify([{"x": 0, "y": 0}, {"x": 3, "y": 1}])["gesture"] == "static"
        assert rec.classify([{"x": 0, "y": 0}, {"x": 50, "y": 2}])["gesture"] == "swipe_right"
        assert rec.classify([{"x": 0, "y": 0}, {"x": 1, "y": -50}])["gesture"] == "swipe_down"

    def test_29_audio_scene(self):
        analyzer = AudioSceneAnalyzer()
        assert analyzer.classify({"speech_ratio": 0.8})["environment"] == "conversation"
        assert analyzer.classify({"volume": 0.1, "speech_ratio": 0.0})["environment"] == "quiet-room"
        assert analyzer.classify({"volume": 0.9, "periodic": True})["environment"] == "machinery"

    def test_30_multimodal_fusion(self):
        out = MultimodalContextFusion().fuse(
            text="a red car parked outside",
            image_labels=["car", "tree"], audio="engine hum")
        assert "car" in out["cross_modal_overlap"]
        assert set(out["modalities"]) == {"text", "image", "audio"}
        assert out["confidence"] == pytest.approx(0.9)


class TestDevX:
    def test_31_autonomous_debugger(self):
        dbg = AutonomousDebugger()
        out = dbg.diagnose(textwrap.dedent("""
            Traceback (most recent call last):
              File "app.py", line 3, in <module>
            KeyError: 'user_id'
        """))
        assert out["error"] == "KeyError"
        assert ".get()" in out["suggestion"]
        assert dbg.diagnose("something odd")["error"] == "unknown"

    def test_32_architecture_designer(self):
        design = ArchitectureDesigner().design(
            "users can log in, upload files, and view analytics dashboards")
        assert "auth-service" in design["modules"]
        assert "storage-service" in design["modules"]
        assert design["layers"] == ["interface", "services", "data"]

    def test_33_code_optimizer(self):
        source = textwrap.dedent("""
            def f(items, x):
                acc = ""
                for i in items:
                    for j in items:
                        acc += str(i)
                    if i in [1, 2, 3]:
                        pass
                return acc
        """)
        tips = CodeOptimizer().analyze(source)["suggestions"]
        issues = {t["issue"] for t in tips}
        assert "nested loop" in issues
        assert "repeated += accumulation" in issues
        assert "membership test against a literal list" in issues

    def test_34_dependency_risk(self):
        out = DependencyRiskAnalyzer().assess(
            ["requests==2.31.0", "torch", "ab", "# comment"])
        by_name = {d["package"]: d for d in out["dependencies"]}
        assert by_name["requests"]["risk"] == "low"
        assert "unpinned version" in by_name["torch"]["reasons"]
        assert by_name["torch"]["risk"] == "medium"
        assert "ab" in out["high_risk"]

    def test_35_documentation_engine(self, tmp_path):
        path = tmp_path / "mod.py"
        path.write_text('def pub(a, b):\n    """Add two numbers."""\n'
                        '    return a + b\n\ndef _hidden():\n    pass\n',
                        encoding="utf-8")
        doc = DocumentationEngine().document_module(path)
        assert "# mod" in doc
        assert "## pub(a, b)" in doc
        assert "Add two numbers." in doc
        assert "_hidden" not in doc

    def test_36_migration_planner(self):
        out = MigrationIntelligence().plan("python2", "python3")
        assert "replace print statements" in out["steps"]
        generic = MigrationIntelligence().plan("perl", "rust")
        assert len(generic["steps"]) == 4


class TestSecOpsX:
    def test_37_threat_intel_correlation(self):
        out = ThreatIntelCorrelator().correlate([
            {"source": "feed-a", "iocs": ["1.2.3.4", "evil.com"], "severity": "high"},
            {"source": "feed-b", "iocs": ["1.2.3.4"], "severity": "medium"},
            {"source": "feed-c", "iocs": ["5.6.7.8"], "severity": "low"},
        ])
        assert out["total_iocs"] == 3
        assert out["correlated"][0]["ioc"] == "1.2.3.4"
        assert set(out["correlated"][0]["sources"]) == {"feed-a", "feed-b"}
        assert out["correlated"][0]["severity"] == "high"

    def test_38_security_posture(self):
        out = SecurityPostureAnalyzer().assess(
            ["mfa", "backups", "logging", "patching", "least_privilege"])
        assert out["score"] == 0.62
        assert out["grade"] == "moderate"
        assert "encryption_at_rest" in out["missing"]

    def test_39_config_risk_detector(self):
        out = ConfigRiskDetector().scan({
            "debug": True, "ssl_verify": False, "cors": "*",
            "db_password": "hunter2", "port": 8080})
        labels = {f["finding"] for f in out["findings"]}
        assert "TLS verification disabled" in labels
        assert "plaintext credential in config" in labels
        assert "permissive CORS wildcard" in labels
        assert out["risk_score"] >= 8

    def test_40_vulnerability_prioritization(self):
        out = VulnerabilityPrioritizer().prioritize([
            {"id": "V-low", "severity": "low"},
            {"id": "V-crit", "severity": "critical", "exploit_available": True,
             "internet_facing": True, "asset_value": 3},
        ])
        assert out["top"] == "V-crit"
        assert out["ranked"][0]["priority"] > out["ranked"][1]["priority"]

    def test_41_zero_trust_advisor(self):
        out = ZeroTrustAdvisor().evaluate([
            {"subject": "ci", "resource": "repo", "action": "read",
             "mfa": True, "device_check": True, "scope": "repo:ci"},
            {"subject": "intern", "resource": "all", "action": "write",
             "mfa": False, "device_check": False, "scope": "*"},
        ])
        assert out["compliant"] == 1
        assert len(out["violations"][0]["issues"]) == 3
        assert out["recommendations"]

    def test_42_incident_response_planner(self):
        out = IncidentResponsePlanner().plan("ransomware", severity="critical")
        assert out["steps"][0] == "declare incident"
        assert "isolate affected hosts" in out["steps"]
        assert len(IncidentResponsePlanner().plan("unknown-thing")["steps"]) == 5

    def test_43_cloud_security(self):
        out = CloudSecurityAnalyzer().scan([
            {"type": "bucket", "name": "assets", "public": True, "encrypted": True},
            {"type": "db", "name": "main", "public": False, "encrypted": False,
             "logging": False},
        ])
        assert out["verdict"] == "needs-attention"
        assert len(out["findings"]) == 3

    def test_44_supply_chain(self):
        out = SupplyChainAnalyzer().analyze([
            {"name": "requests", "pinned": True, "signed": True,
             "maintainers": 40, "license": "apache-2.0"},
            {"name": "shady-pkg", "pinned": False, "signed": False,
             "maintainers": 1, "license": "unknown"},
        ])
        assert out["clean"] == 1
        assert {"unpinned version", "unsigned artifact",
                "single maintainer (bus factor)",
                "license concern: unknown"} == set(out["flagged"][0]["risks"])


class TestSystemX:
    def test_45_dynamic_resource_allocation(self):
        out = DynamicResourceAllocator().allocate(
            {"train": 3.0, "serve": 1.0}, {"cpu": 8, "gpu": 1, "ram": 16})
        assert out["train"]["cpu"] == pytest.approx(6.0)
        assert out["serve"]["ram"] == pytest.approx(4.0)

    def test_46_priority_scheduler(self):
        sched = PriorityScheduler()
        sched.submit("setup", priority=9, deadline=10)
        sched.submit("deploy", priority=1, depends=("setup",))
        sched.submit("alert", priority=5, deadline=100)
        sched.submit("cleanup", priority=2, deadline=50)
        assert sched.next_task() == "setup"
        sched.complete("setup")
        assert sched.next_task() == "cleanup"
        sched.complete("cleanup")
        assert sched.next_task() == "alert"
        sched.complete("alert")
        assert sched.next_task() == "deploy"

    def test_47_failure_prediction(self):
        out = FailurePredictor().predict(
            {"memory_pct": [40, 60, 85, 98], "error_rate": [0.001, 0.002]},
            limits={"memory_pct": 100, "error_rate": 0.05})
        assert out["risks"]["memory_pct"] > 0.7
        assert out["verdict"] in ("act-now", "watch")
        healthy = FailurePredictor().predict({"cpu": [10, 11]}, limits={"cpu": 100})
        assert healthy["verdict"] == "healthy"

    def test_48_self_healing_approval_gate(self):
        approvals = []
        mgr = SelfHealingManager(approve=lambda a, d: approvals.append(d) or False)
        mgr.register("disk full", ["rotate logs"], safe=True)
        mgr.register("db corrupt", ["restore backup"], safe=False)
        assert mgr.heal("disk full")["healed"] is True
        assert mgr.heal("db corrupt") == {"healed": False,
                                          "reason": "awaiting human approval"}
        assert approvals, "unsafe procedure must have asked for approval"
        assert mgr.heal("unknown")["reason"] == "no procedure registered"

    def test_49_distributed_executor(self):
        out = DistributedExecutor({"big": 4, "small": 1}).assign(
            [{"name": "t1", "cost": 4}, {"name": "t2", "cost": 1},
             {"name": "t3", "cost": 1}])
        assert out["plan"]["small"] == ["t2"]
        assert out["plan"]["big"][0] == "t1"
        assert sorted(out["plan"]["big"] + out["plan"]["small"]) == ["t1", "t2", "t3"]

    def test_50_smart_cache(self):
        cache = SmartCacheManager(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        cache.put("c", 3)
        assert cache.stats()["entries"] == 2
        assert "b" not in cache.stats()["keys"]
        assert cache.get("missing") is None


class TestHumanAI:
    def test_51_personalized_interaction(self):
        eng = PersonalizedInteractionEngine()
        eng.set_profile("dev", expertise="expert")
        assert eng.adapt("dev", "answer")["style"] == "concise-technical"
        assert eng.adapt("newcomer", "answer")["glossary"] is True

    def test_52_intent_prediction(self):
        pred = IntentPredictor()
        for _ in range(3):
            pred.observe("chat", "run_goal")
        pred.observe("chat", "train")
        out = pred.predict("chat")
        assert out["intent"] == "run_goal"
        assert out["confidence"] == pytest.approx(0.75)
        assert pred.predict("unknown")["intent"] is None

    def test_53_dialogue_reference_resolution(self):
        dlg = DialogueManager()
        dlg.push("user", "what is a transformer?", entity="transformer")
        dlg.push("octo", "a transformer is a neural network")
        assert dlg.resolve("why is it useful?") == "why is transformer useful?"
        assert dlg.resolve("what about CNNs?") == "what about CNNs?"
        assert len(dlg.context()) == 2

    def test_54_explainable_decision(self):
        out = ExplainableDecisionEngine().choose(
            ["plan-a", "plan-b"], score_fn=lambda p: 0.9 if p == "plan-a" else 0.4,
            evidence_fn=lambda p: [f"{p} passed benchmarks"])
        assert out["decision"] == "plan-a"
        assert out["alternatives"] == [("plan-b", 0.4)]
        assert out["evidence"] == ["plan-a passed benchmarks"]

    def test_55_collaborative_solver(self):
        solver = CollaborativeProblemSolver(max_rounds=5)

        def propose(problem, rnd):
            return f"attempt-{rnd}"

        def feedback(solution):
            return "close" if solution == "attempt-3" else "no"

        out = solver.solve("p", propose, feedback,
                           lambda s, n: n == "close")
        assert out["solved"] and out["rounds"] == 3
        assert len(out["history"]) == 3

    def test_56_adaptive_teaching(self):
        eng = AdaptiveTeachingEngine()
        eng.record("sam", "rope", False)
        first = eng.teach("sam", "rope", "content")
        assert first["style"] == "step-by-step with analogies"
        for _ in range(3):
            eng.record("sam", "rope", True)
        assert eng.teach("sam", "rope", "content")["style"] == "advanced summary"
        eng.record("sam", "rope", False)
        assert eng.teach("sam", "rope", "c")["mastery"] == pytest.approx(0.7)


class TestFrontier:
    def test_57_architecture_evolution(self):
        engine = CognitiveArchitectureEvolutionEngine(seed=7)
        out = engine.evolve(lambda arch: sum(arch.values()), generations=4,
                            population=6)
        assert set(out["best"]) == {"attention", "memory", "tools", "reflection"}
        assert out["score"] >= 0.0

    def test_58_society_simulator(self):
        sim = ArtificialSocietySimulator({"a": 0.1, "b": 0.9, "c": 0.5}, seed=3)
        before_polar = sim.polarization()
        sim.tick(interactions=30)
        assert sim.polarization() < before_polar
        assert len(sim.history) == 1

    def test_59_emergence_analyzer(self):
        logs = ([{"tick": 1, "agent": a, "action": "wander"} for a in "abcd"]
                + [{"tick": 2, "agent": a, "action": "follow"} for a in "abcd"])
        out = EmergenceAnalyzer().analyze(logs)
        assert out["late_consensus"] == "follow"
        assert out["emergent"] is True
        assert EmergenceAnalyzer().analyze(
            [{"tick": 1, "agent": "a", "action": "x"},
             {"tick": 1, "agent": "b", "action": "y"}])["emergent"] is False

    def test_60_research_planner(self):
        out = AutonomousResearchPlanner().plan(
            "How does RoPE help long context? Does it beat APE?",
            known_facts=["rope is rotary embedding"])
        assert len(out["sub_questions"]) == 2
        assert len(out["experiments"]) == 2
        assert out["steps"][0] == "literature scan"
        assert "replicated" in out["success_criteria"]
