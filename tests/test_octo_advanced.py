# -*- coding: utf-8 -*-
import os
import textwrap

import pytest

from octo_core.benchhistory import BenchmarkHistory
from octo_core.cybersec import AttackSurfaceAnalyzer, IncidentTimeline
from octo_core.digital import DigitalTwin, EmbodiedInterface, LoopbackRobot
from octo_core.knowledge import KnowledgeGapDetector, OntologyLearner
from octo_core.metacog import CognitiveArchitectureSimulator
from octo_core.modelint import Ensemble, TestTimeCompute, similarity
from octo_core.predictive import FutureStateSimulator, PredictiveEngine
from octo_core.scientific import (DiscoveryLab, ScientificHypothesisEngine,
                                  extract_variables)
from octo_core.software import CodebaseUnderstanding, TestGenerator


class TestPredictive:
    def test_forecast_rising_series(self):
        out = PredictiveEngine.forecast([1, 2, 3, 4, 5], horizon=2)
        assert out["direction"] == "rising"
        assert out["confidence"] == 1.0
        assert out["forecast"][0] == pytest.approx(6, abs=0.1)
        assert out["forecast"][1] == pytest.approx(7, abs=0.1)

    def test_forecast_short_series(self):
        out = PredictiveEngine.forecast([5])
        assert out["forecast"] == [5]
        assert out["confidence"] == 0.0
        assert out["direction"] == "unknown"

    def test_forecast_noisy_series_lower_confidence(self):
        clean = PredictiveEngine.forecast([1, 2, 3, 4, 5])
        noisy = PredictiveEngine.forecast([1, 5, 1, 6, 1])
        assert noisy["confidence"] < clean["confidence"]

    def test_predict_outcome_from_history(self, tmp_path):
        hist = BenchmarkHistory(path=str(tmp_path / "bh.jsonl"))
        for score in (0.5, 0.6, 0.7):
            hist.record({"math": {"score": score}})
        out = PredictiveEngine().predict_outcome(hist.entries())
        assert out["direction"] == "rising"
        assert out["forecast"][0] > 0.7

    def test_simulator_ranks_low_risk_high_gain_first(self):
        sim = FutureStateSimulator(seed=1, rounds=100)
        result = sim.simulate(
            {"budget": 10},
            [{"name": "safe", "cost": 0.01, "gain": 0.5, "risk": 0.05},
             {"name": "wild", "cost": 0.5, "gain": 0.5, "risk": 0.9}])
        assert result["recommended"] == "safe"
        assert result["futures"][0]["risk_of_loss"] < result["futures"][1]["risk_of_loss"]
        assert all(set(f) >= {"decision", "expected_value", "p10", "p90",
                              "risk_of_loss"} for f in result["futures"])


class TestScientific:
    def test_pearson_perfect_positive(self):
        from octo_core.scientific import _pearson
        assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_generate_ranks_correlations(self):
        obs = [{"temperature": t, "pressure": 2 * t, "noise": (t * 7) % 3}
               for t in range(1, 11)]
        hyps = ScientificHypothesisEngine().generate(obs)
        assert hyps, "expected at least one hypothesis"
        assert hyps[0]["variables"] == ("temperature", "pressure")
        assert hyps[0]["strength"] == "strong"

    def test_generate_needs_three_observations(self):
        assert ScientificHypothesisEngine().generate([{"a": 1, "b": 2}]) == []

    def test_test_hypothesis_supported_and_refuted(self):
        engine = ScientificHypothesisEngine()
        hyp = {"hypothesis": "a increases with b", "variables": ("a", "b"), "r": 0.9}
        good = [{"a": x, "b": 3 * x} for x in range(1, 8)]
        bad = [{"a": x, "b": -3 * x} for x in range(1, 8)]
        assert engine.test(hyp, good)["verdict"] == "supported"
        assert engine.test(hyp, bad)["verdict"] == "refuted"

    def test_discovery_lab_cycle(self):
        def runner(experiment):
            return {"ok": "increase" in experiment["hypothesis"],
                    "value": 0.9, "log": "measured"}
        lab = DiscoveryLab(runner)
        report = lab.run_cycle(["x increases with y", "unrelated guess"],
                               max_rounds=5)
        assert report["supported"] == 1
        assert len(report["findings"]) == 2
        assert lab.experiment_report()["experiments"] == 2

    def test_extract_variables(self):
        assert extract_variables("accuracy=0.9, loss: 0.3") == \
            {"accuracy": "0.9", "loss": "0.3"}
        assert extract_variables("no numbers here") == {}


class TestDigital:
    def test_twin_sync_and_drift(self):
        twin = DigitalTwin("plant")
        twin.sync({"temp": 20, "rpm": 100})
        assert twin.drift({"temp": 20, "rpm": 100})["in_sync"]
        report = twin.drift({"temp": 25, "rpm": 100, "load": 3})
        assert not report["in_sync"]
        assert report["keys"] == 2

    def test_twin_project_with_dynamics(self):
        twin = DigitalTwin("battery")
        twin.sync({"charge": 100})
        projected = twin.project(3, lambda s: {"charge": s["charge"] - 10})
        assert [p["charge"] for p in projected] == [90, 80, 70]
        assert twin.state["charge"] == 100

    def test_robot_moves_and_clamps(self):
        robot = LoopbackRobot(size=3)
        robot.act("up")
        robot.act("left")
        assert (robot.x, robot.y) == (0, 1)
        for _ in range(10):
            robot.act("left")
        assert robot.x == 0
        assert robot.battery < 100

    def test_robot_charge(self):
        robot = LoopbackRobot()
        for _ in range(5):
            robot.act("down")
        robot.act("charge")
        assert robot.battery == 100.0

    def test_embodied_interface(self):
        iface = EmbodiedInterface()
        iface.register("bot", LoopbackRobot())
        assert iface.connect("bot")["battery"] == 100.0
        obs = iface.act_and_observe("bot", "right")
        assert obs["x"] == 1
        assert iface.sense_all()["bot"]["x"] == 1
        with pytest.raises(KeyError):
            iface.connect("missing")


class TestKnowledge:
    def _graph(self):
        from octo_core.memory import KnowledgeGraph
        g = KnowledgeGraph()
        g.add_fact("octo", "is", "model")
        g.add_fact("octo", "used_for", "research")
        g.add_fact("rust", "is", "language")
        g.add_fact("dns", "is", "protocol", source="book")
        g.add_fact("dns", "is", "protocol2", source="web")
        return g

    def test_gap_detection(self):
        report = KnowledgeGapDetector(self._graph(), min_facts=2).gaps()
        subjects = {s["subject"] for s in report["sparse"]}
        assert "rust" in subjects
        assert report["contradictions"], "conflicting dns fact expected"
        assert report["total_facts"] == 5
        assert 0.0 <= report["coverage"] <= 1.0

    def test_gap_questions(self):
        qs = KnowledgeGapDetector(self._graph()).questions()
        assert qs and all(isinstance(q, str) for q in qs)

    def test_ontology_learning(self):
        learner = OntologyLearner()
        added = learner.learn(
            "A transformer is a neural network. The encoder is part of the "
            "transformer. A tokenizer is used for splitting text.")
        assert added >= 3
        assert "transformer" in learner.ancestors("transformer") or \
            "neural network" in learner.ancestors("transformer")
        assert learner.summary()["is_a_links"] >= 1
        assert learner.summary()["part_of_links"] >= 1
        assert learner.summary()["used_for_links"] >= 1

    def test_ontology_hierarchy_renders(self):
        learner = OntologyLearner()
        learner.learn("A dog is an animal. A cat is an animal.")
        lines = learner.hierarchy()
        assert any("animal" in line for line in lines)


class TestSoftware:
    def test_codebase_understanding_on_repo(self, tmp_path):
        (tmp_path / "app.py").write_text(
            "import os\nimport helper\n\nclass App:\n    pass\n\n"
            "def main():\n    pass\n", encoding="utf-8")
        (tmp_path / "helper.py").write_text("X = 1\n", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "junk.py").write_text("bad", encoding="utf-8")
        engine = CodebaseUnderstanding(str(tmp_path))
        mods = engine.analyze()
        assert set(mods) == {"app", "helper"}
        assert mods["app"]["classes"] == ["App"]
        assert mods["app"]["local_deps"] == ["helper"]
        report = engine.report()
        assert report["modules"] == 2
        assert report["most_depended"][0][0] == "helper"
        assert "app" in report["entry_candidates"]
        assert engine.dependencies("app") == ["helper"]

    def test_public_functions_skips_private(self, tmp_path):
        path = tmp_path / "mod.py"
        path.write_text("def pub(a, b=2):\n    return a\n\n"
                        "def _hidden():\n    pass\n", encoding="utf-8")
        funcs = TestGenerator(str(tmp_path)).public_functions(str(path))
        assert [f["name"] for f in funcs] == ["pub"]
        assert funcs[0]["n_defaults"] == 1

    def test_generated_tests_are_valid_and_runnable(self, tmp_path):
        (tmp_path / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8")
        gen = TestGenerator(str(tmp_path))
        out = gen.generate_file(str(tmp_path / "calc.py"),
                                out_dir=str(tmp_path / "generated"))
        assert os.path.exists(out)
        code = open(out, encoding="utf-8").read()
        assert "from calc import add" in code
        assert "test_edge_none" in code
        compile(code, out, "exec")

    def test_generate_for_function_missing(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text("def a():\n    pass\n", encoding="utf-8")
        with pytest.raises(ValueError):
            TestGenerator(str(tmp_path)).generate_for_function(str(path), "nope", "m")


class TestCybersec:
    def test_attack_surface_finds_planted_patterns(self, tmp_path):
        (tmp_path / "risky.py").write_text(textwrap.dedent("""
            import os
            password = "hunter2hunter2"
            result = eval(user_input)
            os.system(cmd)
            requests.get(url, verify=False)
        """), encoding="utf-8")
        report = AttackSurfaceAnalyzer(str(tmp_path)).scan()
        labels = {f["finding"] for f in report["findings"]}
        assert "eval() call" in labels
        assert "hardcoded credential" in labels
        assert "os.system shell call" in labels
        assert "TLS verification disabled" in labels
        assert report["counts"]["high"] >= 3
        assert report["risk_score"] > 5

    def test_attack_surface_clean_tree(self, tmp_path):
        (tmp_path / "clean.py").write_text("x = 1 + 2\n", encoding="utf-8")
        report = AttackSurfaceAnalyzer(str(tmp_path)).scan()
        assert report["findings"] == []
        assert report["verdict"] == "acceptable"

    def test_incident_timeline_reconstruction(self):
        logs = [
            "2024-05-01 10:00:00 INFO port scan detected from 10.0.0.9",
            "2024-05-01 10:05:00 WARN login failed for admin (brute force)",
            "2024-05-01 10:10:00 ERROR sudo privilege escalation attempt",
            "2024-05-01 10:15:00 INFO large outbound transfer to unknown host",
            "2024-05-01 10:20:00 WARN audit log clear requested",
        ]
        report = IncidentTimeline().reconstruct(logs)
        assert report["events"] == 5
        assert report["summary"] == ("recon -> compromise -> escalation -> "
                                     "exfiltration -> cleanup")
        assert report["attack_sequence"][0]["phase"] == "recon"

    def test_incident_timeline_out_of_order_input(self):
        logs = [
            "2024-05-01 10:15:00 INFO outbound export of db dump",
            "2024-05-01 10:00:00 INFO nmap probe observed",
        ]
        report = IncidentTimeline().reconstruct(logs)
        phases = [p["phase"] for p in report["phase_events"]]
        assert phases == ["recon", "exfiltration"]

    def test_incident_timeline_no_indicators(self):
        report = IncidentTimeline().reconstruct(
            ["2024-05-01 09:00:00 INFO daily backup finished"])
        assert report["summary"] == "no attack indicators"


class TestModelIntelligence:
    def test_difficulty_routing(self):
        def gen(prompt, steps):
            return f"ans@{steps}"
        ttc = TestTimeCompute(gen)
        assert ttc.difficulty("hi there") == "easy"
        assert ttc.difficulty("prove that this algorithm is optimal and explain why") == "hard"

    def test_easy_single_sample_hard_multi_sample(self):
        calls = []

        def gen(prompt, steps):
            calls.append(steps)
            return "the same stable answer text"
        ttc = TestTimeCompute(gen, samples_hard=4)
        easy = ttc.solve("what is 2+2?")
        assert easy["samples"] == 1 and easy["agreement"] == 1.0
        hard = ttc.solve("prove why this architecture design is optimal")
        assert hard["samples"] == 4
        assert hard["agreement"] == 1.0
        assert hard["answer"] == "the same stable answer text"

    def test_vote_picks_majority_cluster(self):
        consensus, agreement = TestTimeCompute._vote(
            ["alpha beta gamma", "alpha beta gamma", "totally different words"])
        assert consensus == "alpha beta gamma"
        assert agreement == pytest.approx(2 / 3)

    def test_similarity(self):
        assert similarity("alpha beta gamma", "alpha beta gamma") == 1.0
        assert similarity("alpha beta gamma", "delta epsilon zeta") == 0.0
        assert similarity("", "anything") == 0.0

    def test_ensemble_consensus(self):
        ens = Ensemble([("a", lambda p: "use parameterized queries"),
                        ("b", lambda p: "use parameterized queries"),
                        ("c", lambda p: "shuffle the database rows")])
        out = ens.answer("how to prevent sql injection?")
        assert out["answer"] == "use parameterized queries"
        assert out["consensus"] is True
        assert out["agreement"] == 0.67

    def test_ensemble_handles_member_errors(self):
        def broken(prompt):
            raise RuntimeError("boom")
        ens = Ensemble([("good", lambda p: "safe answer text"),
                        ("bad", broken)])
        out = ens.answer("q")
        assert "error" in out["outputs"]["bad"]
        assert out["answer"] == "safe answer text"

    def test_weighted_answer_prefers_heavy_member(self):
        ens = Ensemble([("light", lambda p: "answer one shared words here"),
                        ("heavy", lambda p: "answer two totally distinct words"),
                        ("mid", lambda p: "answer one shared words here")])
        out = ens.weighted_answer("q", {"light": 1.0, "heavy": 5.0, "mid": 1.0})
        assert out["picked_by"] == "weight"
        assert out["answer"] == "answer two totally distinct words"


class TestMetaCognitive:
    TASKS = [
        {"name": "chat", "needs": {"attention": 0.8, "memory": 0.4}},
        {"name": "act", "needs": {"tools": 0.9, "reflection": 0.3}},
    ]

    def test_simulate_scores_and_success(self):
        sim = CognitiveArchitectureSimulator()
        out = sim.simulate({"attention": 1.0, "memory": 0.5, "tools": 1.0,
                            "reflection": 0.5}, self.TASKS)
        assert 0 <= out["avg_score"]
        assert out["success_rate"] == 1.0
        assert len(out["results"]) == 2

    def test_simulate_missing_capability_fails_task(self):
        sim = CognitiveArchitectureSimulator()
        out = sim.simulate({"attention": 1.0}, [{"name": "act", "needs": {"tools": 0.9}}])
        assert out["results"][0]["success"] is False

    def test_compare_ranks(self):
        sim = CognitiveArchitectureSimulator()
        out = sim.compare({"full": {"attention": 1, "memory": 1, "tools": 1,
                                    "reflection": 1},
                           "minimal": {"attention": 0.5}}, self.TASKS)
        assert out["best"] == "full"
        assert out["ranking"][0]["avg_score"] >= out["ranking"][1]["avg_score"]

    def test_optimize_finds_best_configuration(self):
        sim = CognitiveArchitectureSimulator()
        out = sim.optimize(self.TASKS, levels=(0.0, 0.5, 1.0))
        best = out["best_architecture"]
        assert set(best) == {"attention", "memory", "tools", "reflection"}
        assert best["tools"] >= 0.5
        assert out["score"] >= 0.0


class TestLabSession:
    def session(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        (corpus / "notes.txt").write_text(
            "octo is a modular intelligence platform with predictive, scientific "
            "and cybersecurity capabilities built in.", encoding="utf-8")
        from octo_core.desktop import OctoSession
        return OctoSession(roots=(str(corpus),),
                           checkpoint="/nonexistent/octo.pt")

    def test_predict_trend_empty_history(self, tmp_path):
        s = self.session(tmp_path)
        from octo_core.benchhistory import BenchmarkHistory
        s.history = BenchmarkHistory(path=str(tmp_path / "bh.jsonl"))
        out = s.predict_trend()
        assert out["confidence"] == 0.0
        assert out["direction"] == "unknown"

    def test_predict_trend_uses_recorded_history(self, tmp_path):
        s = self.session(tmp_path)
        from octo_core.benchhistory import BenchmarkHistory
        s.history = BenchmarkHistory(path=str(tmp_path / "bh.jsonl"))
        for score in (0.4, 0.5, 0.6):
            s.history.record({"math": {"score": score}})
        out = s.predict_trend()
        assert out["direction"] == "rising"

    def test_simulate_future(self, tmp_path):
        out = self.session(tmp_path).simulate_future(
            [{"name": "safe", "cost": 0.01, "gain": 0.5, "risk": 0.05},
             {"name": "wild", "cost": 0.5, "gain": 0.5, "risk": 0.9}])
        assert out["recommended"] == "safe"

    def test_knowledge_gaps_and_questions(self, tmp_path):
        s = self.session(tmp_path)
        s.memory.remember("octo", "is", "assistant")
        report = s.knowledge_gaps()
        assert report["total_facts"] >= 1
        assert isinstance(report["questions"], list)

    def test_attack_surface_audits(self, tmp_path):
        s = self.session(tmp_path)
        target = tmp_path / "scanme"
        target.mkdir()
        (target / "risky.py").write_text("eval(x)\n", encoding="utf-8")
        report = s.attack_surface(str(target))
        assert report["findings"]
        assert s.audit.tail(1)[0]["event"] == "attack_surface_scan"

    def test_optimize_architecture(self, tmp_path):
        out = self.session(tmp_path).optimize_architecture()
        assert set(out["best_architecture"]) == {"attention", "memory",
                                                 "tools", "reflection"}
        assert out["score"] >= 0.0
