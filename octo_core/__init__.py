# -*- coding: utf-8 -*-
"""Octo Core — Next-Generation Advanced Intelligence Architecture.

Modular layers (spec Phase 1-7):
    cognitive      reasoning, planning, reflection, decision, introspection
    agents         orchestrator, task decomposition, verification, HITL
    memory         knowledge graph, hybrid RAG, episodic/semantic, provenance
    models         RoPE + GQA + KV-cache transformer, model router
    security       prompt-injection defense, permissions, sandbox, audit
    learning       skills, feedback, regression detection, controlled self-improvement
    evaluation     benchmarks, hallucination detection, regression runner
    infrastructure scheduler, cache, fault recovery, resource manager
    multimodal     vision/OCR/audio adapters (optional deps, graceful fallback)

Advanced intelligence extensions (60-feature set):
    cogx           cognitive load, adaptive/causal/counterfactual reasoning
    swarm          agent collaboration, negotiation, reputation, workflows
    memx           associative memory, consolidation, decay, domain transfer
    evolvex        continual learning, replay, curriculum, adaptation loop
    perception     scene understanding, tracking, spatial/gesture/audio
    devx           debugging, architecture, optimization, docs, migration
    secopsx        threat intel, posture, zero-trust, IR, cloud/supply chain
    sysint         resource allocation, scheduling, self-healing, smart cache
    humanai        personalization, intent prediction, explainability, teaching
    frontier       architecture evolution, society sim, emergence, research plan
    selftrain      autonomous self-training loop (improvement-gated)

Components Deep Dive (cognitive agent architecture):
    reasoning      chain/tree-of-thoughts, reflection, calibration,
                   analogical + abductive reasoning
    worldmodel     entities, typed/weighted relations, states, temporal
                   model, causal graph, spatial awareness, query engine
    planner        goals, decomposition, sequential/parallel plans,
                   execution monitoring, multi-criteria decisions,
                   contingency + long-horizon planning
    identity       personality, emotional model, relationship memory,
                   consistency, persona modulation, ethics engine,
                   communication/clarification layer
    actions        tool registry, permission-checked action engine,
                   guarded desktop automation, external tools, feedback
    agentstate     agi_state.json persistence, agent.save(), versioned
                   snapshots + rollback, encrypted backups

Usage:
    python -m octo_core --goal "research sql injection and answer how to prevent it"
"""
__version__ = "1.2.0"
