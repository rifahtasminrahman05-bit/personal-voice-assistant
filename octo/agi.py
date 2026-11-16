# -*- coding: utf-8 -*-
"""Octo AGI — an agentic layer: goal -> plan -> tools -> report.

Not artificial general intelligence; a tool-using agent loop where Octo LLM,
RAG retrieval and small utilities work together on a stated goal.

Usage:
    python -m octo.agi --goal "how many seconds in 3 days"
    python -m octo.agi --goal "research sql injection and answer how to prevent it"
"""
import argparse
import ast
import datetime
import json
import os
import operator
import re

from .rag import RagIndex, context_for_prompt

MEMORY_FILE = "data/octo/agi_memory.txt"

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_calc(expr):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {expr!r}")
    return _eval(ast.parse(expr, mode="eval"))


def read_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def take_note(text):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    return f"noted: {text.strip()[:80]}"


def tool_search_corpus(query, k=2):
    index = RagIndex.build()
    hits = index.search(query, k=k)
    return "\n".join(f"[{score:.2f}] {chunk[:200]}" for chunk, score in hits) or "no matches"


def tool_research(topic):
    from .research import save_chunks, wiki_topic, slugify
    chunks = wiki_topic(topic, "en")
    if not chunks:
        return f"research failed for: {topic}"
    path, n = save_chunks(chunks, f"agi-{slugify(topic)}")
    return f"collected {n:,} chars about '{topic}' -> {path}"


def tool_websearch(query):
    from .search import search_and_read
    body, sources = search_and_read(query, fetch=2)
    if not body:
        return f"web search returned nothing for: {query}"
    for src in sources:
        take_note(f"web source: {src}")
    return f"read {len(sources)} page(s), {len(body):,} chars. first source: {sources[0] if sources else '-'}"


def tool_hashid(value):
    from .secops import identify_hash
    return "possible hash types: " + ", ".join(identify_hash(value))


def tool_passcheck(password):
    from .secops import password_strength
    score, label, feedback, entropy = password_strength(password)
    out = f"strength {score}/4 ({label}), entropy {entropy:.0f} bits"
    return out + (" | " + "; ".join(feedback) if feedback else "")


def tool_logscan(path):
    from .secops import analyze_log
    return analyze_log(path)


def tool_answer(prompt, steps=120):
    import torch
    from .chat import load_model, complete
    ckpt = "octo_checkpoints/octo.pt"
    if not os.path.exists(ckpt):
        return "Octo is not trained yet; run: python -m octo.train --corpus data/octo"
    model, tokenizer = load_model(ckpt, "cpu")
    context = context_for_prompt(prompt)
    if context:
        prompt = context + "\n\n" + prompt
    return complete(model, tokenizer, prompt, steps, 0.8, 40, "cpu").strip()


def plan(goal):
    """Heuristic planner: pick tool steps for a goal."""
    steps = []
    mathish = re.search(r"[\d]\s*[\+\-\*/%]\s*[\d(]|calculate|কত|হিসাব", goal)
    if mathish:
        expr = re.search(r"[\d\.\s\+\-\*/%\(\)]{3,}", goal)
        if expr:
            steps.append(("calc", expr.group(0).strip()))
    conv = re.search(
        r"(seconds?|minutes?|hours?|days?|weeks?|meters?|kilometers?|km|cm|kb|mb|gb)\s+in\s+"
        r"([\d\.]+)\s+(seconds?|minutes?|hours?|days?|weeks?|meters?|kilometers?|km|cm|kb|mb|gb)",
        goal, re.I)
    if conv:
        steps.append(("convert", conv.group(0)))
    if re.search(r"web search|websearch|ইন্টারনেট|ওয়েব সার্চ|google", goal.lower()):
        topic = re.sub(r".*(web ?search|ইন্টারনেট|ওয়েব সার্চ|google)\s*(for|about|করো)?\s*", "", goal,
                       flags=re.I, count=1).strip(" .?!") or goal
        steps.append(("websearch", topic))
    if re.search(r"research|শেখো|রিসার্চ|study|learn about", goal.lower()):
        topic = re.sub(r".*?(research|শেখো|রিসার্চ|study|learn about)\s*(about|on)?\s*", "", goal,
                       flags=re.I, count=1).strip(" .?!")
        if topic:
            steps.append(("research", topic))
    hashish = re.search(r"\b[0-9a-f]{32,64}\b|hash", goal.lower())
    if hashish:
        m = re.search(r"\b[0-9a-f]{32,64}\b", goal)
        if m:
            steps.append(("hashid", m.group(0)))
    if re.search(r"password (strength|check|audit)|পাসওয়ার্ড চেক|passcheck", goal.lower()):
        m = re.search(r'["\'](.+)["\']', goal)
        if m:
            steps.append(("passcheck", m.group(1)))
    if re.search(r"log ?scan|analyze log|লগ বিশ্লেষণ", goal.lower()):
        m = re.search(r"([\w/\.\-]+\.log)", goal)
        if m:
            steps.append(("logscan", m.group(1)))
    steps.append(("search", goal))
    steps.append(("answer", goal))
    return steps


UNITS = {
    "second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0, "week": 604800.0,
    "meter": 1.0, "kilometer": 1000.0, "km": 1000.0, "cm": 0.01,
    "kb": 1.0, "mb": 1024.0, "gb": 1024.0 ** 2,
}


def tool_convert(arg):
    m = re.search(r"([\w]+)\s+in\s+([\d\.]+)\s+([\w]+)", arg, re.I)
    if not m:
        return f"could not parse conversion: {arg}"
    target, amount, source = m.group(1).lower(), float(m.group(2)), m.group(3).lower()
    target, source = target.rstrip("s"), source.rstrip("s")
    if target not in UNITS or source not in UNITS:
        return f"unknown units in: {arg}"
    value = amount * UNITS[source] / UNITS[target]
    return f"{amount:g} {source}(s) = {value:g} {target}(s)"


TOOLS = {
    "calc": lambda arg: f"{arg} = {safe_calc(arg)}",
    "convert": tool_convert,
    "websearch": tool_websearch,
    "hashid": tool_hashid,
    "passcheck": tool_passcheck,
    "logscan": tool_logscan,
    "research": tool_research,
    "search": tool_search_corpus,
    "answer": tool_answer,
    "now": lambda _: datetime.datetime.now().isoformat(timespec="seconds"),
    "note": take_note,
    "memory": lambda _: read_memory()[-500:] or "(empty)",
}


def run_agent(goal, max_steps=6, verbose=True):
    trace = []
    memory = read_memory()
    if memory:
        trace.append(("memory", memory.strip().splitlines()[-1]))
    for name, arg in plan(goal)[:max_steps]:
        if verbose:
            print(f"[agi] {name}: {arg[:80]}")
        try:
            result = TOOLS[name](arg)
        except Exception as e:
            result = f"tool '{name}' failed: {e}"
        trace.append((name, str(result)))
        if name in ("research", "answer"):
            take_note(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] goal='{goal}' {name}: {str(result)[:120]}")
    return trace


def main():
    ap = argparse.ArgumentParser(description="Octo AGI: goal-driven tool-using agent.")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    print(f"goal: {args.goal}\n")
    trace = run_agent(args.goal, max_steps=args.max_steps)
    print("\n=== report ===")
    for name, result in trace:
        print(f"[{name}] {result[:300]}")
    print("\n(json)")
    print(json.dumps([{"tool": n, "output": r[:200]} for n, r in trace], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
