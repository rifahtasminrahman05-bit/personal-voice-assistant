# -*- coding: utf-8 -*-
"""Internet & Research Intelligence (features 110, 113, 115-118).

Website analysis, fact checking, news monitoring, document analysis,
report generation and source verification. All operate on text the
user supplies locally — Octo never fetches anything by itself.
"""
import re
from collections import Counter


class WebsiteAnalyzer:
    """110. Website Analysis: structure, readability, link inventory."""

    def analyze(self, html):
        text = re.sub(r"<[^>]+>", " ", html or "")
        words = [w for w in text.split() if w]
        links = re.findall(r'href=["\']([^"\']+)["\']', html or "")
        headings = re.findall(r"<h([1-6])[^>]*>(.*?)</h\1>",
                              html or "", re.I | re.S)
        return {"words": len(words),
                "links": len(links),
                "external_links": [l for l in links if l.startswith("http")],
                "headings": [{"level": int(lvl), "text": t.strip()[:80]}
                             for lvl, t in headings],
                "readability": self._readability(words)}

    @staticmethod
    def _readability(words):
        if not words:
            return 0.0
        long_words = sum(1 for w in words if len(w) > 6)
        return round(long_words / len(words), 3)


class FactChecker:
    """113. Fact Checking: cross-check claims against local evidence."""

    HEDGES = ("maybe", "possibly", "allegedly", "rumor", "unverified")

    def check(self, claim, evidence):
        """evidence: list of local text passages."""
        claim_terms = self._terms(claim)
        support, contradict = 0, 0
        for passage in evidence:
            terms = self._terms(passage)
            overlap = claim_terms & terms
            if len(overlap) >= max(1, len(claim_terms) // 3):
                if any(neg in passage.lower()
                       for neg in ("not ", "never ", "no ", "false")):
                    contradict += 1
                else:
                    support += 1
        hedged = any(h in claim.lower() for h in self.HEDGES)
        verdict = "unsupported"
        if contradict and support:
            verdict = "disputed"
        elif support and not contradict:
            verdict = "supported"
        elif hedged:
            verdict = "hedged"
        return {"claim": claim[:160], "verdict": verdict,
                "support": support, "contradict": contradict}

    @staticmethod
    def _terms(text):
        return {w for w in re.findall(r"[a-z\u0980-\u09ff]{3,}",
                                      (text or "").lower())}


class NewsMonitor:
    """115. News Monitoring: topic tracking over local article texts."""

    def __init__(self):
        self.articles = []

    def add(self, title, text, source="local"):
        article = {"title": title, "source": source,
                   "text": text, "added_seq": len(self.articles)}
        self.articles.append(article)
        return article

    def by_topic(self, topic):
        topic = topic.lower()
        return [a for a in self.articles
                if topic in a["title"].lower()
                or topic in a["text"].lower()]

    def trending(self, top=5):
        counter = Counter()
        for article in self.articles:
            for word in re.findall(r"[a-z\u0980-\u09ff]{5,}",
                                   article["text"].lower()):
                counter[word] += 1
        return counter.most_common(top)


class DocumentAnalyzer:
    """116. Document Analysis: sections, keywords, entities."""

    def analyze(self, text):
        paragraphs = [p for p in (text or "").split("\n\n") if p.strip()]
        words = re.findall(r"[A-Za-z\u0980-\u09ff]+", text or "")
        keywords = Counter(w.lower() for w in words if len(w) > 5)
        emails = set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text or ""))
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text or "")
        return {"paragraphs": len(paragraphs),
                "words": len(words),
                "keywords": keywords.most_common(8),
                "emails": sorted(emails),
                "numeric_facts": numbers[:20]}


class ReportGenerator:
    """117. Report Generation: structured local report from findings."""

    def generate(self, title, sections):
        lines = [f"# {title}", ""]
        for name, content in sections:
            lines.append(f"## {name}")
            lines.append(str(content).strip())
            lines.append("")
        report = "\n".join(lines)
        return {"title": title, "report": report,
                "sections": len(sections), "chars": len(report)}


class SourceVerifier:
    """118. Source Verification: provenance scoring for local sources."""

    TRUSTED_LOCAL_MARKS = ("peer-reviewed", "official", "primary",
                           "verified", "local-cache")

    def verify(self, source):
        text = (source.get("note") or "").lower()
        marks = [m for m in self.TRUSTED_LOCAL_MARKS if m in text]
        has_author = bool(source.get("author"))
        has_date = bool(source.get("date"))
        score = 0.35 * bool(marks) + 0.25 * has_author + 0.25 * has_date
        score += 0.15 * (source.get("corroborations", 0) > 0)
        return {"source": source.get("name", "?"),
                "score": round(score, 2),
                "trust": "high" if score >= 0.6 else
                         "medium" if score >= 0.35 else "low",
                "marks": marks}
