# -*- coding: utf-8 -*-
"""Octo research — collect fresh text from the web so Octo can learn new topics.

Usage:
    python -m octo.research --topic "sql injection prevention" --lang en
    python -m octo.research --url https://docs.python.org/3/tutorial/index.html
    python -m octo.research --topic "firewall" --retrain 1000
"""
import argparse
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

RESEARCH_DIR = "data/octo/research"
USER_AGENT = "OctoResearchBot/0.1 (offline assistant; educational)"


class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)


def fetch_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if "<html" not in raw.lower():
        return raw
    parser = TextExtractor()
    parser.feed(raw)
    return " ".join(parser.parts)


def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def wiki_topic(topic, lang="en"):
    api = f"https://{lang}.wikipedia.org/w/api.php"
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "list": "search",
        "srsearch": topic, "srlimit": 3,
    })
    req = urllib.request.Request(f"{api}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        hits = [h["title"] for h in __import__("json").load(resp)["query"]["search"]]

    chunks = []
    for title in hits:
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "titles": title,
            "prop": "extracts", "explaintext": 1, "redirects": 1,
        })
        req = urllib.request.Request(f"{api}?{params}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            pages = __import__("json").load(resp)["query"]["pages"]
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    chunks.append(f"{title}\n{extract}")
    return chunks


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "source"


def save_chunks(chunks, source_label):
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    path = os.path.join(RESEARCH_DIR, f"{slugify(source_label)}.txt")
    body = "\n\n".join(chunks)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path, len(body)


def main():
    ap = argparse.ArgumentParser(description="Octo research: collect web text for training.")
    ap.add_argument("--topic", default=None, help="Wikipedia topic to research")
    ap.add_argument("--url", action="append", default=[], help="URL to fetch (repeatable)")
    ap.add_argument("--lang", default="en", help="Wikipedia language (en, bn, ...)")
    ap.add_argument("--retrain", type=int, default=0, metavar="STEPS",
                    help="after collecting, retrain Octo for STEPS steps on all corpora")
    args = ap.parse_args()

    chunks = []
    label = []
    if args.topic:
        print(f"researching topic: {args.topic} ({args.lang}) ...")
        chunks += wiki_topic(args.topic, args.lang)
        label.append(args.topic)
    for url in args.url:
        print(f"fetching: {url} ...")
        chunks.append(clean_text(fetch_url(url)))
        label.append(slugify(url))

    if not chunks:
        ap.error("nothing to research: pass --topic and/or --url")

    path, n_chars = save_chunks(chunks, "-".join(label))
    print(f"saved {n_chars:,} chars from {len(chunks)} source(s) -> {path}")

    if args.retrain:
        from subprocess import run
        run(["python", "-m", "octo.train", "--corpus", "data/octo",
             "--steps", str(args.retrain)], check=True)


if __name__ == "__main__":
    main()
