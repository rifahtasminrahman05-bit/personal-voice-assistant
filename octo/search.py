# -*- coding: utf-8 -*-
"""Octo web search — Bing search + page fetching for AGI research.

Usage:
    python -m octo.search --query "sql injection prevention" --fetch 1
"""
import argparse
import base64
import html as html_mod
import re
import urllib.request
import urllib.parse

from .research import clean_text, fetch_url, save_chunks, slugify

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
H2_BLOCK = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
ANCHOR = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
TAG = re.compile(r"<[^>]+>")


def _decode_bing_url(href):
    href = html_mod.unescape(href)
    params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    u = params.get("u", [""])[0]
    if u.startswith("a1"):
        b64 = u[2:]
        b64 += "=" * (-len(b64) % 4)
        try:
            return base64.urlsafe_b64decode(b64).decode("utf-8", errors="replace")
        except Exception:
            return href
    return href


def search_bing(query, k=5):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "count": k})
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    results = []
    for block in H2_BLOCK.findall(html):
        m = ANCHOR.search(block)
        if not m:
            continue
        href, title = m.group(1), m.group(2)
        target = _decode_bing_url(href)
        if not target.startswith("http") or "bing.com/ck" in target:
            continue
        results.append({"title": TAG.sub("", title).strip(), "url": target})
        if len(results) >= k:
            break
    return results


def search_and_read(query, k=5, fetch=1, save=True):
    """Search the web, optionally read the top pages. Returns (text, sources)."""
    hits = search_bing(query, k=k)
    if not hits:
        return "", []
    parts, sources = [], []
    for hit in hits[:fetch]:
        try:
            text = clean_text(fetch_url(hit["url"]))
            if len(text) > 200:
                parts.append(f"{hit['title']}\n{text[:4000]}")
                sources.append(hit["url"])
        except Exception as e:
            parts.append(f"{hit['title']} (fetch failed: {e})")
    body = "\n\n".join(parts)
    if save and body:
        save_chunks([body], f"web-{slugify(query)}")
    return body, sources


def main():
    ap = argparse.ArgumentParser(description="Bing search for Octo research.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fetch", type=int, default=0, help="also read top N pages")
    args = ap.parse_args()

    hits = search_bing(args.query, k=args.k)
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']}\n   {hit['url']}")
    if args.fetch:
        body, sources = search_and_read(args.query, k=args.k, fetch=args.fetch, save=True)
        print(f"\nread {len(sources)} page(s), {len(body):,} chars saved to data/octo/research/")


if __name__ == "__main__":
    main()
