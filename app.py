from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="WordAncestry")

# ----------------------------
# Option A: language -> representative coordinate
# ----------------------------
# Keep this small at first; expand over time.
LANG_COORDS: Dict[str, Tuple[float, float]] = {
    "English": (51.5074, -0.1278),         # London
    "Middle English": (52.3555, -1.1743),  # England (approx)
    "Old English": (52.5, -1.5),           # England (approx)
    "French": (48.8566, 2.3522),           # Paris
    "Old French": (48.8566, 2.3522),       # Paris
    "Middle French": (48.8566, 2.3522),    # Paris
    "Latin": (41.9028, 12.4964),           # Rome
    "Late Latin": (41.9028, 12.4964),      # Rome
    "Greek": (37.9838, 23.7275),           # Athens
    "Ancient Greek": (37.9838, 23.7275),   # Athens
    "German": (52.5200, 13.4050),          # Berlin
    "Old High German": (48.1351, 11.5820), # Munich-ish
    "Proto-Germanic": (56.0, 10.0),        # Denmark-ish
    "Proto-Indo-European": (49.0, 36.0),   # steppe-ish (approx)
    "Sanskrit": (25.3176, 82.9739),        # Varanasi-ish
    "Hindi": (28.6139, 77.2090),           # Delhi
    "Japanese": (35.6895, 139.6917),       # Tokyo
    "Chinese": (39.9042, 116.4074),        # Beijing
}

# If a language isn't in the dict, we can fall back to (0,0) or skip plotting.
DEFAULT_COORD = (0.0, 0.0)

WIKTIONARY_BASE = "https://en.wiktionary.org/wiki/"


@dataclass
class Stage:
    language: str
    word: str
    gloss: Optional[str] = None
    approx_year: Optional[int] = None


def _slugify_language(lang: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", lang.strip().lower()).strip("_")


def _make_node_id(query_word: str, language: str, stage_word: str) -> str:
    return f"{query_word}|{_slugify_language(language)}|{stage_word}"


def _lang_coord(lang: str) -> Tuple[float, float]:
    return LANG_COORDS.get(lang, DEFAULT_COORD)


def fetch_wiktionary_html(word: str) -> str:
    url = WIKTIONARY_BASE + requests.utils.quote(word)
    r = requests.get(url, timeout=20, headers={"User-Agent": "WordAncestry/0.1"})
    r.raise_for_status()
    return r.text


def parse_primary_etymology_path(word: str, html: str) -> List[Stage]:
    """
    Pragmatic v1 parser:
    - find the first "Etymology" section on the page
    - extract a linear-ish path by looking for language names + linked terms in that paragraph/list
    This will not be perfect for all words; it’s a good v1 that you can iterate.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the first headline with id containing "Etymology"
    # Wiktionary uses <span class="mw-headline" id="Etymology"> etc
    ety_headline = soup.find("span", class_="mw-headline", string=re.compile(r"^Etymology", re.I))
    if not ety_headline:
        return [Stage(language="Unknown", word=word)]

    # The etymology content is typically in following siblings until next heading of same/higher level.
    # We'll grab the first paragraph or list after the heading.
    heading = ety_headline.parent  # usually an h3/h4
    cursor = heading.find_next_sibling()
    chunks = []
    for _ in range(10):
        if cursor is None:
            break
        if cursor.name and cursor.name.startswith("h"):
            break
        if cursor.name in ("p", "ol", "ul"):
            chunks.append(cursor.get_text(" ", strip=True))
            # p + maybe list
            if cursor.name == "p":
                # take just first meaningful paragraph for v1
                break
        cursor = cursor.find_next_sibling()

    text = " ".join(chunks).strip()
    if not text:
        return [Stage(language="Unknown", word=word)]

    # Very rough extraction heuristics:
    # Recognize patterns like "From Old French salaire, from Latin salarium, ..."
    # We'll pull language-like tokens and the following word token.
    # You will refine this over time; keep it stable for now.

    # Common language labels in Wiktionary prose
    known_langs = sorted(LANG_COORDS.keys(), key=len, reverse=True)
    lang_regex = "|".join(re.escape(l) for l in known_langs)

    # capture (Language) (word-ish token)
    # word tokens often appear as italicized in HTML, but in text they're just words;
    # we match a following token that looks like a word/phrase.
    pattern = re.compile(rf"\b({lang_regex})\b\s+([A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)", re.UNICODE)
    matches = pattern.findall(text)

    stages: List[Stage] = []
    # Always start with the queried word in English-ish context
    stages.append(Stage(language="English", word=word))

    for lang, w in matches:
        stages.append(Stage(language=lang, word=w))

    # De-duplicate consecutive identical stages
    deduped: List[Stage] = []
    for s in stages:
        if not deduped or (deduped[-1].language != s.language or deduped[-1].word != s.word):
            deduped.append(s)

    return deduped


def build_graph(query_word: str, stages: List[Stage]) -> Dict[str, Any]:
    nodes = []
    edges = []
    primary_path = []

    for s in stages:
        nid = _make_node_id(query_word, s.language, s.word)
        lat, lon = _lang_coord(s.language)

        nodes.append(
            {
                "id": nid,
                "word": s.word,
                "language": s.language,
                "time": {"approx_year": s.approx_year},
                "gloss": s.gloss,
                "location": {"lat": lat, "lon": lon},
            }
        )
        primary_path.append(nid)

    for a, b in zip(primary_path[:-1], primary_path[1:]):
        edges.append(
            {
                "source": a,
                "target": b,
                "type": "derived_from",
            }
        )

    return {
        "query": query_word,
        "nodes": nodes,
        "edges": edges,
        "primary_path": primary_path,
    }


@app.get("/api/etymology")
def api_etymology(word: str = Query(..., min_length=1, max_length=80)) -> JSONResponse:
    w = word.strip()
    html = fetch_wiktionary_html(w)
    stages = parse_primary_etymology_path(w, html)
    graph = build_graph(w, stages)
    return JSONResponse(graph)


# Serve a simple frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

