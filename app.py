from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

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
    "Anglo-Norman": (49.2, -0.37),  # Normandy-ish
    "Norman": (49.2, -0.37),    # Old Occitan / Provence-ish (Avignon)
    "Old Occitan (Old Provençal)": (43.9493, 4.8055),    # Old Italian (Florence as a reasonable representative)
    "Old Italian": (43.7696, 11.2558),    # Arabic (choose a representative; Baghdad is a decent classical hub)
    "Arabic": (33.3152, 44.3661),   # Classical Persian (Shiraz as a reasonable Persian cultural hub)
    "Classical Persian": (29.5918, 52.5837),  # Dravidian family (South India centroid-ish: Bengaluru)
    "Dravidian (Proto-/family)": (12.9716, 77.5946),
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
    "Medieval Latin": (41.9028, 12.4964),
    "Early Spanish": (40.4168, -3.7038),
}

LANG_CODE_TO_NAME = {
    "xno": "Anglo-Norman",
    "nrf": "Norman",
    "la-med": "Medieval Latin",
    "es-ear": "Early Spanish",
    "pro": "Old Occitan (Old Provençal)",
    "roa-oit": "Old Italian",
    "ar": "Arabic",
    "fa-cls": "Classical Persian",
    "dra": "Dravidian (Proto-/family)",
    "enm": "Middle English",
    "ang": "Old English",
    "en": "English",
    "fr": "French",
    "fro": "Old French",
    "frm": "Middle French",
    "la": "Latin",
    "LL.": "Late Latin",
    "grc": "Ancient Greek",
    "el": "Greek",
    "de": "German",
    "gmh": "Middle High German",
    "goh": "Old High German",
    "gem-pro": "Proto-Germanic",
    "ine-pro": "Proto-Indo-European",
    "sa": "Sanskrit",
    "hi": "Hindi",
    "ja": "Japanese",
    "zh": "Chinese",
}

def _lang_name(code_or_name: str) -> str:
    # Wiktionary templates often use language codes; sometimes names appear directly.
    c = code_or_name.strip()
    return LANG_CODE_TO_NAME.get(c, c)
def parse_primary_etymology_path_wikitext(word: str, wikitext: str) -> List[Stage]:
    """
    Parse the first English Etymology block and extract an ordered list of source steps.

    Key behavior:
    - Wiktionary often encodes etymology as "From {{inh|en|enm|...}}, from {{der|en|xno|...}}, ..."
      i.e. all templates have to_lang=en, not a to_lang chain.
    - Therefore we read templates IN ORDER and append each (from_lang, term) as the next stage.
    """
    if not wikitext:
        return [Stage(language="English", word=word)]

    # 1) Isolate English section if present
    m_en = re.search(r"^==\s*English\s*==\s*$", wikitext, flags=re.MULTILINE)
    if m_en:
        en_start = m_en.end()
        m_next = re.search(r"^==[^=].*==\s*$", wikitext[en_start:], flags=re.MULTILINE)
        en_text = wikitext[en_start: en_start + (m_next.start() if m_next else len(wikitext))]
    else:
        en_text = wikitext

    # 2) Extract first Etymology subsection if present
    m_et = re.search(r"^===\s*Etymology(?:\s*\d+)?\s*===\s*$", en_text, flags=re.MULTILINE)
    if m_et:
        et_start = m_et.end()
        m_next_h3 = re.search(r"^===.*===\s*$", en_text[et_start:], flags=re.MULTILINE)
        et_text = en_text[et_start: et_start + (m_next_h3.start() if m_next_h3 else len(en_text))]
    else:
        et_text = en_text

    # Start with the queried word as English
    stages: List[Stage] = [Stage(language="English", word=word)]

    # 3) Ordered scan of common etymology templates in the etymology text
    # We preserve appearance order using finditer.
    tpl_pat = re.compile(r"\{\{\s*(inh|bor|der|lbor)\s*\|([^}]*)\}\}", flags=re.IGNORECASE)

    for m in tpl_pat.finditer(et_text):
        kind = m.group(1).lower()
        args = m.group(2)

        parts = [p.strip() for p in args.split("|")]
        # Typical formats:
        #   inh|en|enm|salarie
        #   der|en|la|salārium||wages
        # We want:
        #   from_lang = parts[1] (index 2 overall after split)?? Actually parts include to at [0]
        # parts[0]=to_lang, parts[1]=from_lang, parts[2]=term (often)
        if len(parts) < 2:
            continue

        to_lang = parts[0] if len(parts) >= 1 else None
        from_lang = parts[1] if len(parts) >= 2 else None
        term = parts[2] if len(parts) >= 3 and parts[2] else None

        # Only follow when the template is describing the English word (to_lang == en),
        # otherwise ignore to reduce noise.
        if to_lang != "en":
            continue

        if not from_lang:
            continue

        from_lang_name = _lang_name(from_lang)
        from_word = term or f"(unknown {from_lang_name} form)"
        stages.append(Stage(language=from_lang_name, word=from_word))

    # Optional: also support the {{etyl|fro|enm}} + {{m|fro|word}} pattern
    # (kept from your previous version; appended if present)
    etyl_pat = re.compile(r"\{\{\s*etyl\s*\|([^|}]+)\|([^|}]+)\s*\}\}", re.IGNORECASE)
    m_pat = re.compile(r"\{\{\s*m\s*\|([^|}]+)\|([^|}]+)", re.IGNORECASE)

    etyls = etyl_pat.findall(et_text)
    ms = m_pat.findall(et_text)

    for from_lang, to_lang in etyls:
        if to_lang != "en":
            continue
        for m_lang, m_word in ms:
            if m_lang == from_lang and m_word:
                stages.append(Stage(language=_lang_name(from_lang), word=m_word))

    # Deduplicate consecutive
    out: List[Stage] = []
    for s in stages:
        if not out or (out[-1].language != s.language or out[-1].word != s.word):
            out.append(s)

    return out



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

def _wikitext_to_plain(s: str) -> str:
    """
    Very lightweight wikitext cleanup:
    - replace [[A|B]] -> B and [[A]] -> A
    - strip italics/bold quotes ''
    - strip templates {{...}} (replace with space)
    """
    # [[A|B]] -> B, [[A]] -> A
    s = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)

    # remove templates entirely
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)

    # strip wiki italics/bold markers
    s = s.replace("''", "")

    # collapse whitespace
    s = " ".join(s.split())
    return s


def fetch_wiktionary_html(word: str) -> str:
    url = WIKTIONARY_BASE + requests.utils.quote(word)
    r = requests.get(url, timeout=20, headers={"User-Agent": "WordAncestry/0.1"})
    r.raise_for_status()
    return r.text

def fetch_wiktionary_wikitext(word: str) -> str:
    """
    Fetches page wikitext via MediaWiki API (more reliable than scraping HTML).
    """
    api = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": word,
        "redirects": 1,
    }
    r = requests.get(api, params=params, timeout=20, headers={"User-Agent": "WordAncestry/0.1"})
    r.raise_for_status()
    data = r.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    revs = page.get("revisions", [])
    if not revs:
        return ""
    return revs[0].get("slots", {}).get("main", {}).get("*", "") or ""


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
        "debug": {"n_stages": len(stages)}
    }


@app.get("/api/etymology")
def api_etymology(word: str = Query(..., min_length=1, max_length=80)) -> JSONResponse:
    w = word.strip()
    #html = fetch_wiktionary_html(w)
    #stages = parse_primary_etymology_path(w, html)
    wt = fetch_wiktionary_wikitext(w)
    stages = parse_primary_etymology_path_wikitext(w, wt)
    graph = build_graph(w, stages)
    return JSONResponse(graph)

@app.get("/api/debug_etymology_text")
def debug_etymology_text(word: str) -> JSONResponse:
    w = word.strip()
    wt = fetch_wiktionary_wikitext(w)
    if not wt:
        return JSONResponse({"error": "no wikitext"})

    # isolate English section
    m_en = re.search(r"^==\s*English\s*==\s*$", wt, flags=re.MULTILINE)
    if m_en:
        en_start = m_en.end()
        m_next = re.search(r"^==[^=].*==\s*$", wt[en_start:], flags=re.MULTILINE)
        en_text = wt[en_start: en_start + (m_next.start() if m_next else len(wt))]
    else:
        en_text = wt

    # isolate first Etymology block
    m_et = re.search(r"^===\s*Etymology(?:\s*\d+)?\s*===\s*$", en_text, flags=re.MULTILINE)
    if m_et:
        et_start = m_et.end()
        m_next_h3 = re.search(r"^===.*===\s*$", en_text[et_start:], flags=re.MULTILINE)
        et_text = en_text[et_start: et_start + (m_next_h3.start() if m_next_h3 else len(en_text))]
    else:
        et_text = en_text

    return JSONResponse({
        "raw_etymology_wikitext": et_text[:4000],
        "plain_etymology_text": _wikitext_to_plain(et_text)[:4000],
    })

# Serve a simple frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

