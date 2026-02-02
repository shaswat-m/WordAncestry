# WordAncestry

Interactive etymology explorer that traces a word’s ancestry across languages and plots the lineage on a globe.  
Type a word → see the ancestry chain (right) → click any stage to highlight it on the map (left).

---

## Features

- **Single-word lookup UI** (no history clutter): enter a word and see only the latest result.
- **Etymology chain extraction** from **English Wiktionary** wikitext templates (e.g. `{{inh}}`, `{{der}}`, `{{bor}}`).
- **Map visualization** of language stages using representative geographic anchors (configurable in `app.py`).
- **Clickable stage list**: selecting a stage highlights the corresponding node on the map.

---

## Requirements

- Conda (Miniconda or Anaconda)
- macOS / Linux / Windows (should work anywhere Python runs)

---

## Installation (Conda)

Create and activate the environment:

```bash
conda create -n wordancestry -c conda-forge \
  python=3.12 \
  fastapi \
  uvicorn \
  requests \
  beautifulsoup4 \
  pydantic
conda activate wordancestry
```

---

## Run the App

From the repository root (the directory containing `app.py`):

```bash
uvicorn app:app --reload

```
