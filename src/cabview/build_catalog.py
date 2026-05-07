"""Génère un catalogue HTML des icônes de signaux téléchargées.

Usage:
    PYTHONPATH=. python src/cabview/build_catalog.py
"""

from __future__ import annotations

from pathlib import Path

ROOTS = {
    "France (SNCF)": Path("data/signals_ref/fr_diagrams"),
    "Suisse (CFF)": Path("data/signals_ref/ch_diagrams"),
}

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Catalogue de signaux ferroviaires</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 24px; background: #fafafa; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }}
  .item {{ background: white; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px;
           text-align: center; }}
  .item img {{ max-width: 100%; max-height: 100px; object-fit: contain; }}
  .item .name {{ font-size: 0.7rem; color: #555; margin-top: 6px; word-break: break-word; }}
  .count {{ color: #888; font-weight: normal; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Catalogue de signaux ferroviaires</h1>
<p>Source : Wikimedia Commons. Utiliser comme référence visuelle pour l'annotation.</p>
{sections}
</body>
</html>
"""


def render_section(title: str, root: Path) -> str:
    if not root.exists():
        return ""
    files = sorted(p for p in root.glob("*.svg"))
    items = []
    for p in files:
        rel = p.relative_to(Path("data/signals_ref"))
        label = p.stem.replace("_", " ")
        items.append(
            f'<div class="item"><img src="{rel}" loading="lazy">'
            f'<div class="name">{label}</div></div>'
        )
    return (
        f'<h2>{title} <span class="count">({len(files)} signaux)</span></h2>'
        f'<div class="grid">{"".join(items)}</div>'
    )


def main():
    sections = "\n".join(render_section(t, r) for t, r in ROOTS.items())
    out = Path("data/signals_ref/catalog.html")
    out.write_text(HTML.format(sections=sections))
    print(f"Catalogue généré : {out}")
    print(f"  Ouvrir avec : open {out}")


if __name__ == "__main__":
    main()
