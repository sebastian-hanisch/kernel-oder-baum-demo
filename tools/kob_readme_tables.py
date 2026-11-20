"""Gibt die Markdown-Tabellen der README aus der eingefrorenen Studie aus (dieselben Funktionen wie die App; tests/test_claims.py prüft, dass jede Tabelle wörtlich in der README steht).

Aufruf: python tools/kob_readme_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kob_evaluation as E  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for title, text in (("Hauptreihe, erste Zufalls-Seeds", E.table_main("reihe1")), ("Hauptreihe, frische Zufalls-Seeds", E.table_main("frisch")),
                    ("Differenzen, erste Zufalls-Seeds", E.table_differences("reihe1")), ("Differenzen, frische Zufalls-Seeds", E.table_differences("frisch")),
                    ("Abstimmung", E.table_tuning()), ("Live gegen Studie", E.table_live()), ("Netz", E.table_netz()), ("Ringe", E.table_ringe()), ("Lernkurve", E.table_lernkurve())):
    print(f"### {title}\n\n{text}\n")
