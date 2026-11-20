"""Laden der eingefrorenen Studie (`kob_vormessung.json`, erzeugt von `tools/kob_make_vormessung.py`) und reine Zugriffsfunktionen.

Alles Abgestimmte (Wald, Boosting, Netz, Untergrenze, Lernkurve) ist eingefroren; die App rechnet es nicht zur Laufzeit. Die Datei enthält Fehler je Zufalls-Seed, aus denen Mittel,
Standardfehler und Verdikt in `kob_evaluation.py` berechnet werden.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parent / "kob_vormessung.json"

# Gegner-Einstellung -> Schlüssel der Fehler von Wald und Boosting in der Datei
TREE_KEYS = {
    "Standard": ("rf_std", "gb_std"),
    "abgestimmt": ("rf_t", "gb_t"),
    "am Test gewählt (Untergrenze)": ("rf_orakel", "gb_orakel"),
}
REIHEN = {"reihe1": range(0, 20), "frisch": range(20, 40)}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def haupt_world(world: str) -> dict:
    return load()["haupt"][world]


def reihe_index(world: str, reihe: str = "reihe1") -> list[int]:
    """Positionen der Zufalls-Seeds der Reihe (reihe1 = 0 bis 19, frisch = 20 bis 39) in den Feldern der Welt."""
    seeds = haupt_world(world)["seeds"]
    return [i for i, s in enumerate(seeds) if s in REIHEN[reihe]]


def haupt_array(world: str, key: str, reihe: str = "reihe1") -> np.ndarray:
    idx = reihe_index(world, reihe)
    return np.array([haupt_world(world)[key][i] for i in idx], float)


def best_tree(world: str, setting: str, reihe: str = "reihe1") -> np.ndarray:
    """Fehler des Baums je Zufalls-Seed: der bessere von Wald und Boosting, je Zufalls-Seed am Test gewählt (optimistisch für die Bäume)."""
    k1, k2 = TREE_KEYS[setting]
    return np.minimum(haupt_array(world, k1, reihe), haupt_array(world, k2, reihe))


def has_reihe(world: str, reihe: str) -> bool:
    return bool(reihe_index(world, reihe))


def worlds_with_study() -> list[str]:
    return list(load()["haupt"].keys())


def teil(name: str) -> dict:
    return load()[name]
