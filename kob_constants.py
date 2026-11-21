"""Konstanten: Regler-Grenzen, Welten der Live-Instanz, Presets, Farben, Verdikt-Schwelle.

Die Presets heißen nach der **Welt**, nie nach einem Verdikt: das Live-Modell (Kernel-Ridge-Klassifikator gegen eine einzelne CART) gibt die Verdikte der Studie
(scikit-learn-Gauß-SVM gegen abgestimmten Wald und abgestimmtes Boosting) nur teilweise wieder.
"""

import kob_worlds as W

# --- Live-Instanz ----------------------------------------------------------------------------------------------------------------------
LIVE_WORLDS = ("P", "E", "Er", "B", "D", "F", "F8")          # Welten der Live-Instanz und der Studie (Preset je Welt)
WORLD_SELECT_LABELS = {w: W.WORLD_LABELS[w] for w in LIVE_WORLDS}
DEFAULT_WORLD = "P"
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 600, 50, 300           # Zeilen der Live-Instanz (Obergrenze 600: Cholesky und CART unter etwa 2 s)
SEED_MAX, DEFAULT_SEED = 999999, 7
N_TEST_LIVE = 4000                                            # frische Testpunkte der Live-Instanz (wie in der Studie)
GEGNER = ("Standard", "abgestimmt", "am Test gewählt (Untergrenze)")
DEFAULT_GEGNER = "abgestimmt"
SCALINGS = ("roh", "standardisiert", "[0, 1]")
DEFAULT_SCALING = "standardisiert"
ANGLE_MIN, ANGLE_MAX, ANGLE_STEP, DEFAULT_ANGLE = 0, 45, 5, 0   # Drehwinkel der 2D-Schwellenregel in Grad
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3                  # Rauschmerkmale der 2D-Welt in Experiment 4 (live)

# --- Verdikt-Regel (eine Festlegung für die ganze Demo) --------------------------------------------------------------------------------
SE_FACTOR = 2.0       # Mittel von d (Gauß minus Baum je Zufalls-Seed) unter -2 Standardfehler: Kern vorn; über +2: Baum vorn; sonst gleichauf
KERN_VORN, BAUM_VORN, GLEICHAUF = "Kern vorn", "Baum vorn", "gleichauf"

# --- Studie (eingefroren) --------------------------------------------------------------------------------------------------------------
N_SEEDS = 20          # Zufalls-Seeds je Reihe
N_TEST_STUDY = 4000   # Testpunkte je Zufalls-Seed
STUDY_WORLDS = ("A", "B", "C", "C2", "E", "D", "F", "P", "F8", "Er", "Br")
# Aufwand der Abstimmung je Welt und Zufalls-Seed in angepassten Modellen (ohne Endanpassung): Gitter mal 5 Falten
FITS_LINEAR = 4 * 5
FITS_GAUSS = 5 * 4 * 5
FITS_FOREST = 6 * 5
FITS_BOOST = 6 * 5

COLORS = {"kern": "#2F6B65", "baum": "#c0602a", "netz": "#7a5ea8", "linear": "#8a8a8a", "bayes": "#111111", "wald": "#c0602a", "boost": "#d9a441",
          "klasse0": "#4c78a8", "klasse1": "#e45756"}

PRESETS = {
    "Kreis mit Rauschen": {"welt": "P", "n": 300, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": DEFAULT_SCALING, "winkel": 0, "rauschen": DEFAULT_NOISE},
    "Schwellenregeln": {"welt": "E", "n": 300, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": DEFAULT_SCALING, "winkel": 0, "rauschen": DEFAULT_NOISE},
    "Schwellenregeln gedreht": {"welt": "Er", "n": 300, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": DEFAULT_SCALING, "winkel": 45, "rauschen": DEFAULT_NOISE},
    "Wechselwirkung": {"welt": "B", "n": 300, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": DEFAULT_SCALING, "winkel": 0, "rauschen": DEFAULT_NOISE},
    "Container": {"welt": "D", "n": 400, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": DEFAULT_SCALING, "winkel": 0, "rauschen": DEFAULT_NOISE},
    "Lieferungen ohne Skalierung": {"welt": "F", "n": 400, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": "roh", "winkel": 0, "rauschen": DEFAULT_NOISE},
    "Viele Rauschmerkmale": {"welt": "F8", "n": 400, "seed": DEFAULT_SEED, "gegner": DEFAULT_GEGNER, "skalierung": DEFAULT_SCALING, "winkel": 0, "rauschen": 8},
}
PRESET_HELP = {
    "Kreis mit Rauschen": "Liefergebiet: Kreis mit halber Fläche des Quadrats, jedes Label mit Wahrscheinlichkeit 0,1 vertauscht (Bayes-Fehler genau 10 %), 300 Zeilen, zwei Merkmale.",
    "Schwellenregeln": "Welt E: Klasse hängt an Schwellen einzelner Merkmale (achsenparallel), 10 Merkmale davon 7 irrelevant, 300 Zeilen, Bayes-Fehler genau 10 %.",
    "Schwellenregeln gedreht": "Dieselbe Aufgabe wie Welt E, aber die 10 Merkmale mit einer zufälligen Drehung gemischt (Welt Er); der Drehwinkel-Regler gilt für die 2D-Fassung in Experiment 3.",
    "Wechselwirkung": "Welt B: die Klasse hängt am Produkt zweier Merkmale, 4 Merkmale, 300 Zeilen; Bayes-Fehler rund 24 %.",
    "Container": "Container-Erzeuger nach der Beschreibung aus verweilzeit-demo: 20 Spalten (fast nur Kategorien), Klasse = Verweilzeit über dem Median, 400 Zeilen.",
    "Lieferungen ohne Skalierung": "Lieferungen (zu spät) in natürlichen Einheiten: Ladegewicht in Kilogramm neben Werten zwischen 0 und 1; der Kern rechnet auf den rohen Spalten, 400 Zeilen.",
    "Viele Rauschmerkmale": "Lieferungen mit 8 statt 3 Rauschmerkmalen ohne Bezug zum Ziel (16 Merkmale), 400 Zeilen.",
}

# 2D-Welten für Entscheidungsgrenzen und Drehung (live)
BOUNDARY_GRID = 60
