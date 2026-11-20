"""Entscheidungsbaum „Welches Verfahren passt zu welcher Aufgabe?“ ohne neue Rechnung: jede Antwort nennt ihre Zahl aus der eingefrorenen Studie und gilt für die gemessene Welt.

Reine Funktionen: `empfehlung(antworten)` gibt eine Liste von (Aussage, Beleg) zurück; die Zahlen stammen aus `kob_evaluation` (Studie, Band über 20 Zufalls-Seeds). Die Antwort ist eine Aussage
über die gemessene Welt (erzeugt, 300 bis 400 Trainingszeilen, diese Gitter), nicht über die Daten der Nutzerin oder des Nutzers.
"""

from __future__ import annotations

import kob_evaluation as E

JA, NEIN, WEISS_NICHT = "Ja", "Nein", "Weiß nicht"
ANTWORTEN = (JA, NEIN, WEISS_NICHT)

FRAGEN = {
    "gerade": "Liegt die Wahrheit nahe an einer Geraden (Klassifikation, zwei Klassen)?",
    "schwellen": "Hängt die Klasse an Schwellen einzelner Merkmale (Regeln der Form „Merkmal über Wert“)?",
    "einheiten": "Haben die Merkmale verschiedene Einheiten und sind nicht auf eine gemeinsame Skala gebracht?",
    "rauschen": "Sind viele der Merkmale für die Klasse irrelevant (wenige Zeilen, viele Spalten)?",
    "n_gross": "Ist die Zeilenzahl sehr groß (mehr als einige zehntausend)?",
    "wahrscheinlichkeit": "Brauchen Sie Wahrscheinlichkeiten und nicht nur Klassen?",
}
FRAGEN_REIHENFOLGE = tuple(FRAGEN)

# Verweise auf die anderen Demos der Reihe (Ordner); nur Verweise, ohne eigene Zahlen
STUECKE = (
    ("kerntrick-perzeptron-demo", "Kerntrick im Perzeptron: das Skalarprodukt durch einen Kern ersetzen"),
    ("harte-marge-demo", "Harte Marge: die Marge heißt 2/‖w‖"),
    ("weiche-marge-demo", "Weiche Marge: C regelt Fehler gegen Marge"),
    ("kernwahl-demo", "Kernwahl: die Median-Heuristik ist eine Heuristik, kein Optimum"),
    ("kernel-ridge-demo", "Kernel Ridge ist als Gleichung das Mittel eines Gauß-Prozesses mit λ als Rauschvarianz"),
    ("gram-kondition-demo", "Bandbreite und Kondition: λ nie null, die Gram-Matrix prüfen"),
    ("gram-naeherung-demo", "Gram-Näherung (Nyström-Verfahren, Zufallsmerkmale): bei großem n; die nötige Zahl m hängt stark an der Bandbreite"),
)


def _wins_text(world: str) -> str:
    k, n, g, lin = E.compare_linear(world)
    return f"Welt {world}: Gauß-SVM {E.pct(g)}, lineare SVM {E.pct(lin)}, Gauß besser in {E.k_of(k, n)} Zufalls-Seeds"


def _cmp_text(world: str) -> str:
    c = E.compare_study(world, "abgestimmt")
    return f"Welt {world}: Gauß-SVM {E.pct(c.kern)}, bester abgestimmter Baum {E.pct(c.baum)}, Verdikt {c.verdict} (Gauß besser in {E.k_of(c.wins, c.n)} Zufalls-Seeds)"


def empfehlung(antworten: dict[str, str]) -> list[tuple[str, str]]:
    """Aussagen zu den Antworten (Frage-Schlüssel -> 'Ja' / 'Nein' / 'Weiß nicht'); immer mit Beleg und dem Hinweis, dass die Antwort für die gemessene Welt gilt."""
    out: list[tuple[str, str]] = []
    a = {k: antworten.get(k, WEISS_NICHT) for k in FRAGEN}
    if a["gerade"] == JA:
        out.append(("In den Welten A und D bringt ein Kern gegenüber der linearen SVM in der Klassifikation keinen Vorteil.",
                    f"{_wins_text('A')}; {_wins_text('D')}."))
    if a["einheiten"] == JA:
        s = E.scaling_summary()
        out.append(("Zuerst skalieren: der Gauß-Kern sieht nur Abstände, und eine Spalte in großer Einheit bestimmt sie allein.",
                    f"Lieferungen roh {E.pct(s['roh'], 2)} gegen standardisiert {E.pct(s['std'], 2)} (Gauß-SVM mit Median-Heuristik, {s['n']} Zufalls-Seeds, roh schlechter in {E.k_of(s['roh_schlechter'], s['n'])}); "
                    "Bäume sind gegen die Einheiten unempfindlich (Schwellen)."))
    if a["schwellen"] == JA:
        out.append(("In den gemessenen Welten liegen Bäume vorn, wo die Wahrheit achsenparallel gebaut ist; der Kern sieht nur Abstände und verliert die Achsen.",
                    f"{_cmp_text('E')}; {_cmp_text('F')}. Dieselbe Aufgabe gedreht: {_cmp_text('Er')}."))
    if a["rauschen"] == JA:
        out.append(("In den gemessenen Welten mit vielen irrelevanten Merkmalen und wenigen Zeilen liegen die Bäume vorn (in C2 nur durch die Wahl des besseren Baums am Test, alle Verfahren nahe 50 %); der Anteil des Abstands aus den k relevanten von p Merkmalen ist k/p.",
                    f"{_cmp_text('C')}; {_cmp_text('C2')}; {_cmp_text('F8')}."))
    if a["n_gross"] == JA:
        out.append(("Bei sehr großem n ist der exakte Kern zu teuer (Rechenzeit wächst mit n³, Speicher mit n²): eine Näherung der Gram-Matrix oder ein Baum.",
                    "Zahlen und Verfahren in der Demo gram-naeherung-demo; die nötige Zahl m hängt stark an der Bandbreite; ob und wie sie mit n wächst, ist dort nur an einem Datensatz für wenige n gemessen."))
    if a["wahrscheinlichkeit"] == JA:
        out.append(("Die SVM liefert Klassen, keine Wahrscheinlichkeiten; Platt-Skalierung oder eine andere Kalibrierung wäre ein eigener Schritt, hier nicht gemessen.",
                    "Verweis: kalibrierung-schwelle-demo."))
    if not any(a[k] == JA for k in FRAGEN) or (a["gerade"] == NEIN and a["schwellen"] == NEIN and a["rauschen"] == NEIN):
        out.append(("Kein Merkmal der gemessenen Welten, das eine Seite begünstigt: Kern und Baum beide abstimmen und gegeneinander messen.",
                    f"Glatte Aufgaben liegen beim Kern vorn ({_cmp_text('P')}); bei Wechselwirkung und Container hängt es an der Reihe ({_cmp_text('B')}; {_cmp_text('D')})."))
    out.append(("Jede Antwort gilt für die gemessene Welt (erzeugt, 300 bis 400 Trainingszeilen, diese Gitter), nicht für Ihre Daten.", "Hinweis zu allen Zeilen darüber."))
    return out


# Randsätze mit Quelle (nur Literatur, nicht gemessen); die Zahlen prüft tests/test_literatur.py gegen die gelesenen Quellen
RANDSAETZE = (
    "**TabPFN** (vortrainiertes Tabellenmodell) ist nach Herstellerangabe heute ein sehr starker Gegner bei kleinen Tabellen; hier nicht gemessen, nicht in der App und nicht in den Tests "
    "(unabhängig gemessen ist nur die Version 1 in TabZilla, McElfresh u. a. 2023: mittlerer Rang 4,88 gegen 9,54 für die SVM auf 57 Datensätzen mit höchstens 1 250 Zeilen).",
    "**Wainberg, Alipanahi und Frey 2016** (JMLR 17): die Studie mit 179 Klassifikatoren belegt ihre Rangfolge nicht, die Hyperparameter wurden auf einer Teilmenge der Testdaten gewählt.",
    "**Fernández-Delgado u. a. 2014** (JMLR 15, 3133 bis 3181; Kennzahl: mittlere Prozentzahl der je Datensatz höchsten Genauigkeit, nicht die Genauigkeit): Wald 94,1, Gauß-SVM 92,3; im gepaarten t-Test "
    "über Datensätze, ohne Korrektur für mehrfaches Testen, ist der Wald gegen die Gauß-SVM besser in 56, schlechter in 55 und gleich in 10 von 121 Datensätzen, nicht signifikant. "
    "Mittlere Genauigkeit 82,3 (`rf_t`), 82,0 (`parRF_t`) und 81,8 (`svm_C`); mittlere Friedman-Ränge 32,9 (`parRF_t`), 33,1 (`rf_t`) und 36,8 (`svm_C`).",
    "**Grinsztajn, Oyallon und Varoquaux 2022** (arXiv 2207.08815, Abschnitte 5.3 und 5.4): auf Tabellen mit Netzen gegen Bäume schaden uninformative Merkmale den Netzen stärker, und eine zufällige Drehung der Merkmale kehrt die Rangfolge Netz gegen Baum um; "  # lint-ok: dezimalpunkt
    "kein SVM und kein Gauß-Prozess im Vergleich. Dass dasselbe für den Gauß-Kern gilt, ist hier gemessen (Experiment 3), nicht die Folgerung der Quelle.",
)
