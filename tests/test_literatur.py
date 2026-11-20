"""Literaturwerte (Stufe W, nur aus gelesenen Quellen) gegen die Randsätze der App und die README.

* Fernández-Delgado u. a. 2014 (JMLR 15, 3133 bis 3181; `QUELLEN_KLASSISCH.md` 1.16a, gelesen: Volltext mit Abschnitten 2 bis 4 und Tabelle 5): nur Zahlen aus Fließtext und Tabelle 5.
  **Voraussetzung: gepaarter t-Test über Datensätze, ohne Korrektur für mehrfaches Testen.** Kennzahl PMA = mittlere Prozentzahl der je Datensatz höchsten Genauigkeit, **nicht** die Genauigkeit.
  Kein Test mit p-Werten aus Abbildung 4; kein Satz „zweitbeste“; keine Aussage zu Merkmalszahl, Drehung oder Lieferungen (steht nicht in der Quelle).
* McElfresh u. a. 2023 (NeurIPS Datasets and Benchmarks, Tabelle 2): auf den 57 Datensätzen mit höchstens 1 250 Zeilen mittlerer Rang SVM 9,54 und TabPFN (Version 1) 4,88 von 19 Verfahren.
* Wainberg, Alipanahi, Frey 2016 (JMLR 17): Aussage ohne Zahl (die Studie belegt ihre Rangfolge nicht; Hyperparameter auf einer Teilmenge der Testdaten gewählt).
* TabPFN-Leistungsangaben: kein Test (nur Herstellerangabe, nicht gemessen).
"""

import re
from pathlib import Path

import kob_decision as DEC

ROOT = Path(__file__).resolve().parent.parent
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
RAND = " ".join(" ".join(DEC.RANDSAETZE).split())


def test_delgado_zahlen_im_randsatz_mit_kennzahl_und_voraussetzung():
    assert "Wald 94,1, Gauß-SVM 92,3" in RAND and "mittlere Prozentzahl der je Datensatz höchsten Genauigkeit, nicht die Genauigkeit" in RAND
    assert "Mittlere Genauigkeit 82,3 (`rf_t`), 82,0 (`parRF_t`) und 81,8 (`svm_C`)" in RAND
    assert "mittlere Friedman-Ränge 32,9 (`parRF_t`), 33,1 (`rf_t`) und 36,8 (`svm_C`)" in RAND
    assert "besser in 56, schlechter in 55 und gleich in 10 von 121 Datensätzen" in RAND
    assert "gepaarten t-Test über Datensätze, ohne Korrektur für mehrfaches Testen" in RAND and "nicht signifikant" in RAND
    assert "JMLR 15, 3133 bis 3181" in RAND


def test_delgado_ohne_p_werte_ohne_zweitbeste_ohne_aussage_zu_merkmalszahl():
    for text in (RAND, README):
        assert "zweitbeste" not in text.replace("kein Satz „zweitbeste“", "").replace("kein Satz „zweitbeste“.", "")
        assert "p-Wert" not in text.replace("keine p-Werte aus Abbildung 4", "")
        assert not re.search(r"p\s*[<=]\s*0[,.]\d", text)                       # keine p-Werte als Zahl


def test_delgado_zahlen_in_der_readme_nur_als_randsatz_mit_kennzahl():
    assert "Fernández-Delgado u. a. 2014" in README and "Kennzahl mittlere Prozentzahl der je Datensatz höchsten Genauigkeit, nicht die Genauigkeit" in README
    assert "keine p-Werte aus Abbildung 4" in README


def test_mcelfresh_tabelle_2_und_tabpfn_nur_als_quelle():
    assert "mittlerer Rang 4,88 gegen 9,54 für die SVM auf 57 Datensätzen mit höchstens 1 250 Zeilen" in RAND
    assert "nach Herstellerangabe" in RAND and "hier nicht gemessen, nicht in der App und nicht in den Tests" in RAND
    for tabpfn_zahl in ("TabPFN 9", "TabPFN schlägt", "Genauigkeit von TabPFN"):
        assert tabpfn_zahl not in RAND and tabpfn_zahl not in README


def test_wainberg_aussage_ohne_zahl_und_grinsztajn_ohne_svm():
    assert "belegt ihre Rangfolge nicht" in RAND and "Teilmenge der Testdaten" in RAND
    assert "kein SVM und kein Gauß-Prozess im Vergleich" in RAND and "arXiv 2207.08815" in RAND
