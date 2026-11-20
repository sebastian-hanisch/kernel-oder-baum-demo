"""Verdikte als reine Funktionen (`kob_evaluation.verdict_*`): Schwellen und jede Zahl im Text an handgesetzten Werten (DEMO-PLAYBOOK §12)."""

import re

import numpy as np
import pytest

import kob_constants as C
import kob_evaluation as E


# ---------------------------------------------------------------- Formate
def test_dezimalkomma_prozente_und_prozentpunkte():
    assert E.num(1.5, 1) == "1,5"
    assert E.num(3.0, 2) == "3,00"
    assert E.pct(0.2246) == "22,5 %" and E.pct(0.2246, 2) == "22,46 %" and E.pct(0.1) == "10,0 %"
    assert E.pp(0.106) == "+10,6 Prozentpunkte" and E.pp(-0.042) == "−4,2 Prozentpunkte" and E.pp(0.5, signed=False) == "50,0 Prozentpunkte"
    assert E.k_of(8, 20) == "8 von 20"


# ---------------------------------------------------------------- Regel an der Schwelle
def _diffs_mit_mittel_und_se(m, se, n=20):
    """Differenzen mit genau dem Mittel m und dem Standardfehler se: zentrierte Reihe, auf die Streuung skaliert."""
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)
    return m + z * se * np.sqrt(n)


def test_regel_an_der_schwelle_zwei_standardfehler_sind_noch_gleichauf():
    for m, want in ((-2.01, C.KERN_VORN), (-1.99, C.GLEICHAUF), (1.99, C.GLEICHAUF), (2.01, C.BAUM_VORN)):
        d = _diffs_mit_mittel_und_se(m * 0.01, 0.01)
        assert E.mean_se(d)[0] == pytest.approx(m * 0.01, abs=1e-12) and E.mean_se(d)[1] == pytest.approx(0.01, abs=1e-12)
        assert E.verdict_rule(d) == want, m
    assert E.verdict_rule_se_units(_diffs_mit_mittel_und_se(0.03, 0.01)) == pytest.approx(3.0)


def test_regel_ist_symmetrisch_und_unabhaengig_von_der_skala():
    for d in ([-0.03, -0.01, -0.02, -0.04, -0.01], [0.02, 0.05, 0.03, 0.01, 0.04], [-0.02, 0.03, -0.01, 0.02, 0.0]):
        d = np.array(d)
        flip = {C.KERN_VORN: C.BAUM_VORN, C.BAUM_VORN: C.KERN_VORN, C.GLEICHAUF: C.GLEICHAUF}
        assert E.verdict_rule(-d) == flip[E.verdict_rule(d)]
        assert E.verdict_rule(7.5 * d) == E.verdict_rule(d)


def test_zaehlung_k_von_n_zaehlt_nur_echt_kleinere_fehler():
    assert E.count_better([0.1, 0.2, 0.3], [0.2, 0.2, 0.1]) == 1                  # Gleichstand zählt nicht
    assert E.count_better(np.array([0.1 + 1e-15]), np.array([0.1])) == 0


# ---------------------------------------------------------------- Texte
def _comp(verdict, wins=8):
    return E.Comparison("E", "abgestimmt", "reihe1", 20, 0.2246, 0.1188, 0.1, 0.1058, 0.0045, wins, verdict)


def test_verdict_study_nennt_welt_gegner_zaehlung_und_alle_zahlen():
    kind, text = E.verdict_study(_comp(C.BAUM_VORN, 0))
    assert kind == "warning"
    for teil in ("Welt E", "Gegner abgestimmt", "20 Zufalls-Seeds", "Gauß-SVM 22,5 %", "bester Baum 11,9 %", "Bayes 10,0 %", "+10,6 Prozentpunkte", "Standardfehler 0,4", "0 von 20", "**Baum vorn**"):
        assert teil in text, teil
    assert E.verdict_study(_comp(C.KERN_VORN, 20))[0] == "success" and E.verdict_study(_comp(C.GLEICHAUF, 8))[0] == "info"


def test_verdict_study_frische_reihe_wird_als_solche_genannt():
    c = E.Comparison("B", "Standard", "frisch", 20, 0.28, 0.285, 0.24, -0.005, 0.003, 12, C.GLEICHAUF)
    assert "der frischen Zufalls-Seeds" in E.verdict_study(c)[1] and "der ersten" in E.verdict_study(_comp(C.GLEICHAUF))[1]


def test_verdict_live_instance_enthaelt_kein_verdikt_wort():
    from kob_models import LiveResult
    r = LiveResult(0.15, 0.17, 0.23, 1.0, 0.1, 1.2, 10, 8, 30, 0.097)
    kind, text = E.verdict_live_instance(r, "P", 300, 7)
    assert kind == "info" and "Kernel Ridge 15,0 %" in text and "CART mit Blattgröße 10 17,0 %" in text and "voll gewachsene CART 23,0 %" in text
    assert not any(w in text for w in ("Kern vorn", "Baum vorn", "gleichauf"))
    assert "belegt kein Verdikt" in text


def test_verdict_scaling_nennt_zaehlung_und_die_spalte():
    s = {"n": 20, "roh": 0.2343, "std": 0.2079, "minmax": 0.1997, "roh_schlechter": 18, "diff": 0.0264, "diff_se": 0.003, "anteil_max": 0.9894, "anteil_argmax": 1}
    kind, text = E.verdict_scaling(s)
    assert kind == "warning" and "roh 23,43 %" in text and "standardisiert 20,79 %" in text and "[0, 1] skaliert 19,97 %" in text
    assert "18 von 20" in text and "Ladegewicht bestimmt 98,9 %" in text and "ohne Meldung" in text


def test_verdict_texte_der_studie_haben_keine_unaufgeloesten_platzhalter():
    from kob_evaluation import verdict_live_study, verdict_noise, verdict_rotation, verdict_scaling
    texts = [verdict_live_study(w)[1] for w in C.LIVE_WORLDS]
    texts += [verdict_rotation(E.rotation_summary("E", "Er"))[1], verdict_scaling(E.scaling_summary())[1], verdict_noise(E.noise_loss())[1]]
    for t in texts:
        assert "{" not in t and "}" not in t and "nan" not in t.lower().split() and "None" not in t
        assert not re.search(r"\d\.\d", t), t                       # Dezimalkomma statt Dezimalpunkt
