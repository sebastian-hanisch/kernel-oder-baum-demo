"""Randfälle: kleine Instanzen, nur eine Klasse, konstante Merkmale, gleiche Zeilen, Drehwinkel 0 und 45 Grad, Rauschmerkmale 0, Netz ohne Konvergenz (Anzeige statt Zahl)."""

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import kob_constants as C
import kob_evaluation as E
import kob_frozen as FZ
import kob_models as M
import kob_texts as TX
import kob_tree as T
import kob_worlds as W
from kob_rng import SplitMix64

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_ein_und_zwei_zeilen_geben_eine_meldung_statt_einer_zahl():
    for n in (1, 2):
        with pytest.raises(ValueError, match="zu wenige Zeilen"):
            M.select_krr(np.zeros((n, 2)), np.zeros(n, int), SplitMix64(1))
        with pytest.raises(ValueError, match="zu wenige Zeilen"):
            M.select_leaf(np.zeros((n, 2)), np.zeros(n, int), SplitMix64(1))
    with pytest.raises(ValueError):
        M.median_sigma(np.zeros((1, 3)))


def test_nur_eine_klasse_im_training_gibt_konstante_vorhersage_ohne_absturz():
    X = W.generate("A", 40, 1).X
    for cls in (0, 1):
        y = np.full(40, cls)
        S, _ = M.scale_features(X, X, "standardisiert")
        krr = M.select_krr(S, y, SplitMix64(2))
        assert np.all(krr.predict(S) == cls) and krr.cv_errors == 0
        tree, leaf = M.fit_cart(X, y, SplitMix64(3))
        assert tree.n_leaves == 1 and np.all(T.predict(tree, X) == cls)


def test_konstantes_merkmal_wird_nicht_durch_null_geteilt_und_aendert_nichts_am_modell():
    w = W.generate("P", 80, 3)
    X = np.column_stack([w.X, np.full(80, 5.0)])
    S, _ = M.scale_features(X, X, "standardisiert")
    assert np.all(np.isfinite(S)) and np.all(S[:, 2] == 0.0)
    a = M.select_krr(M.scale_features(w.X, w.X, "standardisiert")[0], w.y, SplitMix64(4))
    b = M.select_krr(S, w.y, SplitMix64(4))
    assert (a.f, a.lam, a.cv_errors) == (b.f, b.lam, b.cv_errors)


def test_lauter_gleiche_zeilen_geben_endliches_gamma_und_endliche_vorhersage():
    X = np.ones((20, 3))
    y = np.array([0, 1] * 10)
    assert M.median_sigma(X) == 0.0 and math.isfinite(M.gamma_from_scale(0.0, 1.0))
    krr = M.select_krr(X, y, SplitMix64(5))
    assert np.all(np.isfinite(krr.decision(X)))


def test_kleinste_und_groesste_live_instanz_jeder_welt_laeuft_ohne_fehler():
    for world in C.LIVE_WORLDS:
        for n in (C.N_MIN, C.N_MAX):
            tr, te = W.generate(world, n, 3, rot_seed=9003), W.generate(world, 500, 700003, rot_seed=9003)
            r = M.run_live(tr, te, seed=3)
            assert 0.0 <= r.err_krr <= 1.0 and 0.0 <= r.err_cart <= 1.0


def test_welt_mit_n_unter_der_mindestblattgroesse_waechst_einen_baum_ohne_fehler():
    w = W.generate("P", 12, 1)
    tree = T.grow(w.X, w.y, task="class", criterion="gini", min_leaf=20)
    pred = T.predict(tree, w.X)
    assert tree.n_leaves == 1 and np.all(pred == pred[0])


def test_verdikt_regel_mit_einem_zufalls_seed_oder_gleichen_differenzen():
    assert E.verdict_rule([0.2]) == C.GLEICHAUF                           # ein Zufalls-Seed: kein Standardfehler
    assert E.verdict_rule([0.1, 0.1, 0.1, 0.1]) == C.BAUM_VORN             # Standardfehler 0, Mittel positiv
    assert E.verdict_rule([0.0, 0.0, 0.0]) == C.GLEICHAUF


def test_nenner_null_in_den_abgeleiteten_saetzen():
    assert E.verdict_rule_se_units([0.1, 0.1, 0.1]) == 0.0
    assert E.pp(0.0) == "+0,0 Prozentpunkte" and E.pct(0.0) == "0,0 %"


def test_netz_ohne_konvergenz_zeigt_nicht_konvergiert_statt_einer_zahl(monkeypatch):
    daten = FZ.load()
    kaputt = {**daten, "netz": {**daten["netz"], "E": {**daten["netz"]["E"], "net_unconv": [1] * 20}}}
    monkeypatch.setattr(FZ, "load", lambda: kaputt)
    text = TX.netz_text()
    zeile = next(z for z in text.splitlines() if z.startswith("| E |"))
    assert "nicht konvergiert" in zeile and "kein Verdikt" in zeile and "%" not in zeile.split("nicht konvergiert")[0]


def _app(**werte):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, wert in werte.items():
        widget = next(w for art in ("slider", "selectbox", "radio", "number_input") for w in getattr(at.sidebar, art) if w.key == key)
        widget.set_value(wert)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_mit_drehwinkel_0_und_45_grad_und_ohne_rauschmerkmale():
    for winkel in (C.ANGLE_MIN, C.ANGLE_MAX):
        _app(winkel_slider=winkel, rauschen_slider=0)


def test_app_mit_kleinster_instanz_jedes_gegners_und_jeder_skala():
    for gegner in C.GEGNER:
        _app(gegner_radio=gegner, n_slider=C.N_MIN)
    for skala in C.SCALINGS:
        _app(skalierung_radio=skala, welt_select="F")
