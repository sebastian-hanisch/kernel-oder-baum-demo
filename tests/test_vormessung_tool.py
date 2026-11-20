"""Werkzeug `tools/kob_make_vormessung.py`: Neu-Rechnen der eingefrorenen Studie als Band (Wald, Boosting und Netz hängen an Version und Plattform; nicht exakt)."""

import json

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import kob_frozen as FZ  # noqa: E402
import kob_make_vormessung as T  # noqa: E402


def test_datei_hat_alle_teile_und_meta():
    d = json.loads(FZ.PATH.read_text(encoding="utf-8"))
    assert set(d) == {"meta", "haupt", "netz", "skalierung", "ringe", "lernkurve", "live"}
    assert set(d["haupt"]) == set(T.HAUPT_WELTEN) and set(d["netz"]) == set(T.NETZ_WELTEN) and set(d["live"]) == set(T.LIVE_WELTEN)
    assert d["meta"]["scikit-learn"] and d["meta"]["python"] and d["meta"]["gitter"]["Netz"]["Iterationsgrenze"] == 5000


def test_neu_rechnen_welt_p_zwei_zufalls_seeds_im_band_gegen_die_datei():
    assert T.check(welten=("P",), seeds=(0, 1), tol=0.02) == 0


def test_eine_zelle_der_hauptreihe_hat_die_erwarteten_felder_und_ordnung():
    r = T.one_haupt(("P", 3))
    for k in T.HAUPT_KEYS:
        assert k in r
    assert r["bayes_exp"] == pytest.approx(0.1, abs=1e-12)
    assert r["rf_orakel"] <= r["rf_t"] + 1e-12 and r["gb_orakel"] <= r["gb_t"] + 1e-12 and r["rbf_orakel"] <= r["rbf"] + 1e-12
    assert r["lin"] > 0.3                                                       # die lineare SVM kann den Kreis nicht trennen


def test_ringe_bayes_fehler_der_welt_ri_ist_klein():
    r = T.one_ringe((0, 1.0))
    assert r["bayes_exp"] < 0.003 and r["med"] < 0.02
