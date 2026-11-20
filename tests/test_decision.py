"""Entscheidungsbaum (kob_decision.py): jede Antwort nennt ihre Zahl aus der Studie, gilt für die gemessene Welt und endet mit dem Hinweis; Verweise auf die Demos der Reihe."""

import kob_decision as DEC
import kob_evaluation as E

SCHWESTERN = ("kerntrick-perzeptron-demo", "harte-marge-demo", "weiche-marge-demo", "kernwahl-demo", "kernel-ridge-demo", "gram-kondition-demo", "gram-naeherung-demo")


def _alle(**ja):
    return {k: (DEC.JA if k in ja else DEC.NEIN) for k in DEC.FRAGEN}


def test_fragen_und_antworten_sind_vollstaendig():
    assert set(DEC.FRAGEN_REIHENFOLGE) == set(DEC.FRAGEN) and len(DEC.FRAGEN) == 6
    assert DEC.ANTWORTEN == ("Ja", "Nein", "Weiß nicht")
    assert all(q.endswith("?") for q in DEC.FRAGEN.values())


def test_jede_antwort_endet_mit_dem_hinweis_dass_sie_fuer_die_gemessene_welt_gilt():
    for antw in ({}, _alle(gerade=1), _alle(schwellen=1, rauschen=1), {k: DEC.JA for k in DEC.FRAGEN}):
        zeilen = DEC.empfehlung(antw)
        assert zeilen[-1][0].startswith("Jede Antwort gilt für die gemessene Welt") and "nicht für Ihre Daten" in zeilen[-1][0]
        assert all(a and b for a, b in zeilen)


def test_gerade_nennt_zaehlung_der_gauss_svm_gegen_die_lineare_in_welt_a_und_d():
    text = " ".join(a + " " + b for a, b in DEC.empfehlung(_alle(gerade=1)))
    for w in ("A", "D"):
        k, n, g, lin = E.compare_linear(w)
        assert f"Welt {w}: Gauß-SVM {E.pct(g)}, lineare SVM {E.pct(lin)}, Gauß besser in {k} von {n} Zufalls-Seeds" in text


def test_schwellen_und_rauschen_nennen_die_welten_der_messung_mit_verdikt():
    text = " ".join(b for _, b in DEC.empfehlung(_alle(schwellen=1, rauschen=1)))
    for w in ("E", "F", "Er", "C", "C2", "F8"):
        c = E.compare_study(w, "abgestimmt")
        assert f"Welt {w}: Gauß-SVM {E.pct(c.kern)}, bester abgestimmter Baum {E.pct(c.baum)}, Verdikt {c.verdict} (Gauß besser in {c.wins} von {c.n} Zufalls-Seeds)" in text


def test_einheiten_nennt_roh_gegen_standardisiert_mit_zaehlung():
    s = E.scaling_summary()
    text = " ".join(b for _, b in DEC.empfehlung(_alle(einheiten=1)))
    assert E.pct(s["roh"], 2) in text and E.pct(s["std"], 2) in text and f"roh schlechter in {s['roh_schlechter']} von {s['n']}" in text


def test_grosses_n_und_wahrscheinlichkeit_verweisen_ohne_eigene_zahl():
    n_text = " ".join(a + b for a, b in DEC.empfehlung(_alle(n_gross=1)))
    assert "gram-naeherung-demo" in n_text and "hängt stark an der Bandbreite" in n_text and "ob und wie sie mit n wächst" in n_text
    w_text = " ".join(a + b for a, b in DEC.empfehlung(_alle(wahrscheinlichkeit=1)))
    assert "kalibrierung-schwelle-demo" in w_text and "nicht gemessen" in w_text


def test_ohne_ja_antworten_oder_bei_lauter_nein_kommt_der_hinweis_beide_abzustimmen():
    for antw in ({}, {k: DEC.NEIN for k in DEC.FRAGEN}, {k: DEC.WEISS_NICHT for k in DEC.FRAGEN}):
        text = " ".join(a for a, _ in DEC.empfehlung(antw))
        assert "Kern und Baum beide abstimmen" in text
    assert "Kern und Baum beide abstimmen" not in " ".join(a for a, _ in DEC.empfehlung(_alle(schwellen=1)))


def test_keine_antwort_sagt_der_baum_ist_besser_ohne_welt():
    for antw in ({}, _alle(gerade=1), _alle(schwellen=1), {k: DEC.JA for k in DEC.FRAGEN}):
        for a, b in DEC.empfehlung(antw):
            assert "Kern ist besser" not in a and "Baum ist besser" not in a


def test_stuecke_der_reihe_sind_die_sieben_demos_und_ohne_verbotene_namensteile():
    namen = tuple(o for o, _ in DEC.STUECKE)
    assert namen == SCHWESTERN
    for n in namen:
        assert n.endswith("-demo") and not any(t in n for t in ("hub", "sso", "prod", "qa"))
