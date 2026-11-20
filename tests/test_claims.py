"""Jede Zahl und jede Aussage der README steht hier als Rechnung aus der eingefrorenen Studie (kob_vormessung.json) über dieselben Auswertungsfunktionen wie die App (DEMO-PLAYBOOK §4, §11a).

Die Studie ist eine Neumessung mit den Strömen der App (SplitMix64); alle Zahlen sind Bänder über Zufalls-Seeds. Was nicht hier steht, steht in der README unter „Nicht im Test“.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kob_constants as C
import kob_evaluation as E
import kob_frozen as F

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP_TEXTS = "\n".join((ROOT / n).read_text(encoding="utf-8") for n in ("app.py", "kob_texts.py", "kob_decision.py", "kob_constants.py", "kob_evaluation.py"))


def _wo(text):
    """Text der README ohne Zeilenumbrüche an Satzmitten (die README bricht Absätze um)."""
    return re.sub(r"\s*\n\s*", " ", text)


R = _wo(README)


# ---------------------------------------------------------------- Tabellen wörtlich
def test_tabellen_stehen_woertlich_in_der_readme():
    for tabelle in (E.table_main("reihe1"), E.table_differences("reihe1"), E.table_differences("frisch"), E.table_tuning(), E.table_live(), E.table_netz(), E.table_ringe(), E.table_lernkurve()):
        assert tabelle in README, tabelle.splitlines()[0]


# ---------------------------------------------------------------- Kernbotschaft je Welt
def _c(w, st="abgestimmt", reihe="reihe1"):
    return E.compare_study(w, st, reihe)


def test_kernbotschaft_je_welt():
    e, er, p = _c("E"), _c("Er"), _c("P")
    rf = F.haupt_array("E", "rf_t").mean()
    assert (f"Gauß-SVM {E.pct(e.kern)}, abgestimmter Wald {E.pct(rf)}, der je Zufalls-Seed bessere von Wald und Boosting im Mittel {E.pct(e.baum)}; Differenz zu diesem besten Baum {E.pp(e.d_mean)}, "
            f"Gauß besser in {E.k_of(e.wins, e.n)} Zufalls-Seeds): **Baum vorn**") in R
    assert e.verdict == C.BAUM_VORN and e.wins == 0 and E.pct(e.baum) == "11,3 %"
    assert abs((e.kern - e.baum) - e.d_mean) < 1e-9                                           # die genannte Differenz gilt gegen den besten Baum, nicht gegen den Wald allein
    assert er.verdict == C.KERN_VORN and f"Differenz zum besten Baum {E.pp(er.d_mean)}, Gauß besser in {E.k_of(er.wins, er.n)} Zufalls-Seeds" in R
    assert f"(Gauß-SVM {E.pct(F.haupt_array('E', 'rbf').mean())} → {E.pct(F.haupt_array('Er', 'rbf').mean())}) und der Wald verliert ({E.pct(rf)} → {E.pct(F.haupt_array('Er', 'rf_t').mean())})" in R
    assert p.verdict == C.KERN_VORN and f"**Kern vorn** in {E.k_of(p.wins, p.n)} Zufalls-Seeds, Differenz zum besten Baum {E.pp(p.d_mean)}" in R
    for w in ("C", "C2", "F8", "F"):
        assert _c(w).verdict == C.BAUM_VORN, w
    # Wechselwirkung und Container: gegen Standard-Bäume gleichauf bis Kern vorn, nach der Abstimmung eine Stufe zu den Bäumen
    assert [_c("B", "Standard").verdict, _c("B", "Standard", "frisch").verdict] == ["gleichauf", "Kern vorn"] and _c("B").verdict == C.BAUM_VORN and _c("B", "abgestimmt", "frisch").verdict == C.BAUM_VORN
    assert [_c("D", "Standard").verdict, _c("D", "Standard", "frisch").verdict] == ["Kern vorn", "Kern vorn"]
    assert [_c("D").verdict, _c("D", "abgestimmt", "frisch").verdict] == ["gleichauf", "Baum vorn"]
    assert "B: Baum vorn, D: gleichauf in der ersten, Baum vorn in der frischen Reihe" in R
    assert "in der frischen Reihe um zwei Stufen (Kern vorn → Baum vorn)" in R
    assert [_c("B", "Standard", "frisch").verdict, _c("B", "abgestimmt", "frisch").verdict, _c("D", "Standard", "frisch").verdict, _c("D", "abgestimmt", "frisch").verdict] == ["Kern vorn", "Baum vorn", "Kern vorn", "Baum vorn"]
    assert _c("A").verdict == C.KERN_VORN


def test_abstimmung_verschiebt_b_und_d_in_beiden_reihen_a_und_p_nicht():
    for w in ("B", "D"):
        assert E.shifts_toward_trees(w, "reihe1") and E.shifts_toward_trees(w, "frisch"), w
    for w in ("A", "P"):
        assert not E.shifts_toward_trees(w, "reihe1") and not E.shifts_toward_trees(w, "frisch"), w
    assert "in B und D verschiebt die Abstimmung das Verdikt in beiden Reihen um mindestens eine Stufe zu den Bäumen, in A und P in keiner" in R
    frisch = __import__("kob_texts").tuning_frisch_text()
    assert "Welt B: erste Reihe gleichauf → Baum vorn, frische Zufalls-Seeds 20 bis 39 Kern vorn → Baum vorn" in frisch
    assert "Welt D: erste Reihe Kern vorn → gleichauf, frische Zufalls-Seeds 20 bis 39 Kern vorn → Baum vorn" in frisch
    assert "Welt A: erste Reihe Kern vorn → Kern vorn, frische Zufalls-Seeds 20 bis 39 Kern vorn → Kern vorn" in frisch
    assert "In den Welten B, D verschiebt die Abstimmung" in frisch and "in A, P in keiner" in frisch
    assert not re.search(r"\b\d+ von 6 Welten kippen", R)


def test_abstimmung_belegbarer_zuwachs_und_schlechter_spanne():
    wald = [w for w in F.worlds_with_study() if E.tuning_gain(w, "rf")[0] > 2 * E.tuning_gain(w, "rf")[1]]
    boost = [w for w in F.worlds_with_study() if E.tuning_gain(w, "gb")[0] > 2 * E.tuning_gain(w, "gb")[1]]
    assert sorted(wald) == ["B", "C", "D", "E"] and sorted(boost) == ["A", "C", "D", "E", "P"]
    assert not {"F", "F8", "C2"} & (set(wald) | set(boost))
    hurts = [E.tuning_gain(w, k)[2] for w in F.worlds_with_study() for k in ("rf", "gb")]
    assert (min(hurts), max(hurts)) == (1, 14)
    assert "Wald in B, C, D und E und dem Boosting in A, C, D, E und P" in R and "in F, F8 und C2 keinen" in R
    assert "abgestimmt schlechter als Standard in 1 bis 14 von 20 Zufalls-Seeds je Welt und Verfahren" in R


def test_untergrenze_und_gitterrand():
    gleich = [w for w in F.worlds_with_study() if _c(w).verdict == _c(w, "am Test gewählt (Untergrenze)").verdict]
    assert len(gleich) == 10 and "D" not in gleich and len(F.worlds_with_study()) == 11
    assert _c("D").verdict == C.GLEICHAUF and _c("D", "am Test gewählt (Untergrenze)").verdict == C.BAUM_VORN
    assert "ändert das Verdikt in 10 von 11 Welten nicht; in D geht es von gleichauf zu Baum vorn" in R
    gb = np.array(F.haupt_world("E")["gb_ntrees"][:20])
    assert int((gb == 25).sum()) == 16 and "(25 Bäume) in 16 von 20 Zufalls-Seeds" in R
    rand = {w: int(np.sum(np.isin(np.array(F.haupt_world(w)["rbf_C"][:20]), (0.1, 100.0)))) for w in ("A", "B", "E", "D", "F", "P")}
    assert (min(rand.values()), max(rand.values())) == (4, 10) and "in 4 bis 10 von 20 Zufalls-Seeds am Rand des Gitters" in R
    gro = {w: int(np.sum(np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 0.1) | np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 10.0))) for w in ("A", "B", "E", "D", "F", "P")}
    beide = {w: int(np.sum(np.isin(np.array(F.haupt_world(w)["rbf_C"][:20]), (0.1, 100.0)) | np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 0.1) | np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 10.0))) for w in ("A", "B", "E", "D", "F", "P")}
    assert (min(gro.values()), max(gro.values())) == (1, 11) and (min(beide.values()), max(beide.values())) == (4, 14)      # zweiter Gitterrand: der Faktor g auf gamma
    assert "der Faktor g auf γ in 1 bis 11, mindestens einer der beiden in 4 bis 14 von 20 Zufalls-Seeds" in R


# ---------------------------------------------------------------- Drehung
def test_drehung():
    rot = E.rotation_summary("E", "Er")
    assert f"lineare SVM {E.pct(rot['lin']['vorher'])} → {E.pct(rot['lin']['nachher'])}" in R
    assert f"**Gauß-SVM {E.pct(rot['rbf']['vorher'])} → {E.pct(rot['rbf']['nachher'])}** (Differenz {E.pp(rot['rbf']['diff'])}, Standardfehler {E.num(100 * rot['rbf']['diff_se'])})" in R
    assert f"abgestimmter Wald {E.pct(rot['rf_t']['vorher'])} → {E.pct(rot['rf_t']['nachher'])} ({E.snum(100 * rot['rf_t']['diff'])}, Standardfehler {E.num(100 * rot['rf_t']['diff_se'])}; schlechter in {E.k_of(rot['rf_t']['schlechter'], 20)} Zufalls-Seeds)" in R
    assert f"abgestimmtes Boosting {E.pct(rot['gb_t']['vorher'])} → {E.pct(rot['gb_t']['nachher'])}" in R
    v0, v1 = rot["verdikt_vorher"], rot["verdikt_nachher"]
    assert (v0.verdict, v1.verdict) == (C.BAUM_VORN, C.KERN_VORN) and v0.wins == 0 and v1.wins == 20
    assert f"ungedreht **Baum vorn** ({E.pp(v0.d_mean)}, Gauß besser in {v0.wins} von 20), gedreht **Kern vorn** ({E.pp(v1.d_mean).replace(' Prozentpunkte', '')}, {v1.wins} von 20)" in R
    rb = E.rotation_summary("B", "Br")
    assert (rb["verdikt_vorher"].verdict, rb["verdikt_nachher"].verdict) == (C.BAUM_VORN, C.KERN_VORN) and (rb["verdikt_vorher"].wins, rb["verdikt_nachher"].wins) == (7, 18)
    assert f"Gauß-SVM {E.pct(rb['rbf']['vorher'])} → {E.pct(rb['rbf']['nachher'])}, abgestimmter Wald {E.pct(rb['rf_t']['vorher'])} → {E.pct(rb['rf_t']['nachher'])} ({E.snum(100 * rb['rf_t']['diff'])}, {rb['rf_t']['schlechter']} von 20 schlechter)" in R
    assert abs(rb["rbf"]["diff"]) < 0.0005 and f"um {E.num(100 * abs(rb['rbf']['diff']), 2)} Prozentpunkte" in R
    assert rb["rf_t"]["schlechter"] == 20 and rot["rf_t"]["schlechter"] == 20
    lv_e, lv_er = F.teil("live")["E"], F.teil("live")["Er"]
    assert f"{E.num(float(np.mean(lv_e['leaves_full'])))} Blätter, in Welt Er {E.num(float(np.mean(lv_er['leaves_full'])))}" in R


def test_gauss_kern_aenderung_unter_drehung_ist_klein_gegen_wald():
    rot = E.rotation_summary("E", "Er")
    assert abs(rot["rbf"]["diff"]) < 2 * rot["rbf"]["diff_se"] + 1e-9 and rot["rf_t"]["diff"] > 10 * abs(rot["rbf"]["diff"])


# ---------------------------------------------------------------- Rauschmerkmale, Einheiten
def test_rauschmerkmale():
    cc, c2, f8 = _c("C"), _c("C2"), _c("F8")
    assert f"Welt C (100 Merkmale, 3 informativ, 100 Zeilen) {E.snum(100 * cc.d_mean)} Prozentpunkte, Gauß besser in {E.k_of(cc.wins, 20)} Zufalls-Seeds; Welt C2 {E.snum(100 * c2.d_mean)}, {E.k_of(c2.wins, 20)}" in R
    assert f"Lieferungen mit 8 Rauschmerkmalen (F8) {E.snum(100 * f8.d_mean)}, {f8.wins} von 20" in R
    loss = E.noise_loss()
    assert (f"(F → F8) {E.num(100 * loss['rbf'][0])} Prozentpunkte (Standardfehler {E.num(100 * loss['rbf'][1])}), der abgestimmte Wald {E.num(100 * loss['rf_t'][0])} ({E.num(100 * loss['rf_t'][1])}), "
            f"das abgestimmte Boosting {E.num(100 * loss['gb_t'][0])} ({E.num(100 * loss['gb_t'][1])})") in R
    assert loss["rbf"][0] > loss["rf_t"][0] > loss["gb_t"][0] > 0
    k, n, g, lin = E.compare_linear("C")
    assert f"({E.pct(g)} gegen {E.pct(lin)}, in {k} von {n} Zufalls-Seeds besser)" in R
    assert all(_c(w).verdict == C.BAUM_VORN for w in ("C", "C2", "F8"))


def test_einheiten():
    s = E.scaling_summary()
    assert (f"**roh {E.pct(s['roh'])}** (Standardfehler {E.num(100 * s['roh_se'])}), **standardisiert {E.pct(s['std'])}** ({E.num(100 * s['std_se'])}), **auf [0, 1] skaliert {E.pct(s['minmax'])}**") in R
    assert f"roh schlechter als standardisiert in {E.k_of(s['roh_schlechter'], s['n'])} Zufalls-Seeds (+{E.num(100 * s['diff'])} Prozentpunkte, Standardfehler {E.num(100 * s['diff_se'])})" in R
    assert f"auf der rohen Skala {E.pct(s['roh_cv'])}" in R
    assert f"σ_med ist roh {E.num(s['sigma_roh'])} gegen {E.num(s['sigma_std'])} standardisiert" in R
    assert s["anteil_argmax"] == 1 and f"bestimmt {E.pct(s['anteil_max'])} des mittleren Abstandsquadrats, die Distanz {E.pct(s['anteil'][0])}" in R
    assert s["roh_schlechter"] == 18 and s["roh"] > s["std"] > s["minmax"]
    assert abs(s["roh_cv"] - s["roh"]) < 0.01                                # die Bandbreite kann zwei Skalen nicht reparieren: gleiches Niveau
    m10 = F.teil("ringe")["mal10"]
    assert (f"Median-Heuristik roh {E.pct(float(np.mean(m10['med'])))}, standardisiert {E.pct(float(np.mean(m10['std'])), 2)}, γ und C per Kreuzvalidierung auf roher Skala (Gitter 0,01 bis 100) {E.pct(float(np.mean(m10['cv_g=0,01 bis 100'])))}") in R
    assert f"Bayes-Fehler {E.pct(float(np.mean(F.teil('ringe')['reihe1']['bayes_exp'])), 2)}" in R


def test_ringe_richtung_bleibt_abstand_nicht_monoton():
    gitter = ("g=0,01 bis 100", "g=0,1 bis 10", "g=0,2 bis 5", "g=0,5 bis 2")
    for reihe in ("reihe1", "frisch"):
        abst = [E.ringe_summary(reihe, g)["cv"] - E.ringe_summary(reihe, g)["med"] for g in gitter]
        assert all(a > 0 for a in abst)                                      # die Heuristik ist in keinem Gitter im Mittel schlechter
    abst1 = [E.ringe_summary("reihe1", g)["cv"] - E.ringe_summary("reihe1", g)["med"] for g in gitter]
    assert any(b > a for a, b in zip(abst1, abst1[1:])) and any(b < a for a, b in zip(abst1, abst1[1:]))     # nicht monoton


# ---------------------------------------------------------------- Netz
def test_netz():
    conv = E.net_convergence()
    assert (conv["unconv5000"], conv["unconv300"], conv["cases"], conv["msg5000"], conv["fits5000"]) == (0, 120, 120, 0, 2520)
    assert f"in {conv['unconv5000']} von {conv['cases']} Fällen ausgeschöpft" in R and f"in {conv['unconv300']} von {conv['cases']} Fällen aus" in R
    assert f"{conv['msg5000']} Konvergenzmeldungen unter {conv['fits5000']:,} Anpassungen".replace(",", " ") in R and f"Iterationen {conv['iter_min']} bis {conv['iter_max']:,}".replace(",", " ") in R
    stufen = {w: E.net_vs_gauss(w)[2] for w in F.teil("netz")}
    assert stufen == {"A": "gleichauf", "B": "Netz vorn", "E": "Netz hinten", "D": "Netz hinten", "F": "Netz hinten", "P": "Netz hinten"}
    assert E.net_other_start_label("A") == "Netz hinten"
    assert all(E.net_other_start_label(w) == stufen[w] for w in ("B", "E", "D", "F", "P"))
    assert "nur in B vorn, in E, D, F und P hinten und in A gleichauf bis hinten" in R


# ---------------------------------------------------------------- Live gegen Studie
def test_live_gegen_studie():
    abweichend = [w for w in F.teil("live") if E.live_deviates(w)]
    assert sorted(abweichend) == ["B", "D"]
    assert E.live_vs_study("B")["verdict"] == C.KERN_VORN and _c("B").verdict == C.BAUM_VORN
    assert E.live_vs_study("D")["verdict"] == C.KERN_VORN and _c("D").verdict == C.GLEICHAUF
    assert "in A, E, Er, F, F8 und P mit dem Studien-Verdikt überein; **in B und D weicht es ab**" in R
    for w in ("E", "F", "F8"):
        assert E.live_vs_study(w, "full")["verdict"] == C.GLEICHAUF, w
    abstand = max(abs(E.live_vs_study(w)["krr"] - _c(w).kern) for w in F.teil("live"))
    assert abstand <= 0.005 and "höchstens 0,5 Prozentpunkte je Welt" in R
    assert "Live-Modell der App** (Kernel Ridge gegen eine einzelne CART) ist nicht die Studie und gibt deren Verdikte in den Welten B und D nicht wieder" in R


def test_kern_gegen_linear():
    zaehl = {w: E.compare_linear(w)[0] for w in F.worlds_with_study() if w not in ("Er", "Br", "F8")}
    assert zaehl == {"A": 8, "B": 20, "C": 12, "C2": 19, "E": 20, "D": 3, "F": 19, "P": 20}
    assert "A in 8, in B in 20, in C in 12, in C2 in 19, in E in 20, in D in 3, in F in 19, in P in 20 von 20 Zufalls-Seeds" in R
    for w in ("A", "D"):
        k, n, g, lin = E.compare_linear(w)
        assert f"{E.pct(g)} gegen {E.pct(lin)}" in R
    assert "auf der fast linearen Welt A (8 von 20 Zufalls-Seeds besser) und auf dem Container-Erzeuger D (3 von 20) keinen Vorteil" in R


# ---------------------------------------------------------------- Lernkurve
def test_lernkurve():
    for w in ("A", "B", "E", "D", "F", "P"):
        lc = E.learning_curve(w)
        assert lc["rbf"][-1] < lc["rbf"][0], w
    e, f = E.learning_curve("E"), E.learning_curve("F")
    assert e["rbf_wins"][-1] == 0 and f["rbf_wins"][-1] == 0
    b, d = E.learning_curve("B"), E.learning_curve("D")
    assert (b["rbf_wins"][-1], d["rbf_wins"][-1]) == (7, 5)
    assert abs(b["rbf"][-1] - b["rf"][-1]) < 0.003 and abs(d["rbf"][-1] - d["rf"][-1]) < 0.003
    assert f"(Gauß-SVM besser in {e['rbf_wins'][-1]} von 10 Zufalls-Seeds); in B und D liegen beide bei n = 1 600 im Mittel gleich ({b['rbf_wins'][-1]} und {d['rbf_wins'][-1]} von 10 Zufalls-Seeds" in R
    assert e["seeds"] == 10


# ---------------------------------------------------------------- Aufbau der README
def test_hauptreihe_zaehlt_zwanzig_zufalls_seeds_je_reihe_und_vierzig_in_a_b_d_p():
    for w in F.worlds_with_study():
        assert len(F.haupt_world(w)["seeds"]) == (40 if w in ("A", "B", "D", "P") else 20), w
    assert F.meta()["testpunkte_je_zufalls_seed"] == 4000 and "SplitMix64" in F.meta()["zufallsquelle"]
    assert E.compare_study("A").n == 20 and E.compare_study("A", reihe="frisch").n == 20


def test_aufwand_der_abstimmung_in_angepassten_modellen():
    assert (C.FITS_LINEAR, C.FITS_GAUSS, C.FITS_FOREST, C.FITS_BOOST) == (20, 100, 30, 30)
    assert "lineare SVM 20, Gauß-SVM 100, Wald 30, Boosting 30" in R


def test_readme_testzahl_stimmt_mit_der_suite_ueberein():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=str(ROOT), timeout=300).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, out[-300:]
    assert f"**{m.group(1)} Tests**" in README, f"README nennt nicht die Testzahl {m.group(1)}"


# ---------------------------------------------------------------- Wortwahl
VERBOTEN = ("trägt", "tragen", "Saatwert", "Phase 0", "Stück ", "Kern der Spiele", "geplant", "nichts gebaut")


def test_keine_verbotenen_woerter_in_readme_und_app_texten():
    for text, name in ((README, "README"), (APP_TEXTS, "App-Texte")):
        for w in VERBOTEN:
            assert w not in text, (name, w)
        assert not re.search(r"(?<![-\w])Seeds?\b", text.replace("Zufalls-Seeds", "").replace("Zufalls-Seed", "")), name
        assert "numpy.random" not in text.replace("`numpy.random.default_rng`", "")


def test_kein_gedankenstrich_im_fliesstext_der_readme():
    for zeile in README.splitlines():
        if zeile.startswith(("# ", "Diese Demo ist Teil", "|")) or "Sebastian Hanisch – Operations" in zeile:
            continue
        assert " – " not in zeile, zeile[:80]


def test_marge_und_regler_namen_sind_getrennt():
    assert "C** (SVM), **λ** (Ridge im Live-Modell)" in R and "γ = 1/(2σ²)" in R
