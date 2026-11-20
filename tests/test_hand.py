"""Handbeispiele (aus handbeispiele_08.py des Plans, als echte Tests mit denselben engen Toleranzen): Brüche und Ganzzahlen, Aufzählung statt Algorithmus.

1 XOR mit vier Punkten, 2 Drehung mit sechs Punkten (zwei unabhängige Rechenwege), 3 Verdikt-Regel in Brüchen gegen die Gleitkomma-Fassung der App, 4 Anteil des Abstands aus relevanten
Merkmalen k/p, 6 Rechenkosten des exakten Kernel Ridge bei n = 20 000, 7 Median-Heuristik und Umrechnung Faktor g auf gamma <-> Skalenfaktor f."""

import itertools
import math
from fractions import Fraction as F
from functools import lru_cache

import numpy as np
import pytest

import kob_evaluation as E
import kob_tree as T


# ---------------------------------------------------------------- Bausteine (Brüche, ohne die Demo-Module)
def gini(counts):
    n = sum(counts)
    return 1 - sum(F(c, n) ** 2 for c in counts) if n else F(0)


def class_counts(ys, idx):
    return [sum(1 for i in idx if ys[i] == c) for c in (-1, 1)]


def candidate_splits(pts, idx):
    out = []
    for j in range(len(pts[0])):
        vals = sorted({pts[i][j] for i in idx})
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            out.append((j, t, tuple(i for i in idx if pts[i][j] < t), tuple(i for i in idx if pts[i][j] >= t)))
    return out


def gini_gain(ys, idx, left, right):
    n = len(idx)
    return gini(class_counts(ys, idx)) - F(len(left), n) * gini(class_counts(ys, left)) - F(len(right), n) * gini(class_counts(ys, right))


def min_leaves_zero_error(pts, ys):
    """Rechenweg 1: kleinste Blätterzahl eines achsenparallelen Baums ohne Trainingsfehler, rekursiv über Teilmengen."""
    @lru_cache(None)
    def f(idx):
        if len({ys[i] for i in idx}) <= 1:
            return 1
        return min(f(left) + f(right) for _, _, left, right in candidate_splits(pts, idx))
    return f(tuple(range(len(pts))))


def enumerate_trees(pts, ys, depth):
    """Rechenweg 2: Aufzählung aller Bäume bis zur Tiefe `depth`: {(Blätter, Trainingsfehler): Zahl der Bäume}; Blattvorhersage = Mehrheit."""
    @lru_cache(None)
    def rec(idx, d):
        res = {(1, min(class_counts(ys, idx))): 1}
        if d == 0:
            return res
        for _, _, left, right in candidate_splits(pts, idx):
            for (l1, e1), c1 in rec(left, d - 1).items():
                for (l2, e2), c2 in rec(right, d - 1).items():
                    res[(l1 + l2, e1 + e2)] = res.get((l1 + l2, e1 + e2), 0) + c1 * c2
        return res
    return rec(tuple(range(len(pts))), depth)


def verdict_exact(diffs):
    """Verdikt in Brüchen: |Mittel| < 2 Standardfehler <=> m^2 < 4 * var / n (var mit n - 1)."""
    n = len(diffs)
    m = sum(diffs) / n
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    if m * m > 4 * var / n:
        return "Kern vorn" if m < 0 else "Baum vorn"
    return "gleichauf"


# ---------------------------------------------------------------- 1. XOR
XOR_PTS = [(F(-1), F(-1)), (F(-1), F(1)), (F(1), F(-1)), (F(1), F(1))]
XOR_Y = [1, -1, -1, 1]


def test_xor_gini_gewinn_jedes_schnitts_ist_genau_null():
    idx = tuple(range(4))
    assert gini(class_counts(XOR_Y, idx)) == F(1, 2)
    splits = candidate_splits(XOR_PTS, idx)
    assert len(splits) == 2
    assert all(gini_gain(XOR_Y, idx, l, r) == 0 for _, _, l, r in splits)


def test_xor_aufzaehlung_aller_baeume_genau_zwei_ohne_fehler_beide_mit_vier_blaettern():
    trees = enumerate_trees(XOR_PTS, XOR_Y, 2)
    zero = {k: v for k, v in trees.items() if k[1] == 0}
    assert sum(zero.values()) == 2 and set(zero) == {(4, 0)}
    assert sum(v for k, v in enumerate_trees(XOR_PTS, XOR_Y, 1).items() if k[1] == 0) == 0       # kein Baum der Tiefe 1 ohne Fehler
    assert trees[(1, 2)] == 1                                                                    # das einzelne Blatt macht 2 von 4 Fehlern
    assert trees == {(1, 2): 1, (2, 2): 2, (3, 1): 4, (4, 0): 2}


def test_xor_der_cart_algorithmus_waechst_trotz_gewinn_null_und_trifft_alle_vier(xor4):
    """Die Kopie von cart_algorithm.py teilt auch bei Gewinn 0 (wie scikit-learn ohne Mindestgewinn): 4 Blätter, kein Trainingsfehler. Mit Mindestgewinn größer als 0 bliebe der gierige Baum ein Blatt."""
    X, y = xor4
    tree = T.grow(X, y, task="class", criterion="gini", min_leaf=1)
    assert tree.n_leaves == 4 and np.array_equal(T.predict(tree, X), y)


def test_xor_cart_blatt_fehlerquote_ist_ein_halb_mit_mindestgewinn(xor4):
    sklearn_tree = pytest.importorskip("sklearn.tree")
    X, y = xor4
    m = sklearn_tree.DecisionTreeClassifier(min_impurity_decrease=1e-9, random_state=0).fit(X, y)
    assert m.get_n_leaves() == 1 and np.mean(m.predict(X) != y) == 0.5
    full = sklearn_tree.DecisionTreeClassifier(random_state=0).fit(X, y)
    assert full.get_n_leaves() == 4 and np.mean(full.predict(X) != y) == 0.0


# ---------------------------------------------------------------- 2. Drehung
PTS = [(-1, -2), (-1, 3), (-1, 0), (1, -3), (2, 3), (1, 1)]
YS = [-1, -1, -1, 1, 1, 1]
P0 = [(F(a), F(b)) for a, b in PTS]
P1 = [(p[0] + p[1], p[1] - p[0]) for p in P0]                  # Drehung um 45 Grad mal Wurzel 2


def _d2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def test_drehung_alle_abstandsquadrate_genau_verdoppelt():
    assert all(_d2(P1[i], P1[j]) == 2 * _d2(P0[i], P0[j]) for i in range(6) for j in range(6))
    assert [(int(a), int(b)) for a, b in P1] == [(-3, -1), (2, 4), (-1, 1), (-2, -4), (5, 1), (2, 0)]


def test_drehung_kleinster_baum_ohne_fehler_zwei_blaetter_vor_vier_nach_der_drehung():
    assert min_leaves_zero_error(P0, YS) == 2 and min_leaves_zero_error(P1, YS) == 4
    e0, e1 = enumerate_trees(P0, YS, 5), enumerate_trees(P1, YS, 5)
    assert min(k[0] for k in e0 if k[1] == 0) == 2 and min(k[0] for k in e1 if k[1] == 0) == 4
    assert not any(k[1] == 0 and k[0] < 4 for k in e1)
    assert e1[(4, 0)] == 10                                      # zehn verschiedene Bäume mit 4 Blättern erreichen 0 (Aufzählung bis Tiefe 5)
    assert not [k for k in enumerate_trees(P1, YS, 1) if k[1] == 0]


def test_drehung_der_cart_algorithmus_braucht_vier_statt_zwei_blaetter(sechs_punkte):
    X, y = sechs_punkte
    T45 = np.array([[1.0, 1.0], [-1.0, 1.0]])
    t0 = T.grow(X, y, task="class", criterion="gini", min_leaf=1)
    t1 = T.grow(X @ T45.T, y, task="class", criterion="gini", min_leaf=1)
    assert t0.n_leaves == 2 and t1.n_leaves == 4
    sk = pytest.importorskip("sklearn.tree")
    assert sk.DecisionTreeClassifier(random_state=0).fit(X, y).get_n_leaves() == 2
    assert sk.DecisionTreeClassifier(random_state=0).fit(X @ T45.T, y).get_n_leaves() == 4


def test_drehung_gerade_trennt_auch_nach_der_drehung():
    assert all((F(1) * p[0] > 0) == (y > 0) for p, y in zip(P0, YS))
    assert all((F(1, 2) * p[0] - F(1, 2) * p[1] > 0) == (y > 0) for p, y in zip(P1, YS))


# ---------------------------------------------------------------- 3. Verdikt-Regel in Brüchen
FAELLE = [([F(-3, 100), F(-1, 100), F(-2, 100), F(-4, 100), F(-1, 100)], "Kern vorn"),
          ([F(-2, 100), F(3, 100), F(-1, 100), F(2, 100), F(0)], "gleichauf"),
          ([F(2, 100), F(5, 100), F(3, 100), F(1, 100), F(4, 100)], "Baum vorn")]


@pytest.mark.parametrize("diffs,want", FAELLE)
def test_verdikt_regel_exakt_gleich_der_gleitkomma_fassung_der_app(diffs, want):
    assert verdict_exact(diffs) == want
    assert E.verdict_rule([float(x) for x in diffs]) == want


def test_verdikt_regel_fall_eins_mittel_und_varianz_in_bruechen():
    d = FAELLE[0][0]
    m = sum(d) / 5
    var = sum((x - m) ** 2 for x in d) / 4
    assert m == F(-11, 500) and var == F(17, 100000) and 4 * var / 5 == F(17, 125000)
    assert m * m == F(121, 250000) and m * m > 4 * var / 5


# ---------------------------------------------------------------- 4. Abstandsanteil
def test_abstandsanteil_relevanter_merkmale_ist_genau_k_durch_p_aufzaehlung_aller_paare():
    p_dim, k_rel = 5, 3
    tot = rel = cnt = 0
    for x in itertools.product((-1, 1), repeat=p_dim):
        for y in itertools.product((-1, 1), repeat=p_dim):
            sq = [(a - b) ** 2 for a, b in zip(x, y)]
            tot += sum(sq)
            rel += sum(sq[:k_rel])
            cnt += 1
    assert cnt == 1024 and F(rel, cnt) == 2 * k_rel and F(tot, cnt) == 2 * p_dim and F(rel, tot) == F(k_rel, p_dim)


# ---------------------------------------------------------------- 6. Kosten
def test_kosten_des_exakten_kernel_ridge_bei_n_20000_nur_querpruefung_der_zahl():
    n = 20000
    assert n * (n + 1) * (2 * n + 1) // 6 == 2_666_866_670_000
    assert sum((2 * j + 1) * (n - j) for j in range(n)) == 2_666_866_670_000       # zweiter Weg: Spaltensumme
    assert n * n == 400_000_000 and round(n * n * 8 / 2 ** 30, 2) == 2.98


# ---------------------------------------------------------------- 7. Umrechnung Faktor g <-> Skalenfaktor f
@pytest.mark.parametrize("g,f", [(F(1, 100), 10.0), (F(1, 10), 3.16), (F(3, 10), 1.83), (F(1), 1.0), (F(3), 0.58), (F(10), 0.32), (F(100), 0.1)])
def test_faktor_g_auf_gamma_gibt_skalenfaktor_f_gleich_g_hoch_minus_ein_halb(g, f):
    f_num = float(g) ** -0.5
    assert abs(f_num - f) < 0.006
    med = F(3)
    g_med = 1 / (2 * med ** 2)
    assert abs(1 / (2 * (f_num * float(med)) ** 2) - float(g * g_med)) < 1e-12


def test_verdikt_text_ist_eine_reine_funktion_der_differenzen_und_symmetrisch():
    d = np.array([-0.03, -0.01, -0.02, -0.04, -0.01])
    assert E.verdict_rule(d) == "Kern vorn" and E.verdict_rule(-d) == "Baum vorn"
    assert E.verdict_rule(10 * d) == E.verdict_rule(d)                    # invariant gegen gemeinsame Skalierung
    assert E.verdict_rule(np.array([0.01, -0.01, 0.0, 0.02, -0.02])) == "gleichauf"
    assert math.isnan(E.mean_se([0.5])[1]) and E.verdict_rule([0.5]) == "gleichauf"
