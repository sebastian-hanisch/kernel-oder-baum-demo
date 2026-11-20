"""CART (kob_tree.py): Kopier-Test gegen cart-demo und Gegenprobe gegen scikit-learn nur dort, wo es keine Gleichstände gibt (DEMO-PLAYBOOK §4)."""

from pathlib import Path

import numpy as np
import pytest

import kob_tree as T
import kob_worlds as W
from kob_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
QUELLE = ROOT.parent / "cart-demo" / "cart_algorithm.py"


def test_kob_tree_ist_eine_kopie_von_cart_algorithm():
    if not QUELLE.exists():
        pytest.skip("cart-demo liegt nicht neben der Demo (nur im Arbeitsordner vorhanden)")
    assert (ROOT / "kob_tree.py").read_bytes() == QUELLE.read_bytes()


def test_voll_gewachsener_baum_hat_keinen_trainingsfehler_ohne_doppelte_zeilen():
    w = W.generate("E", 150, 4)
    tree = T.grow(w.X, w.y, task="class", criterion="gini", min_leaf=1)
    assert np.mean(T.predict(tree, w.X) != w.y) == 0.0 and tree.n_leaves >= 2


def test_mindestblattgroesse_wird_eingehalten_und_begrenzt_die_blaetterzahl():
    w = W.generate("E", 200, 5)
    for ml in (1, 5, 20):
        tree = T.grow(w.X, w.y, task="class", criterion="gini", min_leaf=ml)
        leaves = tree.feature < 0
        assert tree.n[leaves].min() >= ml
    assert T.grow(w.X, w.y, "class", "gini", min_leaf=20).n_leaves < T.grow(w.X, w.y, "class", "gini", min_leaf=1).n_leaves


def test_gegenprobe_gegen_scikit_learn_wurzelschnitt_blaetterzahl_und_flache_baeume():
    """Stetige Merkmale (Welt A), Gini: der Wurzelschnitt (Merkmal und Schwelle, scikit-learn rechnet mit float32) ist in allen 10 Instanzen gleich, die Blätterzahl des voll gewachsenen Baums in mindestens 8 von 10, und Bäume der Tiefe 3
    geben in allen 10 Instanzen auf mehr als 85 % von 300 frischen Zeilen dieselben Vorhersagen (Mehrheitsvergleich: exakte Gleichstände der Split-Suche in tiefen Knoten löst scikit-learn anders auf,
    DEMO-PLAYBOOK §4)."""
    sk = pytest.importorskip("sklearn.tree")
    wurzel = blaetter = flach = 0
    for s in range(10):
        tr, te = W.generate("A", 60, 100 + s), W.generate("A", 300, 900 + s)
        mine = T.grow(tr.X, tr.y, task="class", criterion="gini", min_leaf=1)
        theirs = sk.DecisionTreeClassifier(criterion="gini", random_state=0).fit(tr.X, tr.y)
        wurzel += int(mine.feature[0] == theirs.tree_.feature[0] and abs(mine.threshold[0] - theirs.tree_.threshold[0]) < 1e-5)
        blaetter += int(mine.n_leaves == theirs.get_n_leaves())
        m3 = T.grow(tr.X, tr.y, task="class", criterion="gini", max_depth=3, min_leaf=1)
        t3 = sk.DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0).fit(tr.X, tr.y)
        flach += int(np.mean(T.predict(m3, te.X) == t3.predict(te.X)) > 0.85)
    assert wurzel == 10 and blaetter >= 8 and flach == 10


def test_blaetterzahl_waechst_bei_gedrehten_merkmalen_welt_e_im_mittel():
    """Gegenprobe zu „der Baum ist nicht drehinvariant“: nur als Mittel über Zufalls-Seeds in Welt E (300 Zeilen), nicht für jede Aufgabe."""
    n0, n1 = [], []
    for s in range(8):
        a = W.generate("E", 300, 1000 + s)
        b = W.generate("Er", 300, 1000 + s, rot_seed=9000 + s)
        n0.append(T.grow(a.X, a.y, task="class", criterion="gini", min_leaf=1).n_leaves)
        n1.append(T.grow(b.X, b.y, task="class", criterion="gini", min_leaf=1).n_leaves)
    assert np.mean(n1) > np.mean(n0)
    assert sum(1 for x, y in zip(n0, n1) if y > x) >= 7


def test_eine_wahrheit_entlang_der_diagonale_verliert_durch_drehung_nichts():
    """Gegenbeispiel zur Absolutaussage: Klasse = x1 + x2 > 0; nach der Drehung um 45 Grad ist die Regel achsenparallel: weniger Blätter, nicht mehr."""
    rng = SplitMix64(3)
    X = W.uniform_array(rng, (200, 2), -1.0, 1.0)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xr = X @ W.rotation_2d(45.0).T
    assert T.grow(Xr, y, task="class", criterion="gini", min_leaf=1).n_leaves < T.grow(X, y, task="class", criterion="gini", min_leaf=1).n_leaves
