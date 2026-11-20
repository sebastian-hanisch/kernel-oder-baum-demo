"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2): die Kernroutinen gegen einen anderen Rechenweg.

* Kernel-Ridge-Klassifikator der App gegen scikit-learn `KernelRidge` (Fremdreferenz) und gegen die Eigenzerlegung der Gram-Matrix;
* Median-Heuristik gegen `scipy.spatial.distance.pdist`;
* Drehinvarianz: Gauß-Gram-Matrix vor und nach der Drehung gleich bis 1e-13, Entscheidungsfunktion der scikit-learn-SVM gleich bis 1e-7, Vorhersagen gleich;
* Verdikt-Regel gegen die t-Statistik aus scipy (|t| > 2 gleich „nicht gleichauf“);
* CART gegen scikit-learn und gegen Aufzählung (tests/test_tree.py, tests/test_hand.py), Bayes-Fehler gegen Quadratur (tests/test_worlds.py);
* Eigenschaften über 20 bis 50 Zufallsinstanzen (fester Zufalls-Seed): Untergrenze nie größer als der abgestimmte Weg, Bayes-Fehler unter jedem gemessenen Mittel, Anteil k/p fällt in p.
"""

import math

import numpy as np
import pytest

import kob_evaluation as E
import kob_frozen as FZ
import kob_models as M
import kob_worlds as W
from kob_rng import SplitMix64


def _krr_instanz(s, n=60, world="A"):
    w = W.generate(world, n, 400 + s)
    S, _ = M.scale_features(w.X, w.X, "standardisiert")
    return S, w.y


# ---------------------------------------------------------------- Kernel Ridge
def test_kernel_ridge_gegen_scikit_learn_und_eigenzerlegung_auf_20_instanzen():
    sk = pytest.importorskip("sklearn.kernel_ridge")
    for s in range(20):
        S, y = _krr_instanz(s)
        f, lam = M.SCALE_FACTORS[s % 5], M.LAMBDAS[s % 3]
        mine = M.fit_krr(S, y, f, lam)
        ref = sk.KernelRidge(alpha=lam, kernel="rbf", gamma=mine.gamma).fit(S, 2.0 * y - 1.0)
        Xnew = W.generate("A", 40, 900 + s).X
        Snew = (Xnew - Xnew.mean(0)) / Xnew.std(0)
        assert np.allclose(mine.decision(Snew), ref.predict(Snew), atol=1e-8), s
        K = M.gauss_gram(S, S, mine.gamma)
        lam_vals, U = np.linalg.eigh(K)
        alpha_eig = U @ ((U.T @ (2.0 * y - 1.0)) / (lam_vals + lam))
        assert np.allclose(mine.coef, alpha_eig, atol=1e-8), s


def test_median_heuristik_gegen_pdist():
    spd = pytest.importorskip("scipy.spatial.distance")
    for s in range(20):
        w = W.generate("E", 50 + 3 * s, 500 + s)
        assert M.median_sigma(w.X) == pytest.approx(float(np.median(spd.pdist(w.X))), rel=1e-12)


# ---------------------------------------------------------------- Drehinvarianz
def test_gauss_gram_drehinvariant_und_svm_entscheidung_gleich_bis_1e_minus_7():
    svm = pytest.importorskip("sklearn.svm")
    for s in range(10):
        base = W.generate("E", 60, 600 + s)
        Q = W.random_rotation(10, 700 + s)
        g = 0.05 + 0.02 * s
        assert np.max(np.abs(M.gauss_gram(base.X, base.X, g) - M.gauss_gram(base.X @ Q, base.X @ Q, g))) < 1e-13
        te = W.generate("E", 100, 800 + s).X
        a = svm.SVC(kernel="rbf", gamma=g, C=1.0, tol=1e-12).fit(base.X, base.y)
        b = svm.SVC(kernel="rbf", gamma=g, C=1.0, tol=1e-12).fit(base.X @ Q, base.y)
        assert np.max(np.abs(a.decision_function(te) - b.decision_function(te @ Q))) < 1e-7
        assert np.array_equal(a.predict(te), b.predict(te @ Q))


# ---------------------------------------------------------------- Verdikt-Regel gegen die t-Statistik
def test_verdikt_regel_gegen_die_t_statistik_aus_scipy():
    stats = pytest.importorskip("scipy.stats")
    for s in range(50):
        r = SplitMix64(s)
        d = W.normal_array(r, (20,)) * 0.02 + 0.03 * (s % 5 - 2) / 2.0
        t = stats.ttest_1samp(d, 0.0).statistic
        want = E.verdict_rule(d)
        assert (want == "gleichauf") == (abs(t) <= 2.0), (s, t)
        if want != "gleichauf":
            assert (want == "Baum vorn") == (t > 0)
        assert E.verdict_rule_se_units(d) == pytest.approx(t, rel=1e-10)


# ---------------------------------------------------------------- Eigenschaften der eingefrorenen Studie
def test_untergrenze_nie_groesser_als_der_abgestimmte_weg_und_bayes_unter_jedem_mittel():
    for w in FZ.worlds_with_study():
        for tuned, floor in (("rf_t", "rf_orakel"), ("gb_t", "gb_orakel"), ("rbf", "rbf_orakel")):
            a, b = FZ.haupt_array(w, tuned), FZ.haupt_array(w, floor)
            assert np.all(b <= a + 1e-12), (w, tuned)                           # Definition: die am Test gewählte Zelle ist nie schlechter als die gewählte
        bayes = FZ.haupt_array(w, "bayes").mean()
        for key in ("lin", "rbf", "rbf_med", "rf_std", "gb_std", "rf_t", "gb_t"):
            assert FZ.haupt_array(w, key).mean() >= bayes - 3 * E.mean_se(FZ.haupt_array(w, key))[1] - 1e-9, (w, key)


def test_anteil_k_durch_p_faellt_in_p_monoton_und_liegt_im_band():
    """2D-Schwellenregel plus p - 2 Normalzahl-Merkmale, standardisiert: der Anteil der 2 relevanten Spalten am Abstandsquadrat liegt im Band um 2/p, 300 Zeilen je Instanz, 20 Instanzen je p."""
    mittel = []
    for p in (2, 3, 4, 6, 10):
        shares = []
        for s in range(20):
            base = W.generate("E2", 300, 2000 + s).X
            noise = W.normal_array(SplitMix64(3000 + s), (300, p - 2)) if p > 2 else np.zeros((300, 0))
            X = np.column_stack([base, noise])
            S, _ = M.scale_features(X, X, "standardisiert")
            shares.append(M.relevant_share(S, 2, rows=300))
        mittel.append(np.mean(shares))
        se = np.std(shares, ddof=1) / math.sqrt(len(shares)) if p > 2 else 0.0
        assert abs(np.mean(shares) - 2.0 / p) <= 4.5 * se + 1e-9, p
    assert all(a > b for a, b in zip(mittel, mittel[1:]))


def test_anteil_aendert_sich_bei_skalierung_einer_spalte_ladegewicht_anzeige_haengt_an_der_skala():
    X = W.generate("F", 200, 5).X
    roh = M.column_shares(X)
    assert roh[1] > 0.97 and roh.sum() == pytest.approx(1.0)
    S, _ = M.scale_features(X, X, "standardisiert")
    std = M.column_shares(S)
    assert std[1] < 0.2 and std.sum() == pytest.approx(1.0)
    Y = S.copy()
    Y[:, 3] *= 10.0
    assert M.column_shares(Y)[3] > 0.8


def test_mehr_zeilen_verkleinern_den_fehler_der_gauss_svm_im_mittel_lernkurve():
    for w in FZ.teil("lernkurve"):
        if w == "n":
            continue
        lc = E.learning_curve(w)
        assert lc["rbf"][0] > lc["rbf"][-1], w
        assert lc["bayes"][-1] <= lc["rbf"][-1] + 1e-9
