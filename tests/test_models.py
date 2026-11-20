"""Live-Modell (kob_models.py): jede Einheit an einer Mini-Instanz von Hand geprüft (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import kob_models as M
import kob_worlds as W
from kob_rng import SplitMix64


# ---------------------------------------------------------------- Median-Heuristik und Skalenfaktor
def test_median_sigma_konvention_paarweise_abstaende_i_kleiner_j_ohne_diagonale():
    """Punkte 0, 1, 3, 6: die Abstände i < j sind 1, 2, 3, 3, 5, 6, Median 3; mit Diagonale und beiden Dreiecken wären es 5/2 (Handbeispiel 7)."""
    X = np.array([[0.0], [1.0], [3.0], [6.0]])
    assert M.median_sigma(X) == 3.0
    ganz = sorted(abs(a - b) for a in (0, 1, 3, 6) for b in (0, 1, 3, 6))
    assert (ganz[7] + ganz[8]) / 2 == 2.5 and M.median_sigma(X) != 2.5


def test_median_sigma_braucht_mindestens_zwei_zeilen():
    with pytest.raises(ValueError):
        M.median_sigma(np.array([[1.0]]))


@pytest.mark.parametrize("f", [0.5, 1.0, 2.0, 4.0])
def test_gamma_ist_gamma_med_durch_f_quadrat(f):
    sm = 3.0
    g_med = 1.0 / (2.0 * sm ** 2)
    assert M.gamma_from_scale(sm, f) == pytest.approx(g_med / f ** 2, rel=1e-14)
    assert M.gamma_from_scale(sm, 1.0) == pytest.approx(g_med, rel=1e-14)


def test_skalenfaktoren_sind_g_hoch_minus_ein_halb_zum_gitter_der_studie():
    assert M.GAMMA_FACTORS == (0.1, 0.3, 1.0, 3.0, 10.0)
    assert [round(f, 2) for f in M.SCALE_FACTORS] == [3.16, 1.83, 1.0, 0.58, 0.32]
    for g, f in zip(M.GAMMA_FACTORS, M.SCALE_FACTORS):
        assert M.gamma_from_scale(3.0, f) == pytest.approx(g / (2.0 * 9.0), rel=1e-12)     # gamma = g * gamma_med
    # Gitterpaare der Ringe-Messung: g von 0,5 bis 2 gibt f von 1,41 bis 0,71, g von 0,2 bis 5 gibt 2,24 bis 0,45
    assert 0.5 ** -0.5 == pytest.approx(1.414, abs=1e-3) and 2 ** -0.5 == pytest.approx(0.707, abs=1e-3)
    assert 0.2 ** -0.5 == pytest.approx(2.236, abs=1e-3) and 5 ** -0.5 == pytest.approx(0.447, abs=1e-3)


# ---------------------------------------------------------------- Gauß-Kern
def test_gram_matrix_ist_symmetrisch_hat_einsen_auf_der_diagonale_und_ist_positiv_semidefinit():
    X = W.generate("A", 60, 3).X
    K = M.gauss_gram(X, X, 0.2)
    assert np.allclose(K, K.T, atol=0.0) and np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > -1e-10 and K.max() <= 1.0 + 1e-15 and K.min() > 0.0


def test_gram_matrix_ist_drehinvariant_bis_auf_rundung():
    X = W.generate("E", 60, 4).X
    Q = W.random_rotation(10, 5)
    g = 0.37
    assert np.max(np.abs(M.gauss_gram(X, X, g) - M.gauss_gram(X @ Q, X @ Q, g))) < 1e-12


def test_gram_matrix_aendert_sich_bei_skalierung_einer_spalte():
    X = W.generate("A", 40, 6).X
    Y = X.copy()
    Y[:, 1] *= 10.0
    assert np.max(np.abs(M.gauss_gram(X, X, 0.2) - M.gauss_gram(Y, Y, 0.2))) > 0.1


def test_abstandsquadrate_nie_negativ_und_gleiche_punkte_geben_null():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [4.0, 6.0]])
    d = M.sq_dists(X, X)
    assert d.min() >= 0.0 and d[0, 1] == 0.0 and d[0, 2] == pytest.approx(25.0)


# ---------------------------------------------------------------- Kernel Ridge: XOR von Hand
@pytest.mark.parametrize("u", [1 / 3, 1 / 10, 7 / 8])
def test_xor_kernmatrix_hat_y_als_eigenvektor_zum_eigenwert_eins_minus_u_quadrat(u, xor4):
    """K = [[1, u, u, u^2], ...] mit u = exp(-4 gamma); y = (+1, -1, -1, +1) ist Eigenvektor zum Eigenwert (1 - u)^2; Kernel Ridge liefert h(x_i) = y_i (1 - u)^2 / ((1 - u)^2 + lambda)."""
    X, y01 = xor4
    t = 2.0 * y01 - 1.0
    gamma = -math.log(u) / 4.0
    K = M.gauss_gram(X, X, gamma)
    assert np.allclose(K @ t, (1 - u) ** 2 * t, atol=1e-13)
    for lam in (1e-6, 0.01, 0.1, 1.0, 10.0):
        a = M.krr_solve(K, t, lam)
        h = K @ a
        assert np.allclose(h, t * (1 - u) ** 2 / ((1 - u) ** 2 + lam), atol=1e-12)
        assert np.all((h > 0) == (t > 0))                                      # alle vier Vorzeichen stimmen für jedes lambda >= 0


def test_krr_loesung_stimmt_mit_der_direkten_loesung_der_normalgleichung():
    rng = SplitMix64(4)
    A = W.normal_array(rng, (30, 3))
    K = M.gauss_gram(A, A, 0.3)
    t = np.where(W.uniform_array(rng, (30,)) > 0.5, 1.0, -1.0)
    for lam in (0.01, 0.1, 1.0):
        assert np.allclose(M.krr_solve(K, t, lam), np.linalg.solve(K + lam * np.eye(30), t), atol=1e-9)
        assert np.allclose((K + lam * np.eye(30)) @ M.krr_solve(K, t, lam), t, atol=1e-9)


def test_fit_krr_trennt_die_sechs_punkte_vor_und_nach_der_drehung_mit_gleichen_koeffizienten(sechs_punkte):
    """Handbeispiel 2: T(x) = (x1 + x2, x2 - x1) streckt alle Abstände um sqrt(2); mit halbem gamma ist die Gram-Matrix gleich, die Koeffizienten damit auch."""
    X, y = sechs_punkte
    T = np.array([[1.0, 1.0], [-1.0, 1.0]])
    Xr = X @ T.T
    d0, d1 = M.sq_dists(X, X), M.sq_dists(Xr, Xr)
    assert np.array_equal(d1, 2.0 * d0)
    g = 1.0 / 8.0
    K0, K1 = M.gauss_gram(X, X, g), M.gauss_gram(Xr, Xr, g / 2.0)
    assert np.max(np.abs(K0 - K1)) < 1e-15
    t = 2.0 * y - 1.0
    a0, a1 = M.krr_solve(K0, t, 1e-6), M.krr_solve(K1, t, 1e-6)
    assert np.max(np.abs(a0 - a1)) < 1e-9 and np.all((K0 @ a0 > 0) == (y == 1))


# ---------------------------------------------------------------- Kreuzvalidierung
def test_falten_sind_eine_aufteilung_aller_zeilen():
    folds = M.make_folds(23, 5, SplitMix64(1))
    allidx = np.sort(np.concatenate(folds))
    assert np.array_equal(allidx, np.arange(23)) and len(folds) == 5
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 1
    assert [list(f) for f in folds] == [list(f) for f in M.make_folds(23, 5, SplitMix64(1))]      # gleicher Zufalls-Seed, gleiche Falten
    assert [list(f) for f in folds] != [list(f) for f in M.make_folds(23, 5, SplitMix64(2))]


def test_select_krr_waehlt_eine_zelle_des_gitters_ist_deterministisch_und_zaehlt_cv_fehler():
    w = W.generate("P", 120, 2)
    S, _ = M.scale_features(w.X, w.X, "standardisiert")
    a = M.select_krr(S, w.y, SplitMix64(7))
    b = M.select_krr(S, w.y, SplitMix64(7))
    assert a.f in M.SCALE_FACTORS and a.lam in M.LAMBDAS and 0 <= a.cv_errors <= 120
    assert (a.f, a.lam, a.cv_errors) == (b.f, b.lam, b.cv_errors) and np.array_equal(a.coef, b.coef)
    assert a.gamma == pytest.approx(M.gamma_from_scale(a.sigma_med, a.f))


def test_select_krr_findet_das_optimum_im_gitter_gegen_aufzaehlung():
    """Gegenprobe: je Gitterzelle die Kreuzvalidierung mit denselben Falten einzeln nachrechnen (ohne die Hilfsfunktion): die gewählte Zelle hat die kleinste Fehlerzahl, bei Gleichstand die erste."""
    w = W.generate("P", 100, 4)
    S = (w.X - w.X.mean(0)) / w.X.std(0)
    folds = M.make_folds(100, 5, SplitMix64(9))
    sm = M.median_sigma(S)
    best = None
    for f in M.SCALE_FACTORS:
        for lam in M.LAMBDAS:
            err = 0
            for fold in folds:
                tr = np.setdiff1d(np.arange(100), fold)
                g = 1.0 / (2.0 * (f * sm) ** 2)
                Ktr = np.exp(-g * ((S[tr][:, None, :] - S[tr][None, :, :]) ** 2).sum(2))
                Kte = np.exp(-g * ((S[fold][:, None, :] - S[tr][None, :, :]) ** 2).sum(2))
                a = np.linalg.solve(Ktr + lam * np.eye(len(tr)), 2.0 * w.y[tr] - 1.0)
                err += int(np.sum((Kte @ a > 0) != (w.y[fold] == 1)))
            if best is None or err < best[0]:
                best = (err, f, lam)
    m = M.select_krr(S, w.y, SplitMix64(9))
    assert (m.cv_errors, m.f, m.lam) == best


def test_zu_wenige_zeilen_fuer_die_kreuzvalidierung_werden_abgelehnt():
    with pytest.raises(ValueError):
        M.select_krr(np.zeros((8, 2)), np.zeros(8, int), SplitMix64(1))
    with pytest.raises(ValueError):
        M.select_leaf(np.zeros((8, 2)), np.zeros(8, int), SplitMix64(1))


# ---------------------------------------------------------------- Skalierung
def test_skalen_standardisiert_roh_und_null_eins():
    X = np.array([[1.0, 100.0], [2.0, 300.0], [3.0, 200.0], [6.0, 400.0]])
    r, _ = M.scale_features(X, X, "roh")
    assert np.array_equal(r, X)
    s, _ = M.scale_features(X, X, "standardisiert")
    assert np.allclose(s.mean(0), 0.0) and np.allclose(s.std(0), 1.0)
    m, _ = M.scale_features(X, X, "[0, 1]")
    assert np.allclose(m.min(0), 0.0) and np.allclose(m.max(0), 1.0)
    with pytest.raises(ValueError):
        M.scale_features(X, X, "unbekannt")


def test_konstante_spalte_wird_nicht_durch_null_geteilt():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    for mode in ("standardisiert", "[0, 1]"):
        S, T = M.scale_features(X, X, mode)
        assert np.all(np.isfinite(S)) and np.all(S[:, 1] == (0.0 if mode == "[0, 1]" else 0.0))


def test_test_zeilen_werden_mit_dem_mass_des_trainings_skaliert():
    Xtr = np.array([[0.0], [10.0]])
    Xte = np.array([[20.0]])
    _, T = M.scale_features(Xtr, Xte, "[0, 1]")
    assert T[0, 0] == 2.0
    _, T2 = M.scale_features(Xtr, Xte, "standardisiert")
    assert T2[0, 0] == pytest.approx((20 - 5) / 5)


# ---------------------------------------------------------------- CART mit Kreuzvalidierung
def test_blattgroesse_per_kreuzvalidierung_liegt_im_gitter_und_voll_gewachsen_trifft_das_training(xor4):
    w = W.generate("E", 100, 3)
    leaf = M.select_leaf(w.X, w.y, SplitMix64(1))
    assert leaf in M.LEAF_SIZES
    tree, ml = M.fit_cart(w.X, w.y, min_leaf=1)
    assert ml == 1 and tree.n_leaves >= 2
    import kob_tree as T
    assert np.mean(T.predict(tree, w.X) != w.y) == 0.0                       # voll gewachsen: kein Trainingsfehler (keine zwei gleichen Zeilen mit verschiedener Klasse)


def test_run_live_ist_deterministisch_und_gibt_fehler_zwischen_null_und_eins():
    tr, te = W.generate("P", 150, 1), W.generate("P", 600, 700001)
    a, b = M.run_live(tr, te, seed=1), M.run_live(tr, te, seed=1)
    assert a == b
    for v in (a.err_krr, a.err_cart, a.err_cart_full, a.bayes):
        assert 0.0 <= v <= 1.0
    assert a.leaves_full >= a.leaves_cart >= 2 and a.leaf in M.LEAF_SIZES


def test_baeume_sind_gegen_die_skalierung_der_merkmale_unempfindlich():
    """Monotone Umrechnung je Spalte (hier Streckung mit positivem Faktor) ändert die Schnittreihenfolge nicht: gleiche Vorhersagen auf den Testzeilen (Schwellen liegen in der Mitte zweier Werte)."""
    import kob_tree as T
    tr, te = W.generate("F", 200, 2), W.generate("F", 300, 3)
    scale = np.array([1000.0, 0.001, 7.0, 3.0, 11.0, 2.0, 5.0, 0.1, 13.0, 17.0, 19.0])
    t1 = T.grow(tr.X, tr.y, task="class", criterion="gini", min_leaf=3)
    t2 = T.grow(tr.X * scale, tr.y, task="class", criterion="gini", min_leaf=3)
    assert np.array_equal(T.predict(t1, te.X), T.predict(t2, te.X * scale))


def test_gauss_kern_mit_roher_skala_haengt_an_der_spalte_mit_der_groessten_einheit():
    """Lieferungen roh: der Anteil des mittleren Abstandsquadrats aus dem Ladegewicht ist nahezu 1 und die Gram-Matrix praktisch eine Funktion des Gewichts allein."""
    w = W.generate("F", 150, 6)
    gamma = M.gamma_from_scale(M.median_sigma(w.X), 1.0)
    K_alle = M.gauss_gram(w.X, w.X, gamma)
    K_gewicht = M.gauss_gram(w.X[:, [1]], w.X[:, [1]], gamma)
    assert np.mean(np.abs(K_alle - K_gewicht)) < 0.02
    S, _ = M.scale_features(w.X, w.X, "standardisiert")
    K_std = M.gauss_gram(S, S, M.gamma_from_scale(M.median_sigma(S), 1.0))
    K_std_gewicht = M.gauss_gram(S[:, [1]], S[:, [1]], M.gamma_from_scale(M.median_sigma(S), 1.0))
    assert np.mean(np.abs(K_std - K_std_gewicht)) > 0.05
