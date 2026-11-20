"""Welten: Form, Determinismus, Eigenschaften der Erzeuger und Bayes-Fehler gegen einen anderen Rechenweg (Quadratur, geschlossene Formel).

Zufallsergebnisse als Band mit Standardfehler gegen die **exakte** Größe (4,5 Standardfehler). Lieferungen und Container sind nach Beschreibung gezogen (eigene Ziehungen),
darum prüfen die Tests ihre gemessenen Eigenschaften als Band und nicht Zeilen einer Quelle."""

import math

import numpy as np
import pytest

import kob_rng
import kob_worlds as W

BAND = 4.5


def _band(sample, exact):
    sample = np.asarray(sample, float)
    se = sample.std(ddof=1) / math.sqrt(len(sample))
    return abs(sample.mean() - exact) <= BAND * se


@pytest.mark.parametrize("world,d", [("A", 6), ("B", 4), ("C", 100), ("C2", 50), ("E", 10), ("E2", 2), ("D", 20), ("F", 11), ("F8", 16), ("P", 2), ("Ri", 2), ("Er", 10), ("Br", 4)])
def test_form_etiketten_und_wahrscheinlichkeiten(world, d):
    w = W.generate(world, 120, 5)
    assert w.X.shape == (120, d) and w.y.shape == (120,) and w.p.shape == (120,)
    assert set(np.unique(w.y)) <= {0, 1}
    assert np.all((w.p >= 0.0) & (w.p <= 1.0))
    assert np.all(np.isfinite(w.X))


@pytest.mark.parametrize("world", W.WORLD_NAMES)
def test_gleicher_zufalls_seed_gleiche_welt_anderer_andere(world):
    a, b, c = W.generate(world, 60, 9), W.generate(world, 60, 9), W.generate(world, 60, 10)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and np.array_equal(a.p, b.p)
    assert not np.array_equal(a.X, c.X)


def test_unbekannte_welt_wird_abgelehnt():
    with pytest.raises(ValueError):
        W.generate("Z", 10, 1)


# ---------------------------------------------------------------- Bayes-Fehler
def test_bayes_fehler_der_welten_e_e2_und_p_ist_genau_ein_zehntel():
    """p nimmt nur die Werte 0,1 und 0,9 an: E[min(p, 1 - p)] = 1/10 exakt; die Bayes-Regel macht auf Ziehungen den Fehler 1/10 im Band."""
    for world in ("E", "E2", "P"):
        w = W.generate(world, 120000, 4)
        assert set(np.round(np.unique(w.p), 12)) == {0.1, 0.9}
        assert w.bayes_error_expected == pytest.approx(0.1, abs=1e-12)
        wrong = ((w.p > 0.5).astype(int) != w.y).astype(float)
        assert _band(wrong, 0.1)


def test_welt_p_kreisflaeche_ist_die_haelfte_des_quadrats():
    assert math.pi * W.R0 ** 2 == pytest.approx(0.5 * 4.0, abs=1e-15)
    w = W.generate("P", 150000, 3)
    inside = np.sqrt((w.X ** 2).sum(1)) <= W.R0
    assert _band(inside.astype(float), 0.5)


def test_bayes_fehler_welt_b_gegen_zwei_quadraturen_und_ziehung():
    """E[sigma(-|3 x1 x2|)] mit unabhängigen Standardnormalzahlen x1, x2: Quadratur 1 (Dichte des Produkts K0), Quadratur 2 (Polarkoordinaten), Ziehung aus der Welt B."""
    scipy_special = pytest.importorskip("scipy.special")
    scipy_integrate = pytest.importorskip("scipy.integrate")
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    q1 = 2.0 / math.pi * scipy_integrate.quad(lambda z: sig(-3.0 * z) * scipy_special.k0(z), 0, 40, limit=400)[0]
    # Quadratur 2: Polarkoordinaten x = r cos t, y = r sin t im ersten Viertel (mal 4): x y = r^2 sin(2t) / 2, Dichte exp(-r^2 / 2) / (2 pi)
    q2 = 4.0 * scipy_integrate.dblquad(lambda r, t: sig(-1.5 * r * r * math.sin(2 * t)) * math.exp(-r * r / 2) / (2 * math.pi) * r, 0, math.pi / 2, 0, 12)[0]
    assert q1 == pytest.approx(0.243904, abs=2e-6) and q2 == pytest.approx(q1, abs=1e-6)
    w = W.generate("B", 200000, 6)
    assert _band(np.minimum(w.p, 1 - w.p), q1)


def test_welt_a_und_c_wahrscheinlichkeit_ist_sigmoid_der_logit_funktion():
    w = W.generate("A", 50, 2)
    z = 2.0 * (w.X[:, 0] + w.X[:, 1] + w.X[:, 2]) / math.sqrt(3.0) + 0.3 * w.X[:, 0] * w.X[:, 1]
    assert np.allclose(w.p, 1.0 / (1.0 + np.exp(-z)), atol=1e-15)
    c = W.generate("C", 50, 2)
    assert np.allclose(c.p, 1.0 / (1.0 + np.exp(-2.5 * (c.X[:, 0] + c.X[:, 1] + c.X[:, 2]) / math.sqrt(3.0))), atol=1e-15)
    # Merkmale sind Standardnormalzahlen: Mittel 0, Streuung 1 im Band
    big = W.generate("A", 30000, 1)
    assert all(abs(big.X[:, j].mean()) < BAND / math.sqrt(30000) for j in range(6))


def test_etiketten_folgen_der_wahrscheinlichkeit_kalibrierung_je_zehntel_im_band():
    """Welt B: in jedem Zehntel der Wahrscheinlichkeit liegt der Anteil der Klasse 1 im Band um die mittlere Wahrscheinlichkeit."""
    w = W.generate("B", 120000, 8)
    order = np.argsort(w.p)
    for chunk in np.array_split(order, 10):
        p_bar = w.p[chunk].mean()
        se = math.sqrt(p_bar * (1 - p_bar) / len(chunk))
        assert abs(w.y[chunk].mean() - p_bar) <= BAND * se


# ---------------------------------------------------------------- Lieferungen
def test_lieferungen_bereiche_und_verteilungen_als_band():
    w = W.generate("F", 60000, 21)
    X = w.X
    assert X[:, 0].min() >= 3.0 and X[:, 0].max() <= 160.0                 # Distanz (km), begrenzt
    assert X[:, 1].min() >= 100.0 and X[:, 1].max() <= 1200.0              # Ladegewicht (kg)
    assert X[:, 2].min() >= 2 and X[:, 2].max() <= 40 and np.all(X[:, 2] == np.round(X[:, 2]))
    assert 0.0 < X[:, 3].min() and X[:, 3].max() < 1.0 and 0.0 < X[:, 4].min() and X[:, 4].max() < 1.0
    assert set(np.unique(X[:, 5])) == set(range(7))                        # Wochentag 0 bis 6
    assert X[:, 6].min() >= 0.0 and X[:, 6].max() < 1.0
    assert X[:, 7].min() >= 0.0 and X[:, 7].max() <= 20.0 and np.all(X[:, 7] == np.round(X[:, 7]))
    # Gamma(2,2; 22): Mittel 48,4, ohne Begrenzung; mit Begrenzung auf 3 bis 160 nur wenig kleiner
    se = X[:, 0].std(ddof=1) / math.sqrt(len(X))
    assert abs(X[:, 0].mean() - 2.2 * 22.0) <= 0.5 + BAND * se
    assert _band(X[:, 1], 650.0)                                           # Gleichverteilung 100 bis 1200
    assert _band(X[:, 2], 12.0)                                            # Poisson 12 (die Begrenzung 2 bis 40 trifft fast nie)
    assert _band(X[:, 3], 0.5) and _band(X[:, 4], 1.5 / 4.5) and _band(X[:, 6], 0.5)   # Beta(2,2), Beta(1,5; 3), Gleichverteilung
    assert _band(X[:, 5], 3.0)                                             # Wochentag gleichverteilt 0 bis 6
    for j in range(8, 11):
        assert _band(X[:, j], 0.0) and abs(X[:, j].std(ddof=1) - 1.0) < 0.02


def test_lieferungen_ohne_rauschmerkmale_der_welt_f_und_f8_teilen_ihre_ziehungen():
    """Wie in cart-demo werden immer alle acht Rauschspalten gezogen: F (3) und F8 (8) haben dieselben ersten 11 Spalten, dieselben Etiketten und Wahrscheinlichkeiten."""
    a, b = W.generate("F", 300, 4), W.generate("F8", 300, 4)
    assert np.array_equal(a.X, b.X[:, :11]) and np.array_equal(a.y, b.y) and np.array_equal(a.p, b.p)


def test_lieferungen_zu_spaet_anteil_und_bayes_fehler_gegen_geschlossene_formel():
    """P(zu spät | x) = 1 - Phi((versprochen - Dauer) / 7) ist geschlossen: Anteil zu spät gegen mittlere Wahrscheinlichkeit, Kalibrierung je Zehntel, Bayes-Fehler E[min(p, 1 - p)] gegen die Regel auf Ziehungen."""
    w = W.generate("F", 150000, 31)
    assert _band(w.y - w.p, 0.0)
    order = np.argsort(w.p)
    for chunk in np.array_split(order, 10):
        p_bar = w.p[chunk].mean()
        se = math.sqrt(p_bar * (1 - p_bar) / len(chunk))
        assert abs(w.y[chunk].mean() - p_bar) <= BAND * se
    regel_fehler = ((w.p > 0.5).astype(int) != w.y).astype(float)
    assert _band(regel_fehler, float(np.mean(np.minimum(w.p, 1 - w.p))))
    assert 0.10 < float(np.mean(np.minimum(w.p, 1 - w.p))) < 0.14            # Größenordnung: etwa 0,117


def test_lieferungen_ladegewicht_bestimmt_fast_den_ganzen_abstand_in_natuerlichen_einheiten():
    """Mittleres Abstandsquadrat je Spalte: das Ladegewicht (kg) bestimmt fast alles, die Distanz etwa ein Hundertstel, der Rest zusammen weniger als ein Tausendstel (Band)."""
    w = W.generate("F", 400, 5)
    d2 = ((w.X[:, None, :] - w.X[None, :, :]) ** 2).mean((0, 1))
    share = d2 / d2.sum()
    assert 0.97 < share[1] < 0.995 and 0.004 < share[0] < 0.03 and share[2:].sum() < 0.002


# ---------------------------------------------------------------- Container
def test_container_spalten_und_haeufigkeiten_als_band():
    w = W.generate("D", 30000, 2)
    X = w.X
    assert np.all(X[:, :4].sum(1) == 1) and np.all(X[:, 4:16].sum(1) == 1)    # art und kunde als One-Hot
    for j, p in enumerate(W.ART_P):
        assert _band(X[:, j], p)
    for k in range(W.K_KUNDEN):
        assert _band(X[:, 4 + k], 1.0 / W.K_KUNDEN)
    zoll = X[:, 16]
    assert np.all(zoll[X[:, 0] == 0] == 0)                                    # Zollhalt nur bei Import (Spalte 0 = Import)
    assert _band(zoll[X[:, 0] == 1], W.P_ZOLL)
    assert _band(X[:, 17], 2.0 / 7.0)                                         # Wochenende
    assert _band(X[X[:, 1] == 1][:, 18] * 10.0, 5.5)                          # Vorlauf Export gleichverteilt 1 bis 10
    assert np.all(X[(X[:, 0] == 1) | (X[:, 3] == 1)][:, 18] == 0.0)           # Import und Leer: kein Vorlauf
    assert _band(X[:, 19], 0.0)


def test_container_schwelle_ist_der_median_der_verweilzeit():
    """Klasse 1 = Verweilzeit über der Schwelle T; T ist der Median des Stroms: Anteil Klasse 1 gleich 1/2 im Band, mittlere Wahrscheinlichkeit gleich 1/2."""
    w = W.generate("D", 100000, 12)
    assert _band(w.y.astype(float), 0.5)
    assert abs(float(w.p.mean()) - 0.5) < 5e-3
    assert 3.0 < W.container_threshold() < 5.0


def test_container_bayes_fehler_gegen_unabhaengige_zellenrechnung():
    """E[min(p, 1 - p)] gegen eine zweite, getrennt geschriebene Rechnung über alle Zellen (art, kunde, zoll, wochenende) mit 40 Stützstellen im Vorlauf."""
    T = W.container_threshold()
    total = 0.0
    for art, pa in enumerate(W.ART_P):
        for kunde in range(W.K_KUNDEN):
            for zoll, pz in ((0, 1 - W.P_ZOLL if art == 0 else 1.0), (1, W.P_ZOLL if art == 0 else 0.0)):
                for we, pw in ((0, 5 / 7), (1, 2 / 7)):
                    vor = [(0.0, 1.0)] if art in (0, 3) else [((1 + 9 * (i + 0.5) / 40) if art == 1 else (0.5 + 5.5 * (i + 0.5) / 40), 1 / 40) for i in range(40)]
                    for v, pv in vor:
                        mu, s = W.container_log_params(np.array(art), np.array(kunde), np.array(zoll), np.array(we), np.array(v))
                        p = 1.0 - 0.5 * math.erfc(-((math.log(T) - float(mu)) / float(s)) / math.sqrt(2))
                        total += pa / W.K_KUNDEN * pz * pw * pv * min(p, 1 - p)
    big = W.generate("D", 200000, 14)
    assert _band(np.minimum(big.p, 1 - big.p), total)


def test_container_kundeneffekte_sind_zwoelf_normalzahlen_mit_streuung_0_3():
    assert W.UK.shape == (12,)
    assert 0.1 < W.UK.std() < 0.6


# ---------------------------------------------------------------- Ringe
def test_ringe_posterior_ist_kalibriert_und_bayes_fehler_klein():
    w = W.generate("Ri", 120000, 3)
    assert _band(w.y - w.p, 0.0)
    assert float(np.mean(np.minimum(w.p, 1 - w.p))) < 0.003                    # Größenordnung 0,0013 (Rauschen 0,1, Radienverhältnis 0,4)
    assert w.y.mean() == pytest.approx(0.5, abs=1e-12)                         # gleich viele je Ring


# ---------------------------------------------------------------- Drehung
def test_zufaellige_drehung_ist_orthogonal_und_eindeutig():
    for d in (2, 4, 10):
        Q = W.random_rotation(d, 17)
        assert np.allclose(Q.T @ Q, np.eye(d), atol=1e-12) and abs(abs(np.linalg.det(Q)) - 1.0) < 1e-12
        assert np.array_equal(Q, W.random_rotation(d, 17)) and not np.array_equal(Q, W.random_rotation(d, 18))


def test_gedrehte_welt_hat_dieselben_abstaende_und_dieselben_etiketten():
    base, rot = W.generate("E", 80, 5), W.generate("Er", 80, 5, rot_seed=9005)
    assert np.array_equal(base.y, rot.y) and np.array_equal(base.p, rot.p)
    d0 = ((base.X[:, None, :] - base.X[None, :, :]) ** 2).sum(2)
    d1 = ((rot.X[:, None, :] - rot.X[None, :, :]) ** 2).sum(2)
    assert np.allclose(d0, d1, atol=1e-12) and not np.allclose(base.X, rot.X)
    # Trainings- und Testzeilen derselben Zufalls-Seed-Zelle brauchen dieselbe Drehung: rot_seed ist ein eigener Parameter
    assert np.allclose(W.generate("Er", 10, 1, rot_seed=9005).X @ W.random_rotation(10, 9005).T, W.generate("E", 10, 1).X, atol=1e-12)


def test_drehung_2d_um_winkel_wirkt_nur_auf_zwei_merkmale_und_ist_umkehrbar():
    base = W.generate("E2", 50, 3)
    rot = W.generate("E2", 50, 3, rot_deg=30.0)
    assert np.allclose(rot.X @ W.rotation_2d(30.0), base.X, atol=1e-12)
    with pytest.raises(ValueError):
        W.generate("E", 10, 1, rot_deg=10.0)


# ---------------------------------------------------------------- Verteilungen der Hilfsziehungen
def test_gamma_beta_poisson_ziehungen_haben_die_richtigen_momente():
    r = kob_rng.SplitMix64(77)
    g = np.array([W.gamma_draw(r, 2.2, 22.0) for _ in range(30000)])
    assert _band(g, 48.4) and abs(g.var(ddof=1) - 2.2 * 22.0 ** 2) / (2.2 * 22.0 ** 2) < 0.06
    b1 = np.array([W.beta_draw(r, 2.0, 2.0) for _ in range(30000)])
    b2 = np.array([W.beta_draw(r, 1.5, 3.0) for _ in range(30000)])
    assert _band(b1, 0.5) and _band(b2, 1.5 / 4.5)
    pz = np.array([W.poisson_draw(r, 12.0) for _ in range(30000)], float)
    assert _band(pz, 12.0) and abs(pz.var(ddof=1) - 12.0) < 0.5
    e = np.array([W.gamma_draw(r, 2.0, 3.5) for _ in range(30000)])
    assert _band(e, 7.0)
    small = np.array([W.gamma_draw(r, 0.5, 1.0) for _ in range(20000)])
    assert _band(small, 0.5)
