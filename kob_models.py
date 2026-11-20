"""Live-Modell der App (nur numpy, kein scikit-learn): Kernel-Ridge-Klassifikator mit Gauß-Kern gegen eine einzelne CART.

Das Live-Modell ist **nicht** die Studie: die Studie rechnet die scikit-learn-Gauß-SVM gegen den besseren von abgestimmtem Wald und abgestimmtem Boosting (eingefroren in
`kob_vormessung.json`). Hier gilt: Ziele -1 und +1, Gauß-Kern k(x, z) = exp(-gamma |x - z|^2) mit gamma = 1/(2 sigma^2), sigma = f * sigma_med (Skalenfaktor f), Regler lambda (Ridge);
(f, lambda) per 5-facher Kreuzvalidierung auf dem Training. Die CART kommt aus `kob_tree.py` (Kopie von `cart_algorithm.py` aus `cart-demo`), ihre Mindestblattgröße per 5-facher Kreuzvalidierung.

Jede Einheit ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); Zufall nur über den übergebenen SplitMix64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import kob_tree as T
from kob_rng import SplitMix64

# Skalenfaktoren f = g^(-1/2) zu den Faktoren g = 0,1; 0,3; 1; 3; 10 auf gamma der Vormessung (gleiches Gitter wie die Gauß-SVM der Studie)
GAMMA_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)
SCALE_FACTORS = tuple(g ** -0.5 for g in GAMMA_FACTORS)
LAMBDAS = (0.01, 0.1, 1.0)
LEAF_SIZES = (1, 3, 5, 10, 20)
CV_FOLDS = 5
SCALINGS = ("roh", "standardisiert", "[0, 1]")


# ---------------------------------------------------------------- Einheiten
def scale_features(Xtr: np.ndarray, Xte: np.ndarray, mode: str = "standardisiert"):
    """Merkmale auf der Skala des Trainings: 'roh' (unverändert), 'standardisiert' (Mittel 0, Standardabweichung 1; konstante Spalte bleibt 0) oder '[0, 1]' (Minimum 0, Maximum 1)."""
    if mode == "roh":
        return Xtr.copy(), Xte.copy()
    if mode == "standardisiert":
        m, s = Xtr.mean(0), Xtr.std(0)
        s = np.where(s < 1e-12, 1.0, s)
        return (Xtr - m) / s, (Xte - m) / s
    if mode == "[0, 1]":
        lo, hi = Xtr.min(0), Xtr.max(0)
        span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        return (Xtr - lo) / span, (Xte - lo) / span
    raise ValueError(f"unbekannte Skalierung: {mode}")


def sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix der Abstandsquadrate |a_i - b_j|^2 (nie negativ)."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def median_sigma(X: np.ndarray) -> float:
    """Median der paarweisen Abstände i < j ohne Diagonale (Konvention der Linie; Handbeispiel: Punkte 0, 1, 3, 6 geben 3, mit Diagonale 5/2)."""
    n = len(X)
    if n < 2:
        raise ValueError("Median-Heuristik braucht mindestens 2 Zeilen")
    iu = np.triu_indices(n, 1)
    return float(np.median(np.sqrt(sq_dists(X, X)[iu])))


def column_shares(X: np.ndarray, rows: int = 200) -> np.ndarray:
    """Anteil jeder Spalte am mittleren Abstandsquadrat (Mittel über alle Paare der ersten `rows` Zeilen); die Anteile summieren sich zu 1. Der Gauß-Kern sieht nur diese Abstände."""
    A = np.asarray(X, float)[:rows]
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).mean((0, 1))
    return d2 / d2.sum()


def relevant_share(X: np.ndarray, k_rel: int, rows: int = 200) -> float:
    """Anteil des mittleren Abstandsquadrats, der aus den ersten `k_rel` Spalten stammt; bei p standardisierten Merkmalen mit k relevanten ist der Erwartungswert k/p."""
    return float(column_shares(X, rows)[:k_rel].sum())


def gamma_from_scale(sigma_med: float, f: float) -> float:
    """gamma = 1/(2 sigma^2) mit sigma = f * sigma_med. Sind alle Trainingszeilen gleich (sigma_med = 0), gilt sigma_med = 1 (sonst wäre gamma unendlich)."""
    if sigma_med <= 0.0:
        sigma_med = 1.0
    return 1.0 / (2.0 * (f * sigma_med) ** 2)


def gauss_gram(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * sq_dists(A, B))


def krr_solve(K: np.ndarray, t: np.ndarray, lam: float) -> np.ndarray:
    """Koeffizienten (K + lambda I)^-1 t des Kernel Ridge (symmetrisch positiv definit für lambda > 0, Cholesky)."""
    L = np.linalg.cholesky(K + lam * np.eye(len(K)))
    return np.linalg.solve(L.T, np.linalg.solve(L, t))


def make_folds(n: int, k: int, rng: SplitMix64) -> list[np.ndarray]:
    """Zufällige Aufteilung der Zeilen 0..n-1 in k annähernd gleich große Falten (SplitMix64-Mischen)."""
    order = list(range(n))
    rng.shuffle(order)
    return [np.array(sorted(f), int) for f in np.array_split(np.array(order, int), k)]


@dataclass(frozen=True)
class KrrModel:
    f: float            # Skalenfaktor
    lam: float
    gamma: float
    sigma_med: float
    coef: np.ndarray
    Xtr: np.ndarray
    cv_errors: int      # falsch klassifizierte Zeilen in der Kreuzvalidierung (alle Falten zusammen)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return gauss_gram(X, self.Xtr, self.gamma) @ self.coef

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision(X) > 0.0).astype(int)


def fit_krr(Xs: np.ndarray, y: np.ndarray, f: float, lam: float, sigma_med: float | None = None) -> KrrModel:
    """Kernel-Ridge-Klassifikator mit festem (f, lambda) auf den (bereits skalierten) Zeilen Xs."""
    sm = median_sigma(Xs) if sigma_med is None else sigma_med
    g = gamma_from_scale(sm, f)
    t = 2.0 * np.asarray(y, float) - 1.0
    return KrrModel(f, lam, g, sm, krr_solve(gauss_gram(Xs, Xs, g), t, lam), Xs, 0)


def select_krr(Xs: np.ndarray, y: np.ndarray, rng: SplitMix64, factors=SCALE_FACTORS, lams=LAMBDAS, k: int = CV_FOLDS) -> KrrModel:
    """(f, lambda) mit den wenigsten Kreuzvalidierungsfehlern (Gleichstand: erste Zelle in der Reihenfolge f, dann lambda), danach Anpassung auf allen Zeilen."""
    n = len(y)
    if n < 2 * k:
        raise ValueError(f"zu wenige Zeilen für {k}-fache Kreuzvalidierung: {n}")
    t = 2.0 * np.asarray(y, float) - 1.0
    folds = make_folds(n, k, rng)
    sm = median_sigma(Xs)
    best = (None, None, None)
    for f in factors:
        g = gamma_from_scale(sm, f)
        Kfull = gauss_gram(Xs, Xs, g)
        for lam in lams:
            err = 0
            for fold in folds:
                tr = np.setdiff1d(np.arange(n), fold)
                a = krr_solve(Kfull[np.ix_(tr, tr)], t[tr], lam)
                err += int(np.sum((Kfull[np.ix_(fold, tr)] @ a > 0.0) != (np.asarray(y)[fold] == 1)))
            if best[0] is None or err < best[0]:
                best = (err, f, lam)
    err, f, lam = best
    m = fit_krr(Xs, y, f, lam, sm)
    return KrrModel(m.f, m.lam, m.gamma, m.sigma_med, m.coef, m.Xtr, int(err))


def select_leaf(X: np.ndarray, y: np.ndarray, rng: SplitMix64, leaves=LEAF_SIZES, k: int = CV_FOLDS) -> int:
    """Mindestblattgröße der CART mit den wenigsten Kreuzvalidierungsfehlern (Gleichstand: die kleinste Blattgröße)."""
    n = len(y)
    if n < 2 * k:
        raise ValueError(f"zu wenige Zeilen für {k}-fache Kreuzvalidierung: {n}")
    folds = make_folds(n, k, rng)
    best = (None, None)
    for ml in leaves:
        err = 0
        for fold in folds:
            tr = np.setdiff1d(np.arange(n), fold)
            tree = T.grow(X[tr], y[tr], task="class", criterion="gini", min_leaf=ml)
            err += int(np.sum(T.predict(tree, X[fold]) != y[fold]))
        if best[0] is None or err < best[0]:
            best = (err, ml)
    return best[1]


def fit_cart(X: np.ndarray, y: np.ndarray, rng: SplitMix64 | None = None, min_leaf: int | None = None):
    """CART (Gini). Ohne `min_leaf`: Blattgröße per Kreuzvalidierung (Standard der Live-Anzeige); mit `min_leaf = 1`: voll gewachsen. Gibt (Baum, Blattgröße) zurück."""
    if min_leaf is None:
        min_leaf = select_leaf(X, y, rng if rng is not None else SplitMix64(0))
    return T.grow(X, y, task="class", criterion="gini", min_leaf=int(min_leaf)), int(min_leaf)


# ---------------------------------------------------------------- Live-Paar auf einer Instanz
@dataclass(frozen=True)
class LiveResult:
    err_krr: float
    err_cart: float           # CART mit Blattgröße per Kreuzvalidierung
    err_cart_full: float      # voll gewachsene CART (Blattgröße 1)
    f: float
    lam: float
    sigma_med: float
    leaf: int
    leaves_cart: int
    leaves_full: int
    bayes: float


def run_live(world_train, world_test, scaling: str = "standardisiert", seed: int = 0) -> LiveResult:
    """Kernel Ridge (f, lambda per Kreuzvalidierung) und CART (Blattgröße per Kreuzvalidierung, dazu voll gewachsen) auf einer Instanz; Fehler auf den Testzeilen."""
    Xtr, ytr, Xte, yte = world_train.X, world_train.y, world_test.X, world_test.y
    Str, Ste = scale_features(Xtr, Xte, scaling)
    Str, Ste = np.nan_to_num(Str), np.nan_to_num(Ste)
    krr = select_krr(Str, ytr, SplitMix64(seed + 31))
    e_krr = float(np.mean(krr.predict(Ste) != yte))
    tree, leaf = fit_cart(Xtr, ytr, SplitMix64(seed + 77))
    e_cart = float(np.mean(T.predict(tree, Xte) != yte))
    full = T.grow(Xtr, ytr, task="class", criterion="gini", min_leaf=1)
    e_full = float(np.mean(T.predict(full, Xte) != yte))
    return LiveResult(e_krr, e_cart, e_full, krr.f, krr.lam, krr.sigma_med, leaf, tree.n_leaves, full.n_leaves, world_test.bayes_error_empirical())
