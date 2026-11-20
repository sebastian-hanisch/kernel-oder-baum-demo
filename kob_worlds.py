"""Welten: Erzeugermodelle mit bekannter Wahrscheinlichkeit p(x) = P(y = 1 | x), alle mit SplitMix64 aus `kob_rng.py` gezogen (kein numpy-Zufall).

Eine **Welt** ist hier ein Erzeugermodell, nie ein Datensatz. Alle Welten sind erzeugt; es gibt keine reale Datenquelle.
Die Lieferungen (Welten F und F8) und der Container-Erzeuger (Welt D) folgen der Beschreibung aus `cart-demo` bzw. `verweilzeit-demo`, sind aber **eigene Ziehungen**
(andere Quelle, andere Reihenfolge): ihre Zeilen sind nicht die der Vormessung des Plans und nicht die der beiden Quellen, nur die Verteilung ist dieselbe.

Zufallsquelle: Gleichverteilung bitgleich aus SplitMix64; Normalzahlen als `NormalDist().inv_cdf(random())` (schnell, auf Windows und Linux gleich bis auf die letzte Gleitkommastelle).
Reihenfolge der Ziehungen je Welt: erst alle Merkmale, dann die Etiketten (`random()` gegen p(x)); die Lieferungen ziehen ihre Spalten wie `cart-demo` in fester Reihenfolge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from kob_rng import SplitMix64

_INV = NormalDist().inv_cdf
_TWO_NEG54 = 2.0 ** -54

R0 = math.sqrt(2.0 / math.pi)       # Welt P: Kreisfläche pi r0^2 = halbe Fläche des Quadrats [-1, 1]^2
P_FLIP = 0.1                        # Welt P: Etiketten mit dieser Wahrscheinlichkeit vertauscht
DURATION_NOISE = 7.0                # Lieferungen: Standardabweichung des Zufalls in der Dauer (Minuten)
PROMISE_SLACK = 14.0                # Lieferungen: Puffer in der versprochenen Zeit (Minuten)

# Container-Erzeuger (Beschreibung aus verweilzeit-demo): Anteile der Arten, Kundeneffekte, Streuungen
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
U_SD = 0.30
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
KUNDEN_SEED = 12345

WORLD_NAMES = ("A", "B", "C", "C2", "E", "E2", "D", "F", "F8", "P", "Ri", "Er", "Br")
WORLD_LABELS = {
    "A": "A fast linear (6 Merkmale)",
    "B": "B Wechselwirkung (4 Merkmale, 2 informativ)",
    "C": "C 100 Merkmale, 3 informativ",
    "C2": "C2 Wechselwirkung in 50 Merkmalen",
    "E": "E Schwellenregeln (10 Merkmale, 7 irrelevant)",
    "E2": "E2 Schwellenregel mit 2 Merkmalen (nur live)",
    "D": "D Container-Erzeuger (20 Spalten)",
    "F": "F Lieferungen: zu spät (8 Merkmale + 3 Rauschmerkmale)",
    "F8": "F8 Lieferungen mit 8 Rauschmerkmalen",
    "P": "P Liefergebiet (Kreis mit Etikettenrauschen)",
    "Ri": "Ri Ringe (make_circles, Rauschen 0,1)",
    "Er": "Er wie E, Merkmale zufällig gedreht",
    "Br": "Br wie B, Merkmale zufällig gedreht",
}
N_TRAIN = {"A": 300, "B": 300, "C": 100, "C2": 300, "E": 300, "E2": 300, "D": 400, "F": 400, "F8": 400, "P": 300, "Ri": 100, "Er": 300, "Br": 300}


@dataclass(frozen=True)
class World:
    X: np.ndarray       # (n, d)
    y: np.ndarray       # 0 / 1
    p: np.ndarray       # P(y = 1 | x), exakt aus dem Erzeugermodell

    @property
    def bayes_error_expected(self) -> float:
        """Erwartungswert des Bayes-Fehlers E[min(p, 1 - p)] über die gezogenen Punkte."""
        return float(np.mean(np.minimum(self.p, 1.0 - self.p)))

    def bayes_error_empirical(self) -> float:
        """Fehler der Bayes-Regel 1[p > 1/2] gegen die gezogenen Etiketten."""
        return float(np.mean((self.p > 0.5).astype(int) != self.y))


# ---------------------------------------------------------------- Ziehungen aus SplitMix64
def _u(rng: SplitMix64) -> float:
    """Gleichverteilt in (0, 1]: nie genau 0 (für inv_cdf und Logarithmen)."""
    return rng.random() or _TWO_NEG54


def uniform_array(rng: SplitMix64, shape, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    m = int(np.prod(shape))
    a = np.fromiter((rng.random() for _ in range(m)), float, count=m).reshape(shape)
    return lo + (hi - lo) * a


def normal_array(rng: SplitMix64, shape) -> np.ndarray:
    m = int(np.prod(shape))
    return np.fromiter((_INV(_u(rng)) for _ in range(m)), float, count=m).reshape(shape)


def _normal(rng: SplitMix64) -> float:
    return _INV(_u(rng))


def _expo(rng: SplitMix64) -> float:
    return -math.log(_u(rng))


def gamma_draw(rng: SplitMix64, shape: float, scale: float) -> float:
    """Gamma-Zahl (Marsaglia und Tsang 2000) für shape >= 1; ganzzahlige Formen 1 und 2 über Exponentialzahlen."""
    if shape == 1.0:
        return scale * _expo(rng)
    if shape == 2.0:
        return scale * (_expo(rng) + _expo(rng))
    if shape < 1.0:
        return gamma_draw(rng, shape + 1.0, scale) * _u(rng) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = _normal(rng)
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        if math.log(_u(rng)) < 0.5 * x * x + d - d * v + d * math.log(v):
            return scale * d * v


def beta_draw(rng: SplitMix64, a: float, b: float) -> float:
    g1 = gamma_draw(rng, a, 1.0)
    g2 = gamma_draw(rng, b, 1.0)
    return g1 / (g1 + g2)


def poisson_draw(rng: SplitMix64, lam: float) -> int:
    """Poisson-Zahl nach Knuth (Produkt gleichverteilter Zahlen); für lam <= 30 genau und schnell genug."""
    limit = math.exp(-lam)
    k, prod = 0, 1.0
    while True:
        prod *= rng.random()
        if prod <= limit:
            return k
        k += 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _labels(rng: SplitMix64, p: np.ndarray) -> np.ndarray:
    u = uniform_array(rng, p.shape)
    return (u < p).astype(int)


# ---------------------------------------------------------------- Erzeuger je Welt
def _world_a(rng, n):
    X = normal_array(rng, (n, 6))
    z = 2.0 * (X[:, 0] + X[:, 1] + X[:, 2]) / math.sqrt(3.0) + 0.3 * X[:, 0] * X[:, 1]
    p = sigmoid(z)
    return X, _labels(rng, p), p


def _world_b(rng, n):
    X = normal_array(rng, (n, 4))
    p = sigmoid(3.0 * X[:, 0] * X[:, 1])
    return X, _labels(rng, p), p


def _world_c(rng, n):
    X = normal_array(rng, (n, 100))
    p = sigmoid(2.5 * (X[:, 0] + X[:, 1] + X[:, 2]) / math.sqrt(3.0))
    return X, _labels(rng, p), p


def _world_c2(rng, n):
    X = normal_array(rng, (n, 50))
    p = sigmoid(3.0 * X[:, 0] * X[:, 1])
    return X, _labels(rng, p), p


def rule_e(X):
    """Schwellenregel der Welt E (achsenparallel): ((x1 > 0,3) und (x2 < 0,2)) oder (x3 > 0,5)."""
    return ((X[:, 0] > 0.3) & (X[:, 1] < 0.2)) | (X[:, 2] > 0.5)


def rule_e2(X):
    """Schwellenregel der Welt E2 (zwei Merkmale, achsenparallel): ((x1 > 0,3) und (x2 < 0,2)) oder (x1 < -0,5)."""
    return ((X[:, 0] > 0.3) & (X[:, 1] < 0.2)) | (X[:, 0] < -0.5)


def _world_e(rng, n):
    X = uniform_array(rng, (n, 10), -1.0, 1.0)
    p = np.where(rule_e(X), 0.9, 0.1)
    return X, _labels(rng, p), p


def _world_e2(rng, n):
    X = uniform_array(rng, (n, 2), -1.0, 1.0)
    p = np.where(rule_e2(X), 0.9, 0.1)
    return X, _labels(rng, p), p


def _world_p(rng, n):
    X = uniform_array(rng, (n, 2), -1.0, 1.0)
    inside = (np.sqrt((X ** 2).sum(1)) <= R0).astype(int)
    flips = uniform_array(rng, (n,)) < P_FLIP
    y = np.where(flips, 1 - inside, inside)
    p = np.where(inside == 1, 1.0 - P_FLIP, P_FLIP)
    return X, y, p


RING_NOISE = 0.1
RING_FACTOR = 0.4


def ring_posterior(X, noise=RING_NOISE, factor=RING_FACTOR):
    """P(Klasse 1 | x) der Ringe (außen Radius 1 Klasse 0, innen Radius `factor` Klasse 1, gleich viele je Ring, Gauß-Rauschen `noise` je Koordinate).
    Dichte eines Rings mit Gauß-Rauschen: exp(-(r^2 + R^2)/(2 s^2)) I0(r R / s^2) / (2 pi s^2) mit r = Abstand des Punktes zur Mitte, R = Radius des Rings."""
    rho = np.sqrt((np.asarray(X, float) ** 2).sum(1))
    s2 = noise * noise

    def logf(R):
        return -(rho ** 2 + R * R) / (2 * s2) + np.log(np.i0(rho * R / s2))

    return 1.0 / (1.0 + np.exp(logf(1.0) - logf(factor)))


def _world_ri(rng, n):
    n_in = n // 2
    n_out = n - n_in
    ang = 2.0 * math.pi * uniform_array(rng, (n,))
    rad = np.concatenate([np.full(n_out, 1.0), np.full(n_in, RING_FACTOR)])
    y = np.concatenate([np.zeros(n_out, int), np.ones(n_in, int)])
    noise = normal_array(rng, (n, 2)) * RING_NOISE
    X = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]) + noise
    order = list(range(n))
    rng.shuffle(order)
    order = np.array(order)
    X, y = X[order], y[order]
    return X, y, ring_posterior(X)


# --- Lieferungen (Beschreibung aus cart-demo)
def delivery_duration(X):
    """Dauer in Minuten ohne Zufall: Grundzeit, Stau (Schwelle beim Verkehrsindex, wächst mit der Distanz), Ladezeit je Stopp bei schwerer Ladung, Wetter, Wochenende, Erfahrung."""
    dist, weight, stops, traffic, weather, weekday, _window, years = (X[:, i] for i in range(8))
    base = 25.0 + 0.5 * dist + 1.2 * stops
    jam = 0.9 * dist * np.clip((traffic - 0.6) / 0.4, 0.0, 1.0)
    load = np.where(weight > 800.0, 1.5 * stops, 0.0)
    rain = np.where(weather > 0.65, 0.25 * dist, 0.0)
    weekend = np.where(weekday >= 5, -8.0, 0.0)
    skill = -0.6 * np.minimum(years, 8.0)
    return base + jam + load + rain + weekend + skill


def delivery_promised(X):
    """Versprochene Zeit in Minuten: Grundzeit plus Puffer, bei engem Zeitfenster kleiner."""
    dist, stops, window = X[:, 0], X[:, 2], X[:, 6]
    return 25.0 + 0.5 * dist + 1.2 * stops + PROMISE_SLACK - 10.0 * window


def norm_cdf(z):
    return np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in np.asarray(z, float).ravel()]).reshape(np.shape(z))


def _world_f(rng, n, n_noise=3):
    dist = np.clip([gamma_draw(rng, 2.2, 22.0) for _ in range(n)], 3.0, 160.0)
    weight = uniform_array(rng, (n,), 100.0, 1200.0)
    stops = np.clip([poisson_draw(rng, 12.0) for _ in range(n)], 2, 40).astype(float)
    traffic = np.array([beta_draw(rng, 2.0, 2.0) for _ in range(n)])
    weather = np.array([beta_draw(rng, 1.5, 3.0) for _ in range(n)])
    weekday = np.array([rng.below(7) for _ in range(n)], float)
    window = uniform_array(rng, (n,))
    years = np.floor(np.clip([gamma_draw(rng, 2.0, 3.5) for _ in range(n)], 0.0, 20.0))
    noise = normal_array(rng, (n, 8))                         # immer alle acht gezogen, verwendet werden die ersten n_noise (wie in cart-demo)
    Xb = np.column_stack([dist, weight, stops, traffic, weather, weekday, window, years])
    dur = delivery_duration(Xb) + DURATION_NOISE * normal_array(rng, (n,))
    y = (dur > delivery_promised(Xb)).astype(int)
    p = 1.0 - norm_cdf((delivery_promised(Xb) - delivery_duration(Xb)) / DURATION_NOISE)
    X = np.column_stack([Xb, noise[:, :n_noise]]) if n_noise else Xb
    return X, y, p


# --- Container-Erzeuger (Beschreibung aus verweilzeit-demo), eigene Kundeneffekte
def _kunden_effekte() -> np.ndarray:
    r = SplitMix64(KUNDEN_SEED)
    return U_SD * normal_array(r, (K_KUNDEN,))


UK = _kunden_effekte()


def container_log_params(art, kunde, zoll, we, vorlauf):
    """Bedingte Verteilung der log-Verweilzeit N(mu, s^2) (Beschreibung aus verweilzeit-demo). Vektorisiert."""
    art = np.asarray(art)
    u = UK[np.asarray(kunde)] * np.take(U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * np.asarray(zoll) + 0.30 * np.asarray(we)), 0.6 + 0.85 * np.asarray(vorlauf), 0.3 + 0.8 * np.asarray(vorlauf)],
        default=7.0 * np.exp(0.30 * np.asarray(we)),
    )
    return np.log(med) + u, np.take(S_BASE, art)


def container_threshold() -> float:
    """Schwelle T = Median der Verweilzeit des Stroms (Klasse 1 = länger als T), exakt aus der Merkmalsverteilung: P(Verweilzeit > T) = 1/2, Bisektion auf ln T.
    Die Merkmale sind diskret bis auf den Vorlauf (gleichverteilt): dort Mittelpunktsregel mit 200 Stützstellen."""
    global _T_CACHE
    if _T_CACHE is not None:
        return _T_CACHE
    cells = []   # (Gewicht, art, kunde, zoll, we, vorlauf)
    grid = (np.arange(200) + 0.5) / 200.0
    for art, pa in enumerate(ART_P):
        for kunde in range(K_KUNDEN):
            for zoll, pz in ((0, 1.0 - (P_ZOLL if art == 0 else 0.0)), (1, P_ZOLL if art == 0 else 0.0)):
                if pz == 0.0:
                    continue
                for we, pw in ((0, 5.0 / 7.0), (1, 2.0 / 7.0)):
                    if art == 1:
                        for g in grid:
                            cells.append((pa / K_KUNDEN * pz * pw / 200.0, art, kunde, zoll, we, 1.0 + 9.0 * g))
                    elif art == 2:
                        for g in grid:
                            cells.append((pa / K_KUNDEN * pz * pw / 200.0, art, kunde, zoll, we, 0.5 + 5.5 * g))
                    else:
                        cells.append((pa / K_KUNDEN * pz * pw, art, kunde, zoll, we, 0.0))
    w = np.array([c[0] for c in cells])
    art = np.array([c[1] for c in cells])
    mu, s = container_log_params(art, np.array([c[2] for c in cells]), np.array([c[3] for c in cells]), np.array([c[4] for c in cells]), np.array([c[5] for c in cells]))

    def tail(lt):
        return float(np.sum(w * (1.0 - norm_cdf((lt - mu) / s))))

    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tail(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    _T_CACHE = math.exp(0.5 * (lo + hi))
    return _T_CACHE


_T_CACHE = None


def _world_d(rng, n):
    art = np.empty(n, int)
    kunde = np.empty(n, int)
    zoll = np.zeros(n, int)
    we = np.empty(n, int)
    vorlauf = np.zeros(n)
    gewicht = np.empty(n)
    for i in range(n):
        art[i] = rng.choice(ART_P)
        kunde[i] = rng.below(K_KUNDEN)
        zoll[i] = int(rng.random() < P_ZOLL and art[i] == 0)
        we[i] = int(rng.below(7) >= 5)
        u = rng.random()
        if art[i] == 1:
            vorlauf[i] = 1.0 + 9.0 * u
        elif art[i] == 2:
            vorlauf[i] = 0.5 + 5.5 * u
        gewicht[i] = _normal(rng)
    z = normal_array(rng, (n,))
    mu, s = container_log_params(art, kunde, zoll, we, vorlauf)
    verweil = np.exp(mu + s * z)
    T = container_threshold()
    y = (verweil > T).astype(int)
    p = 1.0 - norm_cdf((math.log(T) - mu) / s)
    oa = np.zeros((n, 4))
    oa[np.arange(n), art] = 1.0
    ok = np.zeros((n, K_KUNDEN))
    ok[np.arange(n), kunde] = 1.0
    X = np.column_stack([oa, ok, zoll, we, vorlauf / 10.0, gewicht])
    return X, y, p


# ---------------------------------------------------------------- Drehung
def random_rotation(d: int, seed: int) -> np.ndarray:
    """Zufällige orthogonale Matrix: Normalzahlen, QR-Zerlegung (Gram-Schmidt), Vorzeichen so, dass die Diagonale von R positiv ist (dann eindeutig)."""
    r = SplitMix64(seed)
    Q, R = np.linalg.qr(normal_array(r, (d, d)))
    return Q * np.sign(np.diag(R))


def rotation_2d(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


_BUILD = {"A": _world_a, "B": _world_b, "C": _world_c, "C2": _world_c2, "E": _world_e, "E2": _world_e2, "P": _world_p, "Ri": _world_ri, "D": _world_d}


def generate(world: str, n: int, seed: int, rot_seed: int | None = None, rot_deg: float = 0.0) -> World:
    """n Zeilen der Welt `world` mit Zufalls-Seed `seed`. Er und Br: dieselbe Welt wie E und B, Merkmale mit der zufälligen Drehung zu `rot_seed` (Standard: `seed`) gedreht;
    für Trainings- und Testzeilen derselben Zufalls-Seed-Zelle muss `rot_seed` gleich sein. `rot_deg` dreht eine 2D-Welt (E2, P) um diesen Winkel in Grad."""
    n = int(n)
    base = world[:-1] if world in ("Er", "Br") else world
    if base not in _BUILD and base not in ("F", "F8"):
        raise ValueError(f"unbekannte Welt: {world}")
    rng = SplitMix64(int(seed))
    if base == "F":
        X, y, p = _world_f(rng, n, 3)
    elif base == "F8":
        X, y, p = _world_f(rng, n, 8)
    else:
        X, y, p = _BUILD[base](rng, n)
    if world in ("Er", "Br"):
        X = X @ random_rotation(X.shape[1], int(seed if rot_seed is None else rot_seed))
    if rot_deg:
        if X.shape[1] != 2:
            raise ValueError("rot_deg gilt nur für Welten mit 2 Merkmalen")
        X = X @ rotation_2d(rot_deg).T
    return World(X, y, p)
