"""Auswertung: die Verdikt-Regel der Demo als reine Funktionen, Zählungen und die Verdikt-Texte der App.

**Verdikt-Regel (eine Festlegung für die ganze Demo):** d = Fehler(Gauß) minus Fehler(Baum) je Zufalls-Seed; **Kern vorn**, wenn der Mittelwert von d unter -2 Standardfehler liegt,
**Baum vorn** über +2 Standardfehler, sonst **gleichauf**. „k/20“ zählt die Zufalls-Seeds, in denen die Gauß-SVM den kleineren Fehler hat. **Der Baum** ist je Zufalls-Seed der bessere von
Wald und Boosting, im Nachhinein am Test gewählt, also optimistisch für die Bäume. Jedes Verdikt gilt für das gemessene Paar, die genannte Einstellung des Gegners und diese Welt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kob_constants as C
import kob_frozen as F


# ---------------------------------------------------------------- Formate (deutsche Dezimalzeichen)
def num(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def snum(x: float, nd: int = 1) -> str:
    """Zahl mit Vorzeichen (+ oder Minuszeichen −) und Dezimalkomma."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl, z. B. 0,2246 -> '22,5 %'."""
    return f"{100.0 * x:.{nd}f} %".replace(".", ",")


def pp(x: float, nd: int = 1, signed: bool = True) -> str:
    """Differenz zweier Anteile in Prozentpunkten, z. B. 0,106 -> '+10,6 Prozentpunkte'."""
    v = 100.0 * x
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace(".", ",").replace("-", "−") + " Prozentpunkte"


def k_of(k: int, n: int) -> str:
    return f"{k} von {n}"


# ---------------------------------------------------------------- Statistik und Regel
def mean_se(v) -> tuple[float, float]:
    """Mittelwert und Standardfehler des Mittels (Stichprobenstandardabweichung mit n - 1)."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def verdict_rule(d) -> str:
    """Verdikt aus den Differenzen d = Fehler(Gauß) minus Fehler(Baum) je Zufalls-Seed: Kern vorn, Baum vorn oder gleichauf."""
    m, se = mean_se(d)
    if math.isnan(se):
        return C.GLEICHAUF
    if m < -C.SE_FACTOR * se:
        return C.KERN_VORN
    if m > C.SE_FACTOR * se:
        return C.BAUM_VORN
    return C.GLEICHAUF


def verdict_rule_se_units(d) -> float:
    """Mittel von d in Standardfehlern (positiv: Baum besser)."""
    m, se = mean_se(d)
    return m / se if se > 1e-12 else 0.0


def count_better(first, second, tol: float = 1e-12) -> int:
    """Zahl der Zufalls-Seeds, in denen `first` den kleineren Fehler hat als `second`."""
    return int(np.sum(np.asarray(first, float) < np.asarray(second, float) - tol))


@dataclass(frozen=True)
class Comparison:
    world: str
    setting: str
    reihe: str
    n: int
    kern: float
    baum: float
    bayes: float
    d_mean: float
    d_se: float
    wins: int            # Zufalls-Seeds, in denen die Gauß-SVM besser ist
    verdict: str

    @property
    def se_units(self) -> float:
        return self.d_mean / self.d_se if self.d_se > 1e-12 else 0.0


def compare_study(world: str, setting: str = "abgestimmt", reihe: str = "reihe1", gauss_key: str = "rbf") -> Comparison:
    """Gauß-SVM (Gitter: Faktor auf gamma und C per Kreuzvalidierung) gegen den besten Baum der Einstellung, je Zufalls-Seed."""
    g = F.haupt_array(world, gauss_key, reihe)
    t = F.best_tree(world, setting, reihe)
    d = g - t
    m, se = mean_se(d)
    return Comparison(world, setting, reihe, len(d), float(g.mean()), float(t.mean()), float(F.haupt_array(world, "bayes", reihe).mean()), m, se, count_better(g, t), verdict_rule(d))


def compare_linear(world: str, reihe: str = "reihe1") -> tuple[int, int, float, float]:
    """(Zufalls-Seeds, in denen die Gauß-SVM besser ist als die lineare, Zahl der Zufalls-Seeds, Fehler Gauß, Fehler linear)."""
    g, lin = F.haupt_array(world, "rbf", reihe), F.haupt_array(world, "lin", reihe)
    return count_better(g, lin), len(g), float(g.mean()), float(lin.mean())


def tuning_gain(world: str, which: str = "rf", reihe: str = "reihe1") -> tuple[float, float, int]:
    """Standard minus abgestimmt (positiv: abgestimmt besser) für Wald ('rf') oder Boosting ('gb'): (Mittel, Standardfehler, Zufalls-Seeds, in denen abgestimmt schlechter ist)."""
    std, tun = F.haupt_array(world, f"{which}_std", reihe), F.haupt_array(world, f"{which}_t", reihe)
    m, se = mean_se(std - tun)
    return m, se, int(np.sum(tun > std + 1e-12))


def step_shift(world: str, reihe: str) -> tuple[str, str]:
    """(Verdikt gegen Standard-Bäume, Verdikt gegen abgestimmte Bäume) in einer Reihe."""
    return compare_study(world, "Standard", reihe).verdict, compare_study(world, "abgestimmt", reihe).verdict


STUFE = {C.KERN_VORN: 0, C.GLEICHAUF: 1, C.BAUM_VORN: 2}


def shifts_toward_trees(world: str, reihe: str) -> bool:
    """Verschiebt die Abstimmung das Verdikt in dieser Reihe um mindestens eine Stufe zu den Bäumen (Kern vorn -> gleichauf -> Baum vorn)?"""
    a, b = step_shift(world, reihe)
    return STUFE[b] > STUFE[a]


# ---------------------------------------------------------------- Netz, Skalierung, Rauschen, Drehung, Ringe, Lernkurve, Live
def net_vs_gauss(world: str) -> tuple[float, float, str, float, float]:
    """Netz (verdeckte Schicht, Grenze 5 000 Iterationen) minus Gauß-SVM: (Mittel, Standardfehler, Verdikt in Netz-Wörtern, Fehler Netz, Fehler Gauß)."""
    t = F.teil("netz")[world]
    net, g = np.array(t["net"], float), np.array(t["rbf"], float)
    m, se = mean_se(net - g)
    lab = "Netz vorn" if m < -C.SE_FACTOR * se else ("Netz hinten" if m > C.SE_FACTOR * se else C.GLEICHAUF)
    return m, se, lab, float(net.mean()), float(g.mean())


def net_vs_forest(world: str) -> tuple[float, float, str]:
    t = F.teil("netz")[world]
    net, rf = np.array(t["net"], float), np.array(t["rf_t"], float)
    m, se = mean_se(net - rf)
    lab = "Netz vorn" if m < -C.SE_FACTOR * se else ("Netz hinten" if m > C.SE_FACTOR * se else C.GLEICHAUF)
    return m, se, lab


def net_other_start_label(world: str) -> str:
    """Stufe des Netzes gegen die Gauß-SVM mit anderen Startwerten (Modell-Seed s + 60000)."""
    t = F.teil("netz")[world]
    d = np.array(t["net_b"], float) - np.array(t["rbf"], float)
    m, se = mean_se(d)
    return "Netz vorn" if m < -C.SE_FACTOR * se else ("Netz hinten" if m > C.SE_FACTOR * se else C.GLEICHAUF)


def net_convergence() -> dict:
    """Konvergenz des gewählten Netzes: (nicht konvergiert, Fälle) bei Grenze 5 000 und bei Grenze 300, Konvergenzmeldungen, Bereich der Iterationen."""
    n = F.teil("netz")
    cases = sum(len(v["seeds"]) for v in n.values())
    return {
        "cases": cases,
        "unconv5000": sum(int(sum(v["net_unconv"])) for v in n.values()),
        "unconv300": sum(int(sum(v["net300_unconv"])) for v in n.values()),
        "unconv300_by_world": {w: int(sum(v["net300_unconv"])) for w, v in n.items()},
        "msg5000": sum(int(sum(v["net_msg"])) for v in n.values()),
        "iter_min": int(min(min(v["net_iter"]) for v in n.values())),
        "iter_max": int(max(max(v["net_iter"]) for v in n.values())),
        "iter_median_by_world": {w: float(np.median(v["net_iter"])) for w, v in n.items()},
        "fits5000": cases * 21,
    }


def scaling_summary() -> dict:
    """Lieferungen in natürlichen Einheiten (Welt F): Gauß-SVM mit Median-Heuristik und C per Kreuzvalidierung roh, standardisiert und auf [0, 1] skaliert."""
    s = F.teil("skalierung")
    roh, std, mm, rohcv = (np.array(s[k], float) for k in ("roh", "std", "minmax", "roh_cv"))
    d = roh - std
    m, se = mean_se(d)
    anteil = np.array(s["anteil"], float).mean(0)
    return {
        "roh": float(roh.mean()), "std": float(std.mean()), "minmax": float(mm.mean()), "roh_cv": float(rohcv.mean()),
        "roh_se": mean_se(roh)[1], "std_se": mean_se(std)[1], "diff": m, "diff_se": se, "roh_schlechter": count_better(std, roh), "n": len(roh),
        "sigma_roh": float(np.mean(s["sigma_roh"])), "sigma_std": float(np.mean(s["sigma_std"])), "anteil": anteil.tolist(), "anteil_max": float(anteil.max()),
        "anteil_argmax": int(anteil.argmax()), "bayes": float(np.mean(s["bayes"])),
    }


DELIVERY_COLUMNS = ("Distanz", "Ladegewicht", "Stopps", "Verkehr", "Wetter", "Wochentag", "Zeitfenster-Enge", "Fahrerjahre")


def noise_loss(base: str = "F", noisy: str = "F8") -> dict:
    """Verlust beim Übergang von 3 auf 8 Rauschmerkmale (F -> F8): Gauß, Wald abgestimmt, Boosting abgestimmt (Mittel in Prozentpunkten, Standardfehler)."""
    out = {}
    for key in ("rbf", "rf_t", "gb_t"):
        a, b = F.haupt_array(base, key), F.haupt_array(noisy, key)
        m, se = mean_se(b - a)
        out[key] = (m, se)
    return out


def rotation_summary(base: str = "E", rotated: str = "Er") -> dict:
    """Welt E gegen Er (und B gegen Br): Fehler und Verdikte ungedreht und gedreht."""
    out = {}
    for key in ("lin", "rbf", "rf_t", "gb_t"):
        a, b = F.haupt_array(base, key), F.haupt_array(rotated, key)
        m, se = mean_se(b - a)
        out[key] = {"vorher": float(a.mean()), "nachher": float(b.mean()), "diff": m, "diff_se": se, "schlechter": count_better(a, b)}
    out["verdikt_vorher"] = compare_study(base, "abgestimmt")
    out["verdikt_nachher"] = compare_study(rotated, "abgestimmt")
    return out


def ringe_summary(reihe: str, gitter: str) -> dict:
    """Ringe, Median-Heuristik (C per Kreuzvalidierung) gegen γ-Faktor und C per Kreuzvalidierung im Gitter: (Mittel, Mittel, besser, schlechter, gleich)."""
    r = F.teil("ringe")[reihe]
    med, cv = np.array(r["med"], float), np.array(r[gitter], float)
    d = cv - med
    return {"med": float(med.mean()), "med_se": mean_se(med)[1], "cv": float(cv.mean()), "cv_se": mean_se(cv)[1], "besser": int((d < -1e-12).sum()), "schlechter": int((d > 1e-12).sum()),
            "gleich": int((np.abs(d) <= 1e-12).sum()), "n": len(d)}


def learning_curve(world: str) -> dict:
    """Lernkurve: Zeilen n, Fehler Gauß-SVM und abgestimmter Wald (Mittel, Standardfehler über Zufalls-Seeds)."""
    lk = F.teil("lernkurve")
    w = lk[world]
    rbf, rf, bayes = np.array(w["rbf"], float), np.array(w["rf_t"], float), np.array(w["bayes"], float)
    return {"n": lk["n"], "rbf": rbf.mean(1), "rbf_se": rbf.std(1, ddof=1) / math.sqrt(rbf.shape[1]), "rf": rf.mean(1), "rf_se": rf.std(1, ddof=1) / math.sqrt(rf.shape[1]),
            "bayes": bayes.mean(1), "rbf_wins": [count_better(rbf[i], rf[i]) for i in range(rbf.shape[0])], "seeds": rbf.shape[1]}


def live_vs_study(world: str, variant: str = "cart") -> dict:
    """Live-Modell (Kernel Ridge gegen eine CART; Blattgröße per Kreuzvalidierung ('cart') oder voll gewachsen ('full')) über 20 Zufalls-Seeds, Verdikt-Regel wie die Studie."""
    t = F.teil("live")[world]
    krr, cart = np.array(t["krr"], float), np.array(t[variant], float)
    d = krr - cart
    m, se = mean_se(d)
    return {"krr": float(krr.mean()), "krr_se": mean_se(krr)[1], "cart": float(cart.mean()), "cart_se": mean_se(cart)[1], "d": m, "d_se": se, "wins": count_better(krr, cart), "n": len(d),
            "verdict": verdict_rule(d), "bayes": float(np.mean(t["bayes"])), "leaves_full": float(np.mean(t["leaves_full"])), "leaf_mean": float(np.mean(t["leaf"]))}


def live_deviates(world: str) -> bool:
    """Weicht das Live-Verdikt (CART mit Blattgröße per Kreuzvalidierung) vom Studien-Verdikt (abgestimmte Bäume, erste Reihe) ab?"""
    return live_vs_study(world, "cart")["verdict"] != compare_study(world, "abgestimmt").verdict


# ---------------------------------------------------------------- Verdikt-Texte (reine Funktionen: (Art, Text), Art ist eine Streamlit-Methode)
KIND = {C.KERN_VORN: "success", C.BAUM_VORN: "warning", C.GLEICHAUF: "info"}


def verdict_study(comp: Comparison) -> tuple[str, str]:
    text = (
        f"**Studie, Welt {comp.world}, Gegner {comp.setting}** ({comp.n} Zufalls-Seeds, Reihe {'der ersten' if comp.reihe == 'reihe1' else 'der frischen'} Zufalls-Seeds): "
        f"Gauß-SVM {pct(comp.kern)}, bester Baum {pct(comp.baum)} (Bayes {pct(comp.bayes)}). Differenz Gauß minus Baum {pp(comp.d_mean)} "
        f"(Standardfehler {num(100 * comp.d_se, 1)}), die Gauß-SVM ist besser in {k_of(comp.wins, comp.n)} Zufalls-Seeds: **{comp.verdict}**."
    )
    return KIND[comp.verdict], text


def verdict_live_study(world: str) -> tuple[str, str]:
    v = live_vs_study(world, "cart")
    text = (
        f"**Live-Modell, Welt {world}** (Kernel Ridge gegen eine CART mit Blattgröße per Kreuzvalidierung, {v['n']} Zufalls-Seeds): Kernel Ridge {pct(v['krr'])}, CART {pct(v['cart'])}, "
        f"Differenz {pp(v['d'])} (Standardfehler {num(100 * v['d_se'], 1)}), Kernel Ridge besser in {k_of(v['wins'], v['n'])}: **{v['verdict']}**."
    )
    return KIND[v["verdict"]], text


def verdict_live_instance(res, world: str, n: int, seed: int) -> tuple[str, str]:
    """Aussage zu einer einzelnen Instanz: ohne Verdikt-Wort (ein Zufalls-Seed entscheidet nichts); nur die Zahlen und der Hinweis."""
    text = (
        f"Diese eine Instanz (Welt {world}, {n} Zeilen, Zufalls-Seed {seed}): Kernel Ridge {pct(res.err_krr)}, CART mit Blattgröße {res.leaf} {pct(res.err_cart)}, voll gewachsene CART "
        f"{pct(res.err_cart_full)} (Bayes {pct(res.bayes)}). Eine einzelne Instanz belegt kein Verdikt: das steht über {C.N_SEEDS} Zufalls-Seeds in den Studien-Zeilen darüber."
    )
    return "info", text


def verdict_scaling(summary: dict) -> tuple[str, str]:
    text = (
        f"Lieferungen in natürlichen Einheiten ({summary['n']} Zufalls-Seeds, Gauß-SVM mit Median-Heuristik und C per Kreuzvalidierung): roh {pct(summary['roh'], 2)}, standardisiert {pct(summary['std'], 2)}, "
        f"auf [0, 1] skaliert {pct(summary['minmax'], 2)}; roh ist schlechter als standardisiert in {k_of(summary['roh_schlechter'], summary['n'])} Zufalls-Seeds "
        f"({pp(summary['diff'])}, Standardfehler {num(100 * summary['diff_se'], 1)}). Die Spalte {DELIVERY_COLUMNS[summary['anteil_argmax']]} bestimmt {pct(summary['anteil_max'])} des mittleren Abstandsquadrats. "
        "Der Löser läuft ohne Meldung durch."
    )
    return "warning", text


def verdict_rotation(rot: dict) -> tuple[str, str]:
    v0, v1 = rot["verdikt_vorher"], rot["verdikt_nachher"]
    text = (
        f"Schwellenregeln, 10 Merkmale, abgestimmte Bäume ({v0.n} Zufalls-Seeds): Gauß-SVM {pct(rot['rbf']['vorher'])} ungedreht und {pct(rot['rbf']['nachher'])} gedreht, "
        f"abgestimmter Wald {pct(rot['rf_t']['vorher'])} und {pct(rot['rf_t']['nachher'])}. Verdikt ungedreht **{v0.verdict}** ({pp(v0.d_mean)}, Gauß besser in {k_of(v0.wins, v0.n)}), "
        f"gedreht **{v1.verdict}** ({pp(v1.d_mean)}, Gauß besser in {k_of(v1.wins, v1.n)})."
    )
    return KIND[v1.verdict], text


def verdict_noise(loss: dict) -> tuple[str, str]:
    g, w, b = loss["rbf"], loss["rf_t"], loss["gb_t"]
    text = (
        f"Von 3 auf 8 Rauschmerkmale (Lieferungen, {C.N_SEEDS} Zufalls-Seeds): Gauß-SVM {pp(g[0])} (Standardfehler {num(100 * g[1], 1)}), abgestimmter Wald {pp(w[0])} ({num(100 * w[1], 1)}), "
        f"abgestimmtes Boosting {pp(b[0])} ({num(100 * b[1], 1)}). Die Richtung (Baum vorn bei Rauschmerkmalen) steht in den Verdikten der Studie; die Größe des Unterschieds im Verlust ist nicht gesichert."
    )
    return "info", text


# ---------------------------------------------------------------- Tabellen (Markdown) für README und App: dieselben Funktionen, damit Text und Messung nicht auseinanderlaufen
def _md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _dse(c: Comparison) -> str:
    return f"{snum(100 * c.d_mean, 1)} ({num(100 * c.d_se, 1)})"


def table_main(reihe: str = "reihe1") -> str:
    """Hauptreihe: Mittel der Testfehler in % über die Zufalls-Seeds der Reihe, Verdikte mit Zählung k/20 (Studie)."""
    rows = []
    for w in F.worlds_with_study():
        if not F.has_reihe(w, reihe):
            continue
        a, b = compare_study(w, "Standard", reihe), compare_study(w, "abgestimmt", reihe)
        rows.append([w, num(100 * a.bayes), num(100 * F.haupt_array(w, "lin", reihe).mean()), num(100 * a.kern),
                     num(100 * F.haupt_array(w, "rf_std", reihe).mean()), num(100 * F.haupt_array(w, "rf_t", reihe).mean()),
                     num(100 * F.haupt_array(w, "gb_std", reihe).mean()), num(100 * F.haupt_array(w, "gb_t", reihe).mean()),
                     f"{a.verdict} ({a.wins})", f"{b.verdict} ({b.wins})"])
    return _md(["Welt", "Bayes", "lineare SVM", "Gauß-SVM", "Wald Standard", "Wald abgestimmt", "Boosting Standard", "Boosting abgestimmt", "Verdikt Standard (k von 20)", "Verdikt abgestimmt (k von 20)"], rows)


def table_differences(reihe: str = "reihe1") -> str:
    """Differenz Gauß minus bester Baum in Prozentpunkten (Standardfehler) und Verdikt, je Einstellung des Gegners."""
    rows = []
    for w in F.worlds_with_study():
        if not F.has_reihe(w, reihe):
            continue
        cs = [compare_study(w, s, reihe) for s in F.TREE_KEYS]
        rows.append([w] + [f"{_dse(c)} {c.verdict}" for c in cs])
    return _md(["Welt", "Standard", "abgestimmt", "am Test gewählt (Untergrenze)"], rows)


def table_tuning() -> str:
    """Wirkung der Abstimmung auf die Bäume: Standard minus abgestimmt in Prozentpunkten (Standardfehler), positiv heißt abgestimmt besser; Zufalls-Seeds, in denen abgestimmt schlechter ist."""
    rows = []
    for w in F.worlds_with_study():
        m1, s1, h1 = tuning_gain(w, "rf")
        m2, s2, h2 = tuning_gain(w, "gb")
        rows.append([w, f"{snum(100 * m1, 2)} ({num(100 * s1, 2)}), schlechter in {h1}", f"{snum(100 * m2, 2)} ({num(100 * s2, 2)}), schlechter in {h2}"])
    return _md(["Welt", "Wald", "Boosting"], rows)


def table_live() -> str:
    """Live-Modell (Kernel Ridge gegen eine CART) neben der Studie, je Welt, Mittel über 20 Zufalls-Seeds."""
    rows = []
    for w in F.teil("live"):
        c = compare_study(w, "abgestimmt")
        lc, lf = live_vs_study(w, "cart"), live_vs_study(w, "full")
        rows.append([w, f"{c.verdict} ({c.wins})", num(100 * lc["krr"], 2), f"{num(100 * lc['cart'], 2)}: {lc['verdict']} ({lc['wins']})", f"{num(100 * lf['cart'], 2)}: {lf['verdict']} ({lf['wins']})"])
    return _md(["Welt", "Studie: Verdikt gegen abgestimmte Bäume (k von 20)", "Live: Kernel Ridge", "Live: CART mit Blattgröße per Kreuzvalidierung", "Live: CART voll gewachsen"], rows)


def table_netz() -> str:
    rows = []
    for w, v in F.teil("netz").items():
        m, se, lab, e_net, e_g = net_vs_gauss(w)
        mr, sr, labr = net_vs_forest(w)
        rows.append([w, num(100 * e_net, 2), num(100 * e_g, 2), f"{snum(100 * m, 2)} ({num(100 * se, 2)}): {lab}", f"{snum(100 * mr, 2)} ({num(100 * sr, 2)}): {labr}",
                     f"{net_other_start_label(w)}"])
    return _md(["Welt", "Netz", "Gauß-SVM", "Netz minus Gauß-SVM (Standardfehler)", "Netz minus abgestimmter Wald", "Stufe gegen Gauß-SVM mit anderen Startwerten"], rows)


def table_ringe() -> str:
    rows = []
    for reihe, label in (("reihe1", "Zufalls-Seeds 0 bis 19"), ("frisch", "frische Zufalls-Seeds 100 bis 139")):
        for g in ("g=0,01 bis 100", "g=0,1 bis 10", "g=0,2 bis 5", "g=0,5 bis 2"):
            r = ringe_summary(reihe, g)
            rows.append([label, g.replace("g=", ""), num(100 * r["med"], 3), num(100 * r["cv"], 3), f"{r['besser']} / {r['schlechter']} / {r['gleich']}"])
    return _md(["Reihe", "Gitter des Faktors g auf gamma", "Median-Heuristik (C per Kreuzvalidierung)", "gamma und C per Kreuzvalidierung", "Kreuzvalidierung besser / schlechter / gleich (Zufalls-Seeds)"], rows)


def table_lernkurve() -> str:
    rows = []
    for w in F.teil("lernkurve"):
        if w == "n":
            continue
        lc = learning_curve(w)
        rows.append([w] + [f"{num(100 * a, 1)} / {num(100 * b, 1)} ({k})" for a, b, k in zip(lc["rbf"], lc["rf"], lc["rbf_wins"])])
    return _md(["Welt"] + [f"n = {n}" for n in F.teil("lernkurve")["n"]], rows)
