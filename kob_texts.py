"""Texte der App. Zahlen stehen entweder fest (exakte Größen der Erzeuger, Namen) oder werden aus der eingefrorenen Studie berechnet (`kob_evaluation`), damit Text und Messung nicht auseinanderlaufen.

Regeln: Verdikte nur über die Verdikt-Regel mit Zählung k/20, jede Aussage nennt Welt, Einstellung des Gegners und Zähleinheit; „Kern“ nur als Methode, „Netz“ nur als neuronales Netz,
„Baum“ als Entscheidungsbaum samt Wald und Boosting.
"""

from __future__ import annotations

import numpy as np

import kob_constants as C
import kob_evaluation as E
import kob_frozen as F

INTRO_MD = """
Welches Verfahren bei einer Klassifikation mit zwei Klassen vorn liegt, hängt in diesen **erzeugten Welten** an der **Bauart der Wahrheit**, an den **Koordinaten** und an der **Einstellung des Gegners**,
nicht am Etikett „Kern oder Baum“. Jede Welt hat eine **bekannte Wahrscheinlichkeit** und damit einen berechenbaren **Bayes-Fehler** (den kleinsten erreichbaren Fehler). Verglichen werden die **Gauß-SVM**
(ein Kernverfahren), **abgestimmte Bäume** (Zufallswald und Gradient Boosting), ein **kleines Netz** und die **lineare SVM**, je Welt über 20 Zufalls-Seeds.

Hier meldet kein Löser einen Fehler: die Gauß-SVM läuft sauber durch, und nichts sagt, dass ihr Verdikt am Gegner (Standard oder abgestimmt), an den Koordinaten (Drehung der Merkmale) oder an den Einheiten
(Skalierung) hängt. **Der Vergleich selbst meldet nichts.** Das ist die schwache Form der Lehre dieser Reihe, dass ein Verfahren ohne Fehlermeldung durchlaufen und trotzdem falsch eingestellt sein kann.

Weiter unten: fünf Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden**, der Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und der Expander **📐 Mathematische Formulierung**.
Alle Welten sind erzeugt; es gibt keine reale Datenquelle, und ein Nutzen von Kernverfahren in Logistik, Lager oder Fertigung ist nicht belegt.
"""

HOW_MD = """
1. **Eine Welt festlegen.** Ein Erzeuger gibt Zeilen mit Merkmalen und Klasse; seine Wahrscheinlichkeit p(x) = P(Klasse 1 | x) ist bekannt. Der **Bayes-Fehler** (Fehler der Regel „Klasse 1, wenn p(x) über 1/2“) ist der Boden.
2. **Jedes Verfahren auf dem Training abstimmen.** Gauß-SVM: Skalenfaktor und C per 5-facher Kreuzvalidierung; Wald und Boosting: je ein Gitter, ebenfalls 5-fach. „Abgestimmt“ heißt: Hyperparameter auf dem Training gewählt, nicht am Test; die Auswahl zwischen Wald und Boosting („der Baum“) erfolgt dagegen am Test.
3. **Am Test messen, je Welt über 20 Zufalls-Seeds.** Verdikt: Differenz d = Fehler(Gauß) minus Fehler(Baum) je Zufalls-Seed; **Kern vorn**, wenn der Mittelwert von d unter −2 Standardfehler liegt, **Baum vorn** über +2, sonst **gleichauf**.
4. **Studie und Live-Modell getrennt.** Die Studie (scikit-learn, eingefroren) ist die Messung; live rechnet die App nur ein kleines Paar auf einer Instanz: Kernel Ridge gegen eine einzelne CART.
"""

KERNFRAGEN_MD = """
1. **Welches Verfahren liegt in dieser Welt vorn?** (Experiment 1)
2. **Was ändert sich, wenn der Gegner abgestimmt ist?** (Experiment 2)
3. **Was ändert eine Drehung der Merkmale?** (Experiment 3)
4. **Was verdirbt den Abstand: irrelevante Merkmale und Einheiten?** (Experiment 4)
5. **Welches Verfahren passt zu meiner Aufgabe?** (Experiment 5, ohne neue Rechnung)

Die Verdikte stehen über 20 Zufalls-Seeds je Welt; eine einzelne Instanz belegt keines.
"""


def live_abweichung() -> str:
    """Satz zur Abweichung des Live-Modells von der Studie, aus den Daten berechnet."""
    abw = [w for w in C.LIVE_WORLDS if E.live_deviates(w)]
    if not abw:
        return "Das Live-Verdikt stimmt in allen sieben Welten mit dem Studien-Verdikt überein."
    return ("Das Live-Modell gibt die Studien-Verdikte nur teilweise wieder: in den Welten " + ", ".join(abw) + " weicht sein Verdikt vom Studien-Verdikt ab (eine einzelne CART ist schwächer als Wald und Boosting, "
            "und Kernel Ridge ist nicht die SVM).")


def lernkurve_text(world: str) -> str:
    lc = E.learning_curve(world)
    teile = [f"n = {n}: Gauß-SVM {E.pct(a)}, Wald {E.pct(b)}, Gauß besser in {k} von {lc['seeds']}" for n, a, b, k in zip(lc["n"], lc["rbf"], lc["rf"], lc["rbf_wins"])]
    return ("Eingefrorene Lernkurve (Welt " + world + ", abgestimmter Wald gegen Gauß-SVM mit Faktor auf gamma und C per Kreuzvalidierung, " + str(lc["seeds"]) + " Zufalls-Seeds je Punkt, 4 000 Testpunkte): "
            + "; ".join(teile) + ". Zählung nur über diese Zufalls-Seeds; Boosting und Netz sind in der Lernkurve nicht gemessen.")


def netz_text() -> str:
    conv = E.net_convergence()
    zeilen = ["| Welt | Netz | Gauß-SVM | Netz minus Gauß (Verdikt) | Netz minus abgestimmter Wald (Verdikt) | gleiche Stufe gegen Gauß mit anderen Startwerten |", "|---|---|---|---|---|---|"]
    for w, v in F.teil("netz").items():
        un = int(sum(v["net_unconv"]))
        if un:
            zeilen.append(f"| {w} | nicht konvergiert (in {un} von {len(v['seeds'])} Zufalls-Seeds die Iterationsgrenze ausgeschöpft) | | kein Verdikt | kein Verdikt | |")
            continue
        m, se, lab, e_net, e_g = E.net_vs_gauss(w)
        mr, sr, labr = E.net_vs_forest(w)
        zeilen.append(f"| {w} | {E.pct(e_net)} | {E.pct(e_g)} | {E.pp(m)} (Standardfehler {E.num(100 * se, 1)}): {lab} | {E.pp(mr)} ({E.num(100 * sr, 1)}): {labr} | "
                      f"{'ja' if E.net_other_start_label(w) == lab else 'nein'} |")
    return (
        f"Netz: scikit-learn-MLP mit einer verdeckten Schicht (16 oder 64 Einheiten), L2-Strafe 0,001 oder 0,1, Adam, Grenze {5000} Iterationen, 5-fach abgestimmt (4 Zellen), standardisierte Eingaben, je Welt {C.N_SEEDS} Zufalls-Seeds.\n\n"
        + "\n".join(zeilen)
        + f"\n\n**Konvergenz** (Zähleinheit Zufalls-Seeds und Welten): das gewählte Netz hat die Iterationsgrenze 5 000 in {conv['unconv5000']} von {conv['cases']} Fällen ausgeschöpft "
          f"({conv['msg5000']} Konvergenzmeldungen unter {conv['fits5000']} Anpassungen); mit der Grenze 300 schöpfte es sie in {conv['unconv300']} von {conv['cases']} Fällen aus und wird nicht verwendet. "
          "Das Netz ist eine Architektur mit vier Zellen und einem Optimierer: andere Tiefen, Strafen, früher Halt und die Drehung für das Netz sind nicht gemessen, und das Netz ist in der App kein Regler."
    )


def verdict_tuning(welt, comp, mw, sw, hw, mb, sb, hb):
    a = E.compare_study(welt, "Standard")
    b = E.compare_study(welt, "abgestimmt")
    text = (
        f"Welt {welt} ({a.n} Zufalls-Seeds): Abstimmung bringt dem Wald {E.pp(mw)} (Standardfehler {E.num(100 * sw, 1)}) und dem Boosting {E.pp(mb)} ({E.num(100 * sb, 1)}); "
        f"abgestimmt schlechter als Standard in {E.k_of(hw, a.n)} Zufalls-Seeds (Wald) und {E.k_of(hb, a.n)} (Boosting). Verdikt gegen Standard-Bäume **{a.verdict}** ({a.wins}/{a.n}), "
        f"gegen abgestimmte Bäume **{b.verdict}** ({b.wins}/{b.n}). Positive Werte: abgestimmt besser."
    )
    return "info", text


def tuning_frisch_text() -> str:
    teile = []
    for w in ("A", "B", "D", "P"):
        a1, b1 = E.step_shift(w, "reihe1")
        a2, b2 = E.step_shift(w, "frisch")
        teile.append(f"Welt {w}: erste Reihe {a1} → {b1}, frische Zufalls-Seeds 20 bis 39 {a2} → {b2}")
    shift = [w for w in ("A", "B", "D", "P") if E.shifts_toward_trees(w, "reihe1") and E.shifts_toward_trees(w, "frisch")]
    no_shift = [w for w in ("A", "B", "D", "P") if not E.shifts_toward_trees(w, "reihe1") and not E.shifts_toward_trees(w, "frisch")]
    return ("Verdikt gegen Standard-Bäume → gegen abgestimmte Bäume, in zwei Reihen mit je 20 Zufalls-Seeds (nur A, B, D, P in der zweiten Reihe gemessen): " + "; ".join(teile) + ". "
            f"In den Welten {', '.join(shift) if shift else 'keiner'} verschiebt die Abstimmung das Verdikt in beiden Reihen um mindestens eine Stufe zu den Bäumen, in {', '.join(no_shift) if no_shift else 'keiner'} in keiner. "
            "Welche Stufe erreicht wird, hängt in Welten nahe an der Schwelle von der Reihe ab; eine Zählung „x von y Welten kippen“ wäre eine Schwellenzählung und steht deshalb nirgends.")


def rotation_b_text() -> str:
    rot = E.rotation_summary("B", "Br")
    v0, v1 = rot["verdikt_vorher"], rot["verdikt_nachher"]
    return (f"Dieselbe Frage an Welt B (Wechselwirkung, 4 Merkmale): Gauß-SVM {E.pct(rot['rbf']['vorher'])} → {E.pct(rot['rbf']['nachher'])} (Differenz {E.pp(rot['rbf']['diff'])}, Standardfehler {E.num(100 * rot['rbf']['diff_se'], 1)}), "
            f"abgestimmter Wald {E.pct(rot['rf_t']['vorher'])} → {E.pct(rot['rf_t']['nachher'])}; Verdikt ungedreht {v0.verdict} ({v0.wins}/{v0.n}), gedreht {v1.verdict} ({v1.wins}/{v1.n}). "
            "Die Gauß-SVM schwankt, weil jede Reihe nach der Drehung neu standardisiert wird; die Invarianz des Kerns selbst steht im Unterschied der Gram-Matrizen (exakt bis auf Rundung).")


def noise_c_text() -> str:
    cc = E.compare_study("C", "abgestimmt")
    c2 = E.compare_study("C2", "abgestimmt")
    f8 = E.compare_study("F8", "abgestimmt")
    k_lin, n_lin, g, lin = E.compare_linear("C")
    return (f"Gauß-SVM gegen den besten abgestimmten Baum: Welt C (100 Merkmale, 3 informativ, 100 Zeilen) {E.pp(cc.d_mean)}, Gauß besser in {E.k_of(cc.wins, cc.n)} Zufalls-Seeds; Welt C2 (Wechselwirkung in 50 Merkmalen) {E.pp(c2.d_mean)}, "
            f"{E.k_of(c2.wins, c2.n)}; Lieferungen mit 8 Rauschmerkmalen (F8) {E.pp(f8.d_mean)}, {E.k_of(f8.wins, f8.n)}: **{cc.verdict}**, **{c2.verdict}**, **{f8.verdict}**. "
            "In C2 liegen alle Verfahren nahe 50 %, das Verdikt entsteht dort erst durch die Wahl des besseren Baums am Test (gegen Wald und Boosting einzeln gleichauf). "
            f"In Welt C ist die Gauß-SVM ({E.pct(g)}) gegen die lineare ({E.pct(lin)}) in {E.k_of(k_lin, n_lin)} Zufalls-Seeds besser. Der Anteil des Abstandsquadrats aus k relevanten von p Merkmalen ist k/p.")


def grenzen_md() -> str:
    rand_b = int(np.sum(np.array(F.haupt_world("E")["gb_ntrees"][:20]) == 25))
    c_rand = {w: int(np.sum((np.array(F.haupt_world(w)["rbf_C"][:20]) == 0.1) | (np.array(F.haupt_world(w)["rbf_C"][:20]) == 100))) for w in ("A", "B", "E", "D", "F", "P")}
    cmin, cmax = min(c_rand.values()), max(c_rand.values())
    g_rand = {w: int(np.sum(np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 0.1) | np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 10.0))) for w in ("A", "B", "E", "D", "F", "P")}
    gmin, gmax = min(g_rand.values()), max(g_rand.values())
    beide = {w: int(np.sum(np.isin(np.array(F.haupt_world(w)["rbf_C"][:20]), (0.1, 100.0)) | np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 0.1) | np.isclose(np.array(F.haupt_world(w)["rbf_gfac"][:20]), 10.0))) for w in ("A", "B", "E", "D", "F", "P")}
    bmin, bmax = min(beide.values()), max(beide.values())
    return f"""
| Annahme | Was passiert, wenn sie verletzt ist | Hinweis |
|---|---|---|
| **Alle Welten sind erzeugt** (Erzeugermodelle mit bekannter Wahrscheinlichkeit) | Es gibt keine Echtdaten; ein Verdikt gilt für die gemessene Welt, nicht für eine andere Welt und nicht für Ihre Daten. | Nutzen in Logistik, Lager oder Fertigung nicht belegt |
| **Ein Gitter je Verfahren, nicht bis zum Optimum erweitert** | In Welt E wählt die Kreuzvalidierung beim Boosting den kleinsten Zwischenstand (25 Bäume) in {rand_b} von 20 Zufalls-Seeds, und der Gauß-C-Wert liegt je Welt in {cmin} bis {cmax} von 20 Zufalls-Seeds am Rand des Gitters, der Faktor g auf gamma in {gmin} bis {gmax}, mindestens einer der beiden in {bmin} bis {bmax}. Die Untergrenze wählt nur im selben Gitter und kann Randwahl nicht prüfen. | erweiterte Gitter nicht gemessen |
| **Der Baum ist je Zufalls-Seed am Test gewählt** (bester von Wald und Boosting) | optimistisch für die Bäume: jedes Verdikt „Kern vorn“ ist damit vorsichtig, jedes „Baum vorn“ großzügig; die Spalte „abgestimmt“ ist durch diese Auswahl teilweise am Test gewählt (Auswahlgewinn in C2 und D etwa 0,7 und 0,5 Punkte) | Auswahl nach Kreuzvalidierung statt am Test nicht gemessen |
| **Das Live-Modell ist nicht die Studie** (Kernel Ridge gegen eine CART, nicht SVM gegen Wald und Boosting) | {live_abweichung()} | Experiment 1 zeigt beide |
| **Das Netz hat eine Architektur** (eine verdeckte Schicht, vier Zellen, Adam) | Aussagen über größere oder anders regularisierte Netze sind nicht gemessen; die Drehung wurde für das Netz nicht gemessen. | früher Halt nicht gemessen |
| **Lieferungen und Container sind nach Beschreibung gezogen** (eigene Ziehungen) | Die Dauer der Lieferungen ist aus Schwellen und Stufen gebaut, damit ein Baum etwas zu tun hat: ein Sieg der Bäume in F ist teils eine Eigenschaft des Erzeugers, keine Aussage über Lieferdaten. | wie Welt E |
| **Andere Baumverfahren und TabPFN** | XGBoost, LightGBM, CatBoost, Extra Trees, HistGradientBoosting und TabPFN sind nicht gemessen. | Literatur: Randsätze in Experiment 5 |
| **Eine Plattform** | Alle Zahlen stammen von Windows mit Python 3.14.6 und scikit-learn 1.9.1; Wald, Boosting und Netz hängen mit festem Zufalls-Seed an Version und Betriebssystem (Bänder, nicht exakt); ein Linux-Lauf ist offen. | |
"""


ANNAHMEN_MD = """
**Annahmen.** Zwei Klassen; Welten mit bekannter Wahrscheinlichkeit; 300 bis 400 Trainingszeilen je Welt in der Studie (100 in Welt C); 4 000 frische Testpunkte je Zufalls-Seed; 20 Zufalls-Seeds je Reihe;
Gauß-Kern mit gamma = 1/(2σ²) und σ = Skalenfaktor mal Median der paarweisen Abstände (i < j) der standardisierten Trainingszeilen; Wald mit 200 Bäumen; Verdikt-Regel mit zwei Standardfehlern.

**Grenzen.**
- Ein Verdikt gilt für das gemessene Paar (Gauß-SVM gegen den besten Baum), die genannte Einstellung des Gegners und die Welt; „Kern vorn“ oder „Baum vorn“ ohne Welt gibt es nicht.
- Die Zahl der Zufalls-Seeds ist klein: Welten nahe an der Schwelle (B, D) wechseln die Stufe von Reihe zu Reihe; belastbar ist die Richtung, nicht die Stufe.
- Das Live-Modell rechnet höchstens 600 Zeilen und ist ein anderes Verfahren als die Studie.
- Die Studie wählt Hyperparameter nur auf dem Training (5-fach); die Untergrenze wählt am Test und ist keine erreichbare Einstellung.

**Bekannte Probleme und Unsicherheiten.**
- Die Zahlen sind Neumessungen mit dem Zufallsgenerator der App (SplitMix64); die ursprünglichen Vormessungen mit numpy-Zufall lieferten in einzelnen Welten andere Stufen (README, Abschnitt „Befunde und Korrekturen“).
- Die Anzeige „Live“ ist eine Instanz: ihre Zahlen streuen von Zufalls-Seed zu Zufalls-Seed um mehrere Prozentpunkte und belegen kein Verdikt.
- Die Entscheidungsflächen sind nur für Welten mit zwei Merkmalen gezeichnet.
- Nicht mit echtem Mausklick im Browser getestet: die App ist mit AppTest und einem Start des Servers geprüft.
"""

FORMEL_MD = r"""
**Lesehilfe.** *Gegeben:* Trainingszeilen $(x_i, y_i)$, $i = 1, \dots, n$, mit $y_i \in \{0, 1\}$, gezogen aus einer Welt mit bekannter Wahrscheinlichkeit $p(x) = P(y = 1 \mid x)$. *Gesucht:* der Testfehler jedes Verfahrens
und das Verdikt zwischen Gauß-SVM und bestem Baum. Dieser Block beschreibt das **Live-Modell vollständig** (Kernel-Ridge-Klassifikator und CART); die SVM der Studie ist nur skizziert (ihre Formeln stehen in den Demos zur harten und zur weichen Marge).

| Symbol | Bedeutung |
|---|---|
| $p(x)$ | bekannte Wahrscheinlichkeit der Klasse 1 in der Welt |
| $\varepsilon^*$ | Bayes-Fehler, kleinster erreichbarer Fehler |
| $\sigma$, $\gamma$ | Bandbreite und Kernparameter, $\gamma = 1/(2\sigma^2)$ |
| $f$ | Skalenfaktor: $\sigma = f \cdot \sigma_{\mathrm{med}}$ |
| $\sigma_{\mathrm{med}}$ | Median der paarweisen Abstände $\lVert x_i - x_j \rVert$ mit $i < j$ |
| $\lambda$ | Ridge-Regler des Kernel Ridge (nicht $C$) |
| $C$ | Regler der SVM der Studie (nicht $\lambda$) |
| $d_s$ | Differenz Fehler(Gauß) minus Fehler(Baum) im Zufalls-Seed $s$ |

**Bayes-Fehler.** $\varepsilon^* = \mathbb{E}\,[\min(p(x),\, 1 - p(x))]$, der Fehler der Regel $1[p(x) > 1/2]$.

**Gauß-Kern und Kernel Ridge.** $k(x, z) = \exp(-\gamma \lVert x - z \rVert^2)$, Gram-Matrix $K_{ij} = k(x_i, x_j)$, Ziele $t_i = 2 y_i - 1$,
$$\alpha = (K + \lambda I)^{-1} t, \qquad h(x) = \sum_{i=1}^{n} \alpha_i\, k(x_i, x), \qquad \hat y(x) = 1[h(x) > 0].$$
$f \in \{3{,}16;\ 1{,}83;\ 1;\ 0{,}58;\ 0{,}32\}$ und $\lambda \in \{0{,}01;\ 0{,}1;\ 1\}$ werden per 5-facher Kreuzvalidierung gewählt (Gleichstand: erste Zelle in der Reihenfolge $f$, dann $\lambda$).

**CART.** Gini-Unreinheit $2 q (1 - q)$ mit dem Anteil $q$ der Klasse 1 im Knoten; jeder Knoten wird am Schnitt (Merkmal $j$, Schwelle $s$) mit dem größten Unreinheitsgewinn geteilt, bis er rein ist oder die Mindestblattgröße unterschritten würde;
die Mindestblattgröße wird per 5-facher Kreuzvalidierung aus $\{1, 3, 5, 10, 20\}$ gewählt (Gleichstand: die kleinste).

**Drehung.** Für eine orthogonale Matrix $Q$ gilt $\lVert Qx - Qz \rVert = \lVert x - z \rVert$, also bleibt $K$ gleich; die Schnitte des Baums liegen an den Achsen und ändern sich mit $Q$.

**Abstandsanteil.** Bei $p$ standardisierten Merkmalen ist $\mathbb{E}\,\lVert x - z \rVert^2 = 2p$, davon $2k$ aus $k$ relevanten Merkmalen: Anteil $k/p$.

**Verdikt-Regel.** $\bar d = \tfrac{1}{S}\sum_s d_s$, $\mathrm{SE} = \sqrt{\tfrac{1}{S(S-1)}\sum_s (d_s - \bar d)^2}$ mit $S = 20$ Zufalls-Seeds; Kern vorn, wenn $\bar d < -2\,\mathrm{SE}$; Baum vorn, wenn $\bar d > 2\,\mathrm{SE}$; sonst gleichauf.
Der Baum ist je Zufalls-Seed der bessere von Wald und Boosting am Test, also optimistisch für die Bäume.

**Studie (Skizze).** Gauß-SVM, Zufallswald und Gradient Boosting stammen aus scikit-learn; ihre Gitter und Einstellungen stehen im README. Was hier fehlt: die duale Form der SVM, der Aufbau von Wald und Boosting.

**Literatur.** Wainberg, Alipanahi, Frey (2016), JMLR 17; Fernández-Delgado u. a. (2014), JMLR 15, 3133 bis 3181; McElfresh u. a. (2023), NeurIPS Datasets and Benchmarks; Grinsztajn, Oyallon, Varoquaux (2022), arXiv 2207.08815.
Alle Angaben anhand der Quellen geprüft, soweit in den Randsätzen von Experiment 5 genannt.
"""
