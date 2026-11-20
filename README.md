# Kernel oder Baum oder Netz: welches Verfahren, auf welcher Welt, gegen welchen Gegner – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kernel-oder-baum-demo.streamlit.app/)** -->

Achte und letzte Demo der Reihe **Kernmethoden** („Skalarprodukte tauschen“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, der Zusammenfluss der Reihe.
Die sieben Demos davor setzen stillschweigend voraus, dass die Aufgabe ein Kernverfahren verlangt, und fragen nur, wie man es richtig rechnet. Diese Demo hebt die Annahme auf und misst die **Verfahrenswahl selbst**:
die Gauß-SVM gegen abgestimmte Bäume (Zufallswald, Gradient Boosting), ein kleines Netz und die lineare SVM, auf erzeugten **Welten**, in denen die Wahrscheinlichkeit der Klasse bekannt ist, mit der Zähleinheit Zufalls-Seeds.
Sie baut die Demos davor nicht nach, sondern gibt ihnen einen Gegner. Namensfalle: „Kern“ heißt hier nur die Methode; „Netz“ ist das neuronale Netz; „Baum“ ist der Entscheidungsbaum samt Wald und Boosting.

## Warum dieses Problem

Ein Löser, der ohne Meldung durchläuft, sagt nichts darüber, ob sein Verdikt am Gegner (Standard oder abgestimmt), an den Koordinaten (Drehung der Merkmale) oder an den Einheiten (Skalierung) hängt. Hier ist kein Löser kaputt:
die Gauß-SVM läuft fehlerfrei durch, **der Vergleich selbst meldet nichts**. Das ist die schwache Form der Lehre dieser Reihe. Die Kernfragen der App: (1) Welches Verfahren liegt in dieser Welt vorn? (2) Was ändert sich, wenn der Gegner abgestimmt ist?
(3) Was ändert eine Drehung der Merkmale? (4) Was verdirbt den Abstand? (5) Welches Verfahren passt zu meiner Aufgabe?

**Kernbotschaft, nach Welt gegliedert** (Studie: scikit-learn-Gauß-SVM gegen den besseren von abgestimmtem Wald und abgestimmtem Boosting je Zufalls-Seed, am Test gewählt; Welten erzeugt, 100 bis 400 Trainingszeilen, 20 Zufalls-Seeds je Reihe,
Bäume 5-fach abgestimmt auf den Gittern dieser Demo; Verdikt-Regel unten):

- **Schwellenregeln entlang der Achsen (Welt E):** der abgestimmte Wald schlägt die Gauß-SVM (Gauß-SVM 22,2 %, abgestimmter Wald 11,4 %, der je Zufalls-Seed bessere von Wald und Boosting im Mittel 11,3 %; Differenz zu diesem besten Baum +11,0 Prozentpunkte, Gauß besser in 0 von 20 Zufalls-Seeds): **Baum vorn**.
- **Dieselbe Aufgabe, nur gedreht (Welt Er):** **Kern vorn**, Differenz zum besten Baum −3,7 Prozentpunkte, Gauß besser in 20 von 20 Zufalls-Seeds, weil der Gauß-Kern nur Abstände sieht (Gauß-SVM 22,2 % → 22,3 %) und der Wald verliert (11,4 % → 26,2 %).
- **Glatter Kreis mit Etikettenrauschen (Welt P):** **Kern vorn** in 20 von 20 Zufalls-Seeds, Differenz zum besten Baum −3,2 Prozentpunkte.
- **Wechselwirkung (Welt B) und Container (Welt D):** gegen Standard-Bäume gleichauf bis Kern vorn; die Abstimmung verschiebt das Verdikt zu den Bäumen (B: Baum vorn, D: gleichauf in der ersten, Baum vorn in der frischen Reihe), in der frischen Reihe um zwei Stufen (Kern vorn → Baum vorn). Welche Stufe erreicht wird, hängt in diesen Welten von der Reihe und von der Wahl des besseren Baums am Test ab: in B liegt gegen das abgestimmte Boosting allein der Kern vorn (−1,1 Prozentpunkte), in D sind in der ersten Reihe Wald und Boosting einzeln gleichauf.
- **Viele irrelevante Merkmale:** Baum vorn in Welt C, in Welt F (Lieferungen) und in den Lieferungen mit 8 Rauschmerkmalen (F8). In Welt C2 liegen alle Verfahren nahe 50 % (Gauß-SVM 48,5 %, Wald 48,7 %, Boosting 48,2 %); das Verdikt „Baum vorn“ (+1,0) entsteht dort erst durch die Wahl des besseren Baums am Test, gegen Wald und Boosting einzeln sind sie gleichauf.
- **Einheiten:** auf den Lieferungen in natürlichen Einheiten bestimmt das Ladegewicht 98,9 % des mittleren Abstandsquadrats; die Gauß-SVM läuft ohne Meldung durch und ist 2,6 Prozentpunkte schlechter als standardisiert (18 von 20 Zufalls-Seeds).
- **Gegen die lineare SVM** bringt der Kern in der Klassifikation auf der fast linearen Welt A (8 von 20 Zufalls-Seeds besser) und auf dem Container-Erzeuger D (3 von 20) keinen Vorteil.
- **Das kleine Netz** (eine verdeckte Schicht, mit Konvergenz gemessen: 120 von 120 Netzen) liegt gegen die Gauß-SVM nur in B vorn, in E, D, F und P hinten und in A gleichauf bis hinten.
- **Das Live-Modell der App** (Kernel Ridge gegen eine einzelne CART) ist nicht die Studie und gibt deren Verdikte in den Welten B und D nicht wieder.

Alle Aussagen gelten für diese Erzeuger, diese Einstellungen und die genannten Gitter. **Ein Nutzen von Kernverfahren in Logistik, Lager oder Fertigung ist nicht belegt** und nicht Gegenstand; belegt sind in den gelesenen Quellen nur Kriging als
Simulations-Metamodell und die One-Class SVM bei Fahrzeug-Telemetrie, und für Verweilzeit und Nachfrage liegen in den gelesenen Studien Bäume vorn.

## Befunde und Korrekturen gegenüber dem Plan

Die Vormessungen des Plans liefen mit `numpy.random.default_rng`; diese Demo erzeugt alle Welten mit SplitMix64 (`kob_rng.py`) und hat die Studie **mit den Strömen der App neu gemessen** (`tools/kob_make_vormessung.py`, 20 Zufalls-Seeds je Reihe,
frische Zufalls-Seeds 20 bis 39 für A, B, D, P). Alle Zahlen dieser README sind die Neumessung. Was sich gegenüber dem Plan geändert hat (Planzahl → Neumessung):

| Aussage des Plans | Neumessung (Zufalls-Seed-Reihe, Gitter wie im Plan) |
|---|---|
| Welt B, Standard-Bäume: Kern vorn (12 von 20) | gleichauf (11 von 20); in der frischen Reihe Kern vorn (14 von 20): die Stufe in B liegt an der Schwelle |
| Welt B, abgestimmte Bäume: gleichauf (8 von 20) | Baum vorn (7 von 20) in der ersten und Baum vorn (6 von 20) in der frischen Reihe |
| Welt D, abgestimmte Bäume: gleichauf in beiden Reihen | gleichauf (6 von 20) in der ersten, Baum vorn (4 von 20) in der frischen Reihe |
| „weiteres Abstimmen innerhalb dieser Gitter ändert kein Verdikt“ | gilt in 10 von 11 Welten der Hauptreihe (nicht in D: gleichauf → Baum vorn) |
| Abstimmung bringt in A dem Wald +0,21 und dem Boosting +0,34 Punkte | Wald −0,01 (Standardfehler 0,15), Boosting +0,94 (0,18) |
| Welt C2: Baum vorn mit +3,3 Punkten (4 von 20) | +1,0 Punkte (Standardfehler 0,4), Gauß besser in 5 von 20: Baum vorn, aber nur knapp über der Schwelle |
| Welt B → Br: gleichauf → Kern vorn | Baum vorn (7 von 20) → Kern vorn (18 von 20) |
| Netz in A: hinten gegen die Gauß-SVM | gleichauf (+0,6, Standardfehler 0,3); mit anderen Startwerten hinten |
| Netz mit 300 Iterationen: 19 bis 20 von 20 je Welt nicht konvergiert | 20 von 20 je Welt (120 von 120); mit Grenze 5 000 Iterationen 0 von 120 |
| Median-Heuristik gegen Kreuzvalidierung auf den Ringen: 0,0019 gegen 0,0040 (Kreuzvalidierung besser in 0 von 20) | 0,0025 gegen 0,0048 (3 von 20 besser, 12 schlechter, 5 gleich); der Abstand schrumpft nicht monoton mit dem Gitter (Tabelle unten) |
| Lieferungen roh gegen standardisiert: 23,6 gegen 20,3 %, 20 von 20 | 23,4 gegen 20,8 %, 18 von 20 (Differenz 2,6 statt 3,3 Prozentpunkte) |
| Live: voll gewachsene CART gibt F gar nicht wieder (Kern vorn statt Baum vorn) und E nur an der Schwelle | F und E: gleichauf (12 und 6 von 20): die Voll-Gewachsene verfehlt Baum vorn in beiden |

Unverändert gegenüber dem Plan (Stufe, Richtung): A und P Kern vorn, E, F und F8 Baum vorn, Drehung E → Er von Baum vorn zu Kern vorn, die Abstimmung verschiebt B und D in beiden Reihen um mindestens eine Stufe zu den Bäumen (A und P nicht),
das Netz liegt nur in B vorn, Skalierung roh schlechter, 120 von 120 Netze konvergieren mit Grenze 5 000. Die Zahl „x von 6 Welten kippen“ steht nirgends: sie wäre eine Schwellenzählung.
Alle Zahlen zum Container-Erzeuger (Welt D) sind hier mit den Strömen dieser Demo gemessen und gelten für die **Klassifikation** (Verweilzeit über dem Median); die Regressionszahlen der Demo zu Kernel Ridge und Gauß-Prozess-Mittel (Kern gegen lineare Ridge bei n = 100, 300, 1 000) sind andere Aufgabe und andere Ströme und stehen hier nicht.
Zwei Planzahlen wurden nicht nachgemessen: die Gegenprobe einer unabhängigen Prüfung mit erweitertem Gitter (8 Zufalls-Seeds: C bis 300, Faktoren 0,25 bis 4, Boosting ab 10 Bäumen) und die Streuung des Verlusts F → F8 bei anderem Gitter (Gauß +0,78, Wald +0,62 bei Standardfehler etwa 0,4);
beide stehen nur als Hinweis, nicht als Messung dieser Demo, und ein erweitertes Gitter ist nicht gemessen.

## Modell

**Aufgabe:** Klassifikation mit zwei Klassen, Trainingszeilen (xᵢ, yᵢ), yᵢ ∈ {0, 1}, **bekannte Wahrscheinlichkeit** p(x) = P(y = 1 | x) in jeder Welt. **Bayes-Fehler** = Fehler der Regel „Klasse 1, wenn p(x) über 1/2“ auf frischen Testpunkten; er steht als Boden in jeder Tabelle.
**Welt** heißt ein Erzeugermodell mit bekannter Wahrscheinlichkeit, nie „Datensatz“. Welten (alle erzeugt): **A** fast linear (6 Merkmale, 300 Zeilen), **B** Wechselwirkung x₁x₂ (4 Merkmale, 2 informativ, 300 Zeilen), **C** 100 Merkmale, 3 informativ,
100 Zeilen, **C2** Wechselwirkung in 50 Merkmalen (300 Zeilen), **E** Schwellenregeln (10 Merkmale, 7 irrelevant, 300 Zeilen, Bayes-Fehler genau 10 %), **D** Container-Erzeuger (20 Spalten, 400 Zeilen, Beschreibung aus `verweilzeit-demo`),
**F** Lieferungen, Klassifikation „zu spät“ (8 Merkmale plus 3 Rauschmerkmale, 400 Zeilen, Beschreibung aus `cart-demo`), **F8** wie F mit 8 Rauschmerkmalen, **P** Liefergebiet (Kreis mit halber Fläche des Quadrats, Etikettenrauschen 0,1, 300 Zeilen,
Bayes-Fehler genau 10 %), **Er, Br** wie E und B mit zufällig gedrehten Merkmalen (eine Drehung je Zufalls-Seed), dazu nur live **E2** (Schwellenregel mit 2 Merkmalen) und **Ri** (Ringe, make_circles mit Rauschen 0,1 und Radienverhältnis 0,4; nur die Belegzeile zur Median-Heuristik).
Lieferungen und Container sind **nach der Beschreibung selbst mit SplitMix64 gezogen**: eigene Ziehungen, nicht die Zeilen der beiden Quellen und nicht die der Vormessung (andere Kundeneffekte im Container-Erzeuger, damit eine andere Schwelle T).

**Verfahren (Name = Einstellung, immer genannt).** *Lineare SVM:* C aus {0,01; 0,1; 1; 10}. *Gauß-SVM:* k(x, z) = exp(−γ‖x − z‖²) mit **γ = 1/(2σ²)** und σ = f · σ_med, Skalenfaktor **f**, σ_med = Median der paarweisen Abstände i < j ohne Diagonale der standardisierten Trainingszeilen;
die Studie wählt den Faktor g auf γ aus {0,1; 0,3; 1; 3; 10} (das ist f = g^(−1/2) ∈ {3,16; 1,83; 1; 0,58; 0,32}) und C aus {0,1; 1; 10; 100} per 5-facher Kreuzvalidierung. *Zufallswald:* 200 Bäume, Standard (scikit-learn-Voreinstellung) oder abgestimmt
(max_features aus {Wurzel, alle} mal Mindestblattgröße aus {1, 3, 10}, 6 Zellen, 5-fach). *Gradient Boosting:* Standard (100 Bäume, Tiefe 3, Lernrate 0,1) oder abgestimmt (6 Paare aus Tiefe und Lernrate: (3; 0,1), (1; 0,1), (2; 0,1), (5; 0,1), (3; 0,3), (2; 0,3), mal Zahl der Bäume 25, 50, 100, 200 über Zwischenstände,
24 Konfigurationen, 5-fach). *Netz:* scikit-learn-MLP, eine verdeckte Schicht (16 oder 64 Einheiten), L2-Strafe 0,001 oder 0,1, Adam, höchstens 5 000 Iterationen, 5-fach. **„Abgestimmt“ heißt: Hyperparameter auf dem Training per Kreuzvalidierung gewählt, nicht am Test**; die Auswahl zwischen Wald und Boosting („der Baum“, unten) ist dagegen am Test erfolgt,
die Spalte „abgestimmt“ ist also teilweise am Test gewählt, und das kann in C2 (Auswahlgewinn 0,7 Punkte) und D (0,5) das Verdikt um eine Stufe verschieben. Die am Test
im Nachhinein gewählte Gitterzelle heißt **optimistische Untergrenze** und wird getrennt gezeigt. Regler getrennt benannt: **C** (SVM), **λ** (Ridge im Live-Modell).
Aufwand der Abstimmung je Welt und Zufalls-Seed in angepassten Modellen (ohne Endanpassung): lineare SVM 20, Gauß-SVM 100, Wald 30, Boosting 30.

**Zufalls-Seeds sind getrennt:** Daten (Training 1000 + s, Test 700000 + s), Aufteilung der Kreuzvalidierung s, Modell der Bäume und des Netzes s + 50000 (Netz mit anderen Startwerten s + 60000), Drehung 9000 + s. Je Welt 20 Zufalls-Seeds, je Zufalls-Seed 4 000 frische Testpunkte.
**Verdikt-Regel (eine Festlegung für die ganze Demo):** d = Fehler(Gauß) minus Fehler(Baum) je Zufalls-Seed; **Kern vorn**, wenn der Mittelwert von d unter −2 Standardfehler liegt, **Baum vorn** über +2, sonst **gleichauf**; „k von 20“ zählt die Zufalls-Seeds, in denen die Gauß-SVM den kleineren Fehler hat.
**Der Baum** ist je Zufalls-Seed der bessere von Wald und Boosting, im Nachhinein am Test gewählt, also **optimistisch für die Bäume**.

**Studie und Live-Modell sind zwei verschiedene Dinge.** Die **Studie** ist die eingefrorene Messung (`kob_vormessung.json`, scikit-learn, nur in Werkzeug und Tests). Das **Live-Modell** rechnet die App selbst mit numpy: ein **Kernel-Ridge-Klassifikator** (Ziele −1 und +1, Gauß-Kern,
f aus {3,16; 1,83; 1; 0,58; 0,32} und λ aus {0,01; 0,1; 1} per 5-facher Kreuzvalidierung) gegen **eine einzelne CART** (Gini, Mindestblattgröße per Kreuzvalidierung aus {1, 3, 5, 10, 20}; voll gewachsen mit Blattgröße 1 wird mitgezeigt). Eine einzelne CART ist schwächer als ein abgestimmter Wald und Kernel Ridge ist nicht die SVM:
die beiden Verdikte sind verschiedene Aussagen, die App zeigt beide nebeneinander. Die Live-Instanz zieht andere Daten als die Studie, auch bei gleichem Zufalls-Seed.

## Methodik

- **Welten und Zufall:** `kob_worlds.py` mit SplitMix64 (`kob_rng.py`, Kopie von `rng_splitmix64_exact.py`); Normalzahlen als `NormalDist().inv_cdf(random())`; Gamma-Zahlen nach Marsaglia und Tsang, Beta aus Gamma, Poisson nach Knuth.
- **Studie:** `tools/kob_make_vormessung.py` rechnet Hauptreihe, Netz, Skalierung, Ringe, Lernkurve und das Live-Modell auf denselben Zufalls-Seeds und schreibt `kob_vormessung.json`; `--check` rechnet 3 Zufalls-Seeds je Welt (A, E, P) neu und prüft als Band gegen die Datei (Wald und Boosting hängen an Version und Plattform).
- **Live:** Kernel Ridge (Cholesky, höchstens 600 Zeilen) und CART in `kob_models.py`, `kob_tree.py` (Kopie von `cart_algorithm.py`); jede Einheit eine Funktion, Mini-Instanz-Tests je Einheit.
- **Auswertung:** `kob_evaluation.py` (reine Funktionen: Verdikt-Regel, Zählungen, Tabellen, Verdikt-Texte); `tools/kob_readme_tables.py` gibt die Tabellen dieser README aus.

## Befunde (gemessen, keine Behauptungen)

Alle Tabellen: Mittel über 20 Zufalls-Seeds, Testfehler in Prozent, Standardfehler in Klammern, wo genannt. Quelle ist `kob_vormessung.json` (Zufallsquelle SplitMix64, Python 3.14.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, Windows).
Jede Tabelle steht wörtlich in `tests/test_claims.py` als Funktionsausgabe und wird dort gegen diese README geprüft.

### 1. Welches Verfahren liegt in dieser Welt vorn? (Hauptreihe, Zufalls-Seeds 0 bis 19)

| Welt | Bayes | lineare SVM | Gauß-SVM | Wald Standard | Wald abgestimmt | Boosting Standard | Boosting abgestimmt | Verdikt Standard (k von 20) | Verdikt abgestimmt (k von 20) |
|---|---|---|---|---|---|---|---|---|---|
| A | 22,3 | 23,6 | 23,8 | 25,9 | 25,9 | 26,6 | 25,7 | Kern vorn (19) | Kern vorn (19) |
| B | 24,4 | 48,4 | 28,5 | 29,0 | 27,5 | 29,9 | 29,6 | gleichauf (11) | Baum vorn (7) |
| C | 19,0 | 35,8 | 35,6 | 34,8 | 30,8 | 31,2 | 30,3 | Baum vorn (1) | Baum vorn (0) |
| C2 | 24,3 | 50,1 | 48,5 | 48,6 | 48,7 | 48,1 | 48,2 | Baum vorn (3) | Baum vorn (5) |
| E | 9,9 | 26,4 | 22,2 | 12,4 | 11,4 | 13,8 | 12,4 | Baum vorn (0) | Baum vorn (0) |
| D | 20,9 | 23,3 | 24,0 | 26,4 | 24,4 | 25,6 | 24,0 | Kern vorn (17) | gleichauf (6) |
| F | 11,5 | 23,3 | 21,0 | 15,6 | 15,7 | 15,8 | 16,1 | Baum vorn (0) | Baum vorn (0) |
| P | 10,0 | 48,3 | 13,0 | 16,1 | 16,4 | 18,1 | 17,1 | Kern vorn (20) | Kern vorn (20) |
| F8 | 11,5 | 23,9 | 22,6 | 16,1 | 16,2 | 16,0 | 16,2 | Baum vorn (0) | Baum vorn (0) |
| Er | 9,9 | 26,5 | 22,3 | 25,9 | 26,2 | 27,6 | 27,4 | Kern vorn (19) | Kern vorn (20) |
| Br | 24,4 | 48,4 | 28,5 | 31,3 | 31,3 | 33,0 | 33,4 | Kern vorn (19) | Kern vorn (18) |

Gauß-SVM mit Faktor auf γ und C per Kreuzvalidierung. Differenz Gauß minus bester Baum in Prozentpunkten (Standardfehler in Klammern), je Einstellung des Gegners (positiv: Baum besser):

| Welt | Standard | abgestimmt | am Test gewählt (Untergrenze) |
|---|---|---|---|
| A | −2,0 (0,2) Kern vorn | −1,6 (0,3) Kern vorn | −1,0 (0,2) Kern vorn |
| B | −0,3 (0,4) gleichauf | +1,1 (0,5) Baum vorn | +1,8 (0,4) Baum vorn |
| C | +4,4 (0,6) Baum vorn | +6,1 (0,6) Baum vorn | +7,2 (0,6) Baum vorn |
| C2 | +1,0 (0,4) Baum vorn | +1,0 (0,4) Baum vorn | +3,6 (0,5) Baum vorn |
| E | +9,9 (0,3) Baum vorn | +11,0 (0,3) Baum vorn | +11,3 (0,3) Baum vorn |
| D | −1,5 (0,3) Kern vorn | +0,5 (0,3) gleichauf | +0,9 (0,2) Baum vorn |
| F | +5,5 (0,3) Baum vorn | +5,4 (0,3) Baum vorn | +6,0 (0,3) Baum vorn |
| P | −3,1 (0,3) Kern vorn | −3,2 (0,4) Kern vorn | −2,5 (0,3) Kern vorn |
| F8 | +6,8 (0,2) Baum vorn | +6,6 (0,2) Baum vorn | +7,4 (0,3) Baum vorn |
| Er | −3,5 (0,3) Kern vorn | −3,7 (0,3) Kern vorn | −3,3 (0,3) Kern vorn |
| Br | −2,7 (0,4) Kern vorn | −2,8 (0,5) Kern vorn | −2,2 (0,4) Kern vorn |

**Gegen die lineare SVM** (Klassifikation, Zähleinheit Zufalls-Seeds): die Gauß-SVM ist besser in A in 8, in B in 20, in C in 12, in C2 in 19, in E in 20, in D in 3, in F in 19, in P in 20 von 20 Zufalls-Seeds; in A und D bringt der Kern keinen Vorteil gegenüber der Geraden (A: 23,8 % gegen 23,6 %, D: 24,0 % gegen 23,3 %).
Der Löser sagt es nicht.

**Die Hoffnung „wenige Merkmale: Baum gewinnt“ ist nicht allgemein bestätigt:** auf den glatten Erzeugern A und P liegt der Kern vorn, in D gleichauf bis Baum vorn, in B (Wechselwirkung, vier Merkmale, nicht achsenparallel) je nach Reihe gleichauf bis Baum vorn; Bäume liegen vorn in E und F (achsenparallel gebaut)
und bei vielen irrelevanten Merkmalen. Die Lieferungen sind in der Baum-Reihe bewusst aus Schwellen und Stufen gebaut: ein Sieg der Bäume in F ist teils eine Eigenschaft des Erzeugers, keine Aussage über Lieferdaten.

### 2. Was ändert sich, wenn der Gegner abgestimmt ist?

Frische Zufalls-Seeds 20 bis 39 (nur A, B, D, P gemessen), dieselbe Zählung:

| Welt | Standard | abgestimmt | am Test gewählt (Untergrenze) |
|---|---|---|---|
| A | −2,0 (0,2) Kern vorn | −1,7 (0,2) Kern vorn | −1,2 (0,2) Kern vorn |
| B | −0,6 (0,3) Kern vorn | +0,9 (0,3) Baum vorn | +1,5 (0,3) Baum vorn |
| D | −1,4 (0,2) Kern vorn | +0,7 (0,3) Baum vorn | +1,3 (0,2) Baum vorn |
| P | −2,8 (0,3) Kern vorn | −2,7 (0,3) Kern vorn | −2,2 (0,2) Kern vorn |

**Belastbar ist die Richtung:** in B und D verschiebt die Abstimmung das Verdikt in beiden Reihen um mindestens eine Stufe zu den Bäumen, in A und P in keiner; welche Stufe in B und D erreicht wird, hängt von der Reihe ab. Die Zählung „x von y Welten kippen“ ist eine Schwellenzählung und steht nirgends.
Wirkung der Abstimmung auf die Bäume (Standard minus abgestimmt in Prozentpunkten, Standardfehler in Klammern, positiv: abgestimmt besser; „schlechter in k“: Zufalls-Seeds von 20, in denen abgestimmt schlechter ist als Standard):

| Welt | Wald | Boosting |
|---|---|---|
| A | −0,01 (0,15), schlechter in 10 | +0,94 (0,18), schlechter in 2 |
| B | +1,55 (0,24), schlechter in 1 | +0,32 (0,28), schlechter in 9 |
| C | +3,94 (0,58), schlechter in 1 | +0,85 (0,31), schlechter in 4 |
| C2 | −0,12 (0,21), schlechter in 11 | −0,11 (0,50), schlechter in 10 |
| E | +1,01 (0,23), schlechter in 3 | +1,45 (0,26), schlechter in 2 |
| D | +1,98 (0,35), schlechter in 2 | +1,58 (0,25), schlechter in 1 |
| F | −0,09 (0,14), schlechter in 10 | −0,33 (0,20), schlechter in 12 |
| P | −0,29 (0,20), schlechter in 11 | +0,98 (0,29), schlechter in 4 |
| F8 | −0,15 (0,14), schlechter in 12 | −0,21 (0,11), schlechter in 14 |
| Er | −0,35 (0,18), schlechter in 11 | +0,25 (0,21), schlechter in 9 |
| Br | −0,05 (0,22), schlechter in 8 | −0,38 (0,29), schlechter in 13 |

Die Abstimmung bringt dem Wald in B, C, D und E und dem Boosting in A, C, D, E und P einen belegbaren Zuwachs (mehr als zwei Standardfehler), in F, F8 und C2 keinen. Sie ist nicht in jedem Zufalls-Seed besser: abgestimmt schlechter als Standard in 1 bis 14 von 20 Zufalls-Seeds je Welt und Verfahren.
Die **Untergrenze** (Gitterzelle je Zufalls-Seed am Test gewählt) ändert das Verdikt in 10 von 11 Welten nicht; in D geht es von gleichauf zu Baum vorn. Sie wählt im selben Gitter und kann Randwahl nicht prüfen: das Boosting wählt in Welt E den kleinsten Zwischenstand (25 Bäume) in 16 von 20 Zufalls-Seeds,
und der C-Wert der Gauß-SVM liegt in den Welten A, B, E, D, F, P in 4 bis 10 von 20 Zufalls-Seeds am Rand des Gitters, der Faktor g auf γ in 1 bis 11, mindestens einer der beiden in 4 bis 14 von 20 Zufalls-Seeds. Erweiterte Gitter sind nicht gemessen.

### 3. Dieselbe Aufgabe, andere Koordinaten

Welt E gegen Er (zufällige orthogonale Drehung je Zufalls-Seed, Training und Test gleich gedreht): lineare SVM 26,4 % → 26,5 %, **Gauß-SVM 22,2 % → 22,3 %** (Differenz +0,1 Prozentpunkte, Standardfehler 0,2), abgestimmter Wald 11,4 % → 26,2 % (+14,9, Standardfehler 0,3; schlechter in 20 von 20 Zufalls-Seeds),
abgestimmtes Boosting 12,4 % → 27,4 %. Verdikt: ungedreht **Baum vorn** (+11,0 Prozentpunkte, Gauß besser in 0 von 20), gedreht **Kern vorn** (−3,7, 20 von 20). Welt B gegen Br: Gauß-SVM 28,5 % → 28,5 %, abgestimmter Wald 27,5 % → 31,3 % (+3,9, 20 von 20 schlechter);
Verdikt Baum vorn (7 von 20) → Kern vorn (18 von 20). **Der Gauß-Kern ist drehinvariant (bis auf Rundung):** die Gram-Matrizen vor und nach der Drehung unterscheiden sich um weniger als 1e-13 (Test: 10 Drehungen, 60 Zeilen, 10 Merkmale), die Entscheidungsfunktion der scikit-learn-SVM um weniger als 1e-7 und alle Vorhersagen sind gleich.
Dass die Gauß-SVM in B um 0,03 Prozentpunkte schwankt, kommt daher, dass jede Reihe nach der Drehung neu standardisiert wird. Eine voll gewachsene CART hat in Welt E (300 Zeilen, Mittel über 20 Zufalls-Seeds) 38,6 Blätter, in Welt Er 54,0 (Live-Modell).
Am Handbeispiel mit sechs Punkten (Abbildung T(x) = (x₁ + x₂, x₂ − x₁), eine Drehung um 45 Grad mit Streckung √2): alle Abstandsquadrate genau verdoppelt, kleinster Baum ohne Trainingsfehler **2 Blätter vor, 4 nach** der Drehung (zwei Rechenwege plus scikit-learn).
Der Baum ist nicht für jede Aufgabe drehempfindlich: eine Wahrheit entlang der Diagonalen verliert durch Drehung nichts (Test).

### 4. Was den Abstand verdirbt: irrelevante Merkmale und Einheiten

**Irrelevante Merkmale.** Gauß-SVM gegen den besten abgestimmten Baum: Welt C (100 Merkmale, 3 informativ, 100 Zeilen) +6,1 Prozentpunkte, Gauß besser in 0 von 20 Zufalls-Seeds; Welt C2 +1,0, 5 von 20 (nur knapp über der Schwelle, gegen Wald und Boosting einzeln gleichauf); Lieferungen mit 8 Rauschmerkmalen (F8) +6,6, 0 von 20.
Der Kern verliert beim Übergang von 3 auf 8 Rauschmerkmalen (F → F8) 1,6 Prozentpunkte (Standardfehler 0,3), der abgestimmte Wald 0,5 (0,1), das abgestimmte Boosting 0,1 (0,2); die Größe dieses Unterschieds hängt am Gitter (Gegenprobe mit anderem Gitter oben: Gauß +0,78, Wald +0,62). In Welt C ist die Gauß-SVM nicht besser als die lineare (35,6 % gegen 35,8 %, in 12 von 20 Zufalls-Seeds besser).
Mechanismus (exakt): der Erwartungswert des Abstandsquadrats ist 2 je standardisiertem Merkmal, der Anteil aus k relevanten von p Merkmalen ist **k/p** (Aufzählung aller 1 024 Paare von ±1-Vektoren mit p = 5, k = 3: 3/5; die App zeigt den Anteil für die 2D-Schwellenregel mit bis zu 8 Rauschmerkmalen).
Dass die Median-Heuristik die Gram-Matrix numerisch rettet, aber die Konzentration der Abstände auch lineare SVM und Wald trifft, ist hier nicht gemessen.

**Einheiten (Lieferungen in natürlichen Einheiten, Welt F, 400 Zeilen, 20 Zufalls-Seeds):** Gauß-SVM mit Median-Heuristik und C per Kreuzvalidierung **roh 23,4 %** (Standardfehler 0,2), **standardisiert 20,8 %** (0,2), **auf [0, 1] skaliert 20,0 %** (0,3); roh schlechter als standardisiert in 18 von 20 Zufalls-Seeds (+2,6 Prozentpunkte, Standardfehler 0,3);
mit Faktor auf γ und C per Kreuzvalidierung auf der rohen Skala 23,4 %: die Bandbreite kann zwei Skalen nicht reparieren. σ_med ist roh 325,8 gegen 4,6 standardisiert; **das Ladegewicht (Kilogramm) bestimmt 98,9 % des mittleren Abstandsquadrats, die Distanz 1,0 %**.
Bäume sind gegen die Einheiten unempfindlich (Schwellen; Test: gleiche Vorhersagen nach Umrechnung jeder Spalte mit einem positiven Faktor). Die Aussage „roh schlechter“ hängt nicht an der Einstellung der Gauß-SVM.

**Belegzeile zur Median-Heuristik** (die Heuristik als Verfahren führt die Demo zur Kernwahl): Ringe (make_circles, Rauschen 0,1, 100 Trainingszeilen, Bayes-Fehler 0,15 %), Median-Heuristik mit C per Kreuzvalidierung gegen γ und C per Kreuzvalidierung, je Gitter des Faktors g auf γ (Testfehler in Prozent):

| Reihe | Gitter des Faktors g auf gamma | Median-Heuristik (C per Kreuzvalidierung) | gamma und C per Kreuzvalidierung | Kreuzvalidierung besser / schlechter / gleich (Zufalls-Seeds) |
|---|---|---|---|---|
| Zufalls-Seeds 0 bis 19 | 0,01 bis 100 | 0,253 | 0,477 | 3 / 12 / 5 |
| Zufalls-Seeds 0 bis 19 | 0,1 bis 10 | 0,253 | 0,378 | 3 / 10 / 7 |
| Zufalls-Seeds 0 bis 19 | 0,2 bis 5 | 0,253 | 0,545 | 4 / 15 / 1 |
| Zufalls-Seeds 0 bis 19 | 0,5 bis 2 | 0,253 | 0,419 | 4 / 14 / 2 |
| frische Zufalls-Seeds 100 bis 139 | 0,01 bis 100 | 0,248 | 0,408 | 0 / 21 / 19 |
| frische Zufalls-Seeds 100 bis 139 | 0,1 bis 10 | 0,248 | 0,408 | 0 / 19 / 21 |
| frische Zufalls-Seeds 100 bis 139 | 0,2 bis 5 | 0,248 | 0,416 | 5 / 23 / 12 |
| frische Zufalls-Seeds 100 bis 139 | 0,5 bis 2 | 0,248 | 0,363 | 6 / 20 / 14 |

**Die Richtung bleibt** (die Heuristik ist in keinem Gitter im Mittel schlechter), der Abstand schrumpft nicht monoton mit dem Gitter; Größe und Niveau hängen an Gitter und Datenreihe, nicht an der Heuristik allein. Die Aussage „die Heuristik ist optimal“ folgt daraus nicht.
Zweite Koordinate mal 10: Median-Heuristik roh 13,9 %, standardisiert 0,24 %, γ und C per Kreuzvalidierung auf roher Skala (Gitter 0,01 bis 100) 4,8 % (20 Zufalls-Seeds).

### 5. Das kleine Netz (eingefroren)

Netz: scikit-learn-MLP, eine verdeckte Schicht, vier Zellen, Adam, höchstens 5 000 Iterationen, standardisierte Eingaben, 20 Zufalls-Seeds je Welt (Testfehler in Prozent). **Konvergenz** (Zähleinheit Zufalls-Seeds mal Welten): das gewählte Netz hat die Grenze 5 000 in 0 von 120 Fällen ausgeschöpft
(0 Konvergenzmeldungen unter 2 520 Anpassungen; Iterationen 316 bis 2 018); mit der Grenze 300 schöpfte es sie in 120 von 120 Fällen aus und wird nicht verwendet.

| Welt | Netz | Gauß-SVM | Netz minus Gauß-SVM (Standardfehler) | Netz minus abgestimmter Wald | Stufe gegen Gauß-SVM mit anderen Startwerten |
|---|---|---|---|---|---|
| A | 24,43 | 23,84 | +0,60 (0,31): gleichauf | −1,45 (0,34): Netz vorn | Netz hinten |
| B | 26,43 | 28,51 | −2,08 (0,31): Netz vorn | −1,02 (0,35): Netz vorn | Netz vorn |
| E | 25,35 | 22,25 | +3,10 (0,39): Netz hinten | +13,98 (0,46): Netz hinten | Netz hinten |
| D | 26,51 | 23,97 | +2,54 (0,35): Netz hinten | +2,14 (0,37): Netz hinten | Netz hinten |
| F | 22,22 | 20,98 | +1,23 (0,27): Netz hinten | +6,54 (0,25): Netz hinten | Netz hinten |
| P | 14,59 | 13,04 | +1,56 (0,22): Netz hinten | −1,80 (0,34): Netz vorn | Netz hinten |

Das Netz liegt gegen die Gauß-SVM nur in B vorn; in A steht es an der Schwelle (gleichauf, mit anderen Startwerten hinten). Es gilt für diese Architektur, diese vier Zellen und diesen Optimierer; andere Tiefen, Strafen, früher Halt und die Drehung für das Netz sind nicht gemessen.
Das Netz ist in der App kein Regler und wird eingefroren gezeigt.

### 6. Studie gegen Live-Modell

Zwei Spalten mit je eigenem Verdikt, derselben Regel und denselben Zufalls-Seeds und Testpunkten; „Live“ rechnet Kernel Ridge gegen eine einzelne CART (Fehler in Prozent, Mittel über 20 Zufalls-Seeds):

| Welt | Studie: Verdikt gegen abgestimmte Bäume (k von 20) | Live: Kernel Ridge | Live: CART mit Blattgröße per Kreuzvalidierung | Live: CART voll gewachsen |
|---|---|---|---|---|
| A | Kern vorn (19) | 24,08 | 29,74: Kern vorn (19) | 32,72: Kern vorn (20) |
| B | Baum vorn (7) | 28,03 | 34,54: Kern vorn (20) | 36,01: Kern vorn (20) |
| E | Baum vorn (0) | 22,23 | 11,72: Baum vorn (0) | 21,33: gleichauf (6) |
| Er | Kern vorn (20) | 22,33 | 33,38: Kern vorn (20) | 35,27: Kern vorn (20) |
| D | gleichauf (6) | 23,89 | 25,73: Kern vorn (15) | 29,64: Kern vorn (20) |
| F | Baum vorn (0) | 20,50 | 19,08: Baum vorn (5) | 21,12: gleichauf (12) |
| F8 | Baum vorn (0) | 22,25 | 19,04: Baum vorn (2) | 21,76: gleichauf (7) |
| P | Kern vorn (20) | 13,26 | 18,75: Kern vorn (20) | 22,30: Kern vorn (20) |

Das Live-Verdikt (CART mit Blattgröße per Kreuzvalidierung) stimmt in A, E, Er, F, F8 und P mit dem Studien-Verdikt überein; **in B und D weicht es ab** (live Kern vorn, Studie Baum vorn bzw. gleichauf), weil eine einzelne CART schwächer ist als Wald und Boosting.
Die voll gewachsene CART gibt in E und F und F8 das Studien-Verdikt „Baum vorn“ nicht wieder (gleichauf). Deshalb heißen die Presets nach der Welt, nie nach einem Verdikt. Der Kernel Ridge liegt nahe an der scikit-learn-Gauß-SVM der Studie (Abstand der Mittel höchstens 0,5 Prozentpunkte je Welt).

### 7. Lernkurve (eingefroren)

Gauß-SVM / abgestimmter Wald, Testfehler in Prozent, Mittel über 10 Zufalls-Seeds, in Klammern die Zahl der Zufalls-Seeds von 10, in denen die Gauß-SVM besser ist (Lernkurve nur für diese sechs Welten und nicht für Boosting oder Netz gemessen):

| Welt | n = 50 | n = 100 | n = 200 | n = 400 | n = 800 | n = 1600 |
|---|---|---|---|---|---|---|
| A | 29,8 / 31,4 (5) | 25,6 / 29,2 (10) | 24,0 / 26,5 (10) | 22,8 / 25,6 (10) | 22,9 / 24,2 (10) | 22,5 / 23,5 (9) |
| B | 39,2 / 39,2 (5) | 33,1 / 33,1 (4) | 29,1 / 28,2 (4) | 27,4 / 26,5 (4) | 26,4 / 25,6 (2) | 25,4 / 25,6 (7) |
| E | 29,7 / 22,6 (1) | 27,8 / 15,0 (0) | 23,4 / 11,8 (0) | 21,2 / 10,7 (0) | 19,1 / 10,5 (0) | 16,9 / 10,1 (0) |
| D | 35,9 / 30,5 (1) | 27,9 / 26,4 (3) | 26,2 / 25,9 (4) | 24,1 / 24,7 (5) | 22,1 / 22,0 (3) | 22,0 / 22,0 (5) |
| F | 29,7 / 25,8 (1) | 26,3 / 20,9 (0) | 23,2 / 18,1 (0) | 20,9 / 15,7 (0) | 18,9 / 14,5 (0) | 17,3 / 13,8 (0) |
| P | 21,8 / 23,8 (7) | 17,5 / 19,7 (8) | 15,0 / 17,6 (9) | 12,9 / 14,9 (10) | 11,6 / 13,4 (10) | 11,1 / 12,5 (10) |

Bei n = 1 600 liegt der Fehler der Gauß-SVM in jeder der sechs Welten unter dem bei n = 50 (nicht monoton in n). In E und F bleibt der Wald auch bei n = 1 600 vorn (Gauß-SVM besser in 0 von 10 Zufalls-Seeds); in B und D liegen beide bei n = 1 600 im Mittel gleich (7 und 5 von 10 Zufalls-Seeds, Mittel 25,4 und 25,6 % bzw. 22,0 und 22,0 %; kein Verdikt gebildet, die Lernkurve nennt keine Standardfehler).

### 8. Orakel-Größen (exakt oder Band)

Bayes-Fehler Welten E und P exakt 0,1 (p nimmt nur die Werte 0,1 und 0,9 an); Welt B: E[σ(−|3·x₁x₂|)] = 0,243904 auf zwei Quadraturen (Dichte des Produkts zweier Normalzahlen, Polarkoordinaten) und im Band auf 200 000 Ziehungen;
Lieferungen: P(zu spät | x) = 1 − Φ((versprochen − Dauer)/7) ist geschlossen, Anteil zu spät gleich mittlerer Wahrscheinlichkeit, alle zehn Zehntel der Kalibrierung im Band, Bayes-Fehler E[min(p, 1 − p)] rund 11,7 %. Container-Schwelle T (Median der Verweilzeit) aus der Merkmalsverteilung gerechnet, nicht gezogen.
Rechenkosten des exakten Kernel Ridge bei n = 20 000 (Querprüfung des Handbeispiels): n(n+1)(2n+1)/6 = 2 666 866 670 000 Operationen, 4·10⁸ Gram-Einträge, 2,98 GiB; Eigentümer der Zählung ist die Demo zur Gram-Näherung.

### Zahl → Test

| Zahl oder Aussage | Test (alle in `tests/`) |
|---|---|
| Tabellen 1 bis 7 (Hauptreihe, Differenzen, Abstimmung, Ringe, Netz, Live, Lernkurve) wörtlich aus der Studie | `test_claims.py::test_tabellen_stehen_woertlich_in_der_readme` |
| Kernbotschaft je Welt (Verdikt, Zählung, Prozentzahlen) | `test_claims.py::test_kernbotschaft_je_welt` |
| Abstimmung verschiebt B und D in beiden Reihen, A und P nicht | `test_claims.py::test_abstimmung_verschiebt_b_und_d_in_beiden_reihen` |
| Untergrenze ändert 10 von 11 Verdikten nicht, in D ja; Gitterrand 16 von 20 und 4 bis 10 von 20 | `test_claims.py::test_untergrenze_und_gitterrand` |
| Drehung E → Er und B → Br, Blätter 38,6 und 54,0 | `test_claims.py::test_drehung`; Invarianz: `test_oracle_kernel_oder_baum.py`, `test_models.py` |
| Rauschmerkmale, Verlust F → F8, Anteil k/p | `test_claims.py::test_rauschmerkmale`; `test_hand.py` (1 024 Paare), `test_oracle_kernel_oder_baum.py` (Band) |
| Einheiten (roh, standardisiert, [0, 1], Ladegewicht 98,9 %) | `test_claims.py::test_einheiten`; `test_worlds.py` (Ladegewicht im Band) |
| Netz: Konvergenz 0 von 120 und 120 von 120, Stufen | `test_claims.py::test_netz` |
| Live gegen Studie, Abweichung in B und D | `test_claims.py::test_live_gegen_studie` |
| Kern gegen lineare SVM (Zählungen) | `test_claims.py::test_kern_gegen_linear` |
| Bayes-Fehler E, P, B, F, Kalibrierung, Container-Schwelle | `test_worlds.py` |
| Handbeispiele (XOR, 2 gegen 4 Blätter, Verdikt in Brüchen, k/p, Kosten, Umrechnung f und g) | `test_hand.py` |
| Literaturzahlen (Delgado, McElfresh) | `test_literatur.py` |
| Neu-Rechnen der Studie als Band | `test_vormessung_tool.py` |

Nicht im Test, Messung dieser README: die Streuung der Netz-Rechenzeit (Laufzeiten sind kein Befund), die Gegenproben der unabhängigen Prüfung (erweitertes Gitter, F → F8 bei anderem Gitter, Hinweis oben).

## Ehrliche Grenzen

- **Alle Welten sind erzeugt;** es gibt keine reale Datenquelle. Welches Verfahren in einer Welt vorn liegt, sagt nichts über eine andere Welt. Die Lieferungen sind aus Schwellen und Stufen gebaut, damit ein Baum etwas zu tun hat.
- **Ein Gitter je Verfahren,** nicht bis zum Optimum erweitert (Gitterrand siehe Abschnitt 2). Die Untergrenze im selben Gitter kann Randwahl nicht prüfen.
- **Der Baum ist je Zufalls-Seed am Test gewählt:** optimistisch für die Bäume; die Auswahl nach Kreuzvalidierung statt am Test ist nicht gemessen.
- **Die Stufe in Welten nahe an der Schwelle (B, D, A beim Netz) hängt von der Reihe ab;** belastbar ist die Richtung.
- **Das Live-Modell ist nicht die Studie** (Abschnitt 6). Die Live-Instanz ist eine einzelne Ziehung mit höchstens 600 Zeilen und belegt kein Verdikt.
- **Eine Plattform:** Windows, Python 3.14.6, scikit-learn 1.9.1; Wald, Boosting und Netz hängen mit festem Zufalls-Seed an Version und Betriebssystem (Bänder, nicht exakt). Ein Linux-Lauf (`demo_linux_check.py`) steht vor dem Push noch aus.
- **Rechenzeiten sind nur Orientierung** und kein Befund.
- **Nutzen in Logistik, Lager oder Fertigung ist nicht belegt;** TabPFN ist nur Quelle (Herstellerangabe), nicht gemessen und nicht in App und Tests.

## Bewusst nicht umgesetzt (nicht gemessen)

- **Weitere Baumverfahren** (XGBoost, LightGBM, CatBoost, Extra Trees, HistGradientBoosting): nur der Satz „andere Baumverfahren nicht gemessen“.
- **Erweiterte Gitter** (Boosting unter 25 Bäumen, Gauß-C über 100, mehr Wald-Konfigurationen): nicht gemessen und kein Regler.
- **TabPFN:** nur Quelle und ein Satz mit Herstellerangabe (Lizenz nichtkommerziell, Torch).
- **Lernkurve für Boosting, Netz, Welten Er, F8, C, C2:** nicht gemessen.
- **Netz:** andere Tiefen, Strafen, Optimierer, früher Halt und die Drehung für das Netz; das Netz ist kein Regler.
- **Platt-Skalierung, konforme Bänder, Kern gegen Baum auf Welt F bei n weit über 400:** nicht gemessen (Verweis `kalibrierung-schwelle-demo`).
- **Randsätze aus der Literatur** (Experiment 5): Fernández-Delgado u. a. 2014 nur mit den Zahlen aus Fließtext und Tabelle 5 (Kennzahl mittlere Prozentzahl der je Datensatz höchsten Genauigkeit, nicht die Genauigkeit), keine p-Werte aus Abbildung 4, kein Satz „zweitbeste“.

## Verifikation

- **Welten:** Form, Determinismus, Bayes-Fehler gegen Quadratur und geschlossene Formel, Verteilungen der Lieferungen und des Container-Erzeugers als Band über 4,5 Standardfehler (`tests/test_worlds.py`, `tests/test_rng.py`; die Zufallsquelle ist eine Kopie mit Kopier-Test).
- **Live-Modell:** jede Einheit an einer Mini-Instanz von Hand (Median-Heuristik, Skalenfaktor, Gram-Matrix, XOR mit Eigenvektor, Kreuzvalidierung gegen Aufzählung), Randfälle (`tests/test_models.py`, `tests/test_edge_cases.py`).
- **Orakel:** Kernel Ridge gegen scikit-learn `KernelRidge` und die Eigenzerlegung, Median-Heuristik gegen `pdist`, Drehinvarianz gegen die scikit-learn-SVM, Verdikt-Regel gegen die t-Statistik aus scipy, CART gegen scikit-learn und gegen Aufzählung (`tests/test_oracle_kernel_oder_baum.py`, `tests/test_tree.py`, `tests/test_hand.py`).
- **Verdikte:** reine Funktionen mit Schwellen und jeder Zahl im Text, je Preset die erwartete Verdikt-Zeile (`tests/test_verdicts.py`, `tests/test_presets.py`, `tests/test_claims.py`).
- **Studie:** `--check` des Werkzeugs als Band (`tests/test_vormessung_tool.py`).
- **Oberfläche:** AppTest-Rauchtest, Footer, Laufzeit-Importe, Layout mit Schnellstart und Hervorhebung; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_app.py`, `tests/test_footer.py`, `tests/test_runtime_imports.py`, `tests/test_layout.py`).
- **Handbeispiele des Plans** (XOR, Drehung, Verdikt in Brüchen, k/p, Kosten, Median-Heuristik) laufen als Tests in `tests/test_hand.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `kob_rng.py` | Zufallsquelle SplitMix64 (Kopie der Vorlage, Kopier-Test) |
| `kob_worlds.py` | Welten (Erzeuger mit bekannter Wahrscheinlichkeit), Drehung |
| `kob_models.py` | Live-Modell: Median-Heuristik, Gauß-Kern, Kernel Ridge, Kreuzvalidierung, Abstandsanteile |
| `kob_tree.py` | CART (Kopie von `cart_algorithm.py` aus `cart-demo`, Kopier-Test) |
| `kob_frozen.py` | Laden der eingefrorenen Studie |
| `kob_evaluation.py` | Verdikt-Regel, Zählungen, Tabellen, Verdikt-Texte (reine Funktionen) |
| `kob_decision.py` | Entscheidungsbaum und Randsätze (ohne neue Rechnung) |
| `kob_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kob_constants.py` | Regler-Grenzen, Welten, Presets, Farben, Verdikt-Schwelle |
| `kob_layout.py` | Fester Kopf (Stylesheet) |
| `kob_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `kob_texts.py` | Texte der App (Zahlen aus der Studie berechnet) |
| `kob_vormessung.json` | eingefrorene Studie (Fehler je Zufalls-Seed) |
| `tools/kob_make_vormessung.py` | erzeugt und prüft (`--check`) die eingefrorene Studie |
| `tools/kob_readme_tables.py` | gibt die Tabellen dieser README aus |
| `requirements.txt` | Laufzeit: streamlit, numpy, pandas, plotly (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scikit-learn, scipy, joblib (nur Tests und Werkzeug) |
| `tests/` | Tests (Zahl siehe unten) |

## Tests

**206 Tests** (`pytest --collect-only -q tests`; Zusammensetzung: Welten 45, Handbeispiele 22, Live-Modell 27, Claims 20, Layout 18, Randfälle 11, Entscheidungsbaum 9, Presets 9, Verdikte 9, Orakel 8, Zufallsquelle 7, CART 6, Literatur 5, Werkzeug 4, AppTest-Rauchtest 3, Footer 2, Laufzeit-Importe 1).
Lokal im Test-venv mit `pytest tests -q -n 8` rund 3 Minuten (Laufzeiten sind nur Orientierung und kein Befund). Die Zahl der Tests wird von `tests/test_claims.py` gegen `pytest --collect-only` geprüft.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas und NumPy; scikit-learn und SciPy nur in Tests und Werkzeug.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html).
