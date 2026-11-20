"""Plotly-Figuren der App (alle Achsen fest, `fixedrange`, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kob_constants as C

_FONT = dict(size=13)


def _fixed(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def fig_error_bars(rows, bayes: float | None = None, title: str = "", x_max: float | None = None):
    """Waagerechte Balken: `rows` = [(Beschriftung, Fehler als Anteil, Standardfehler oder None, Farbe)]; senkrechte Linie für den Bayes-Fehler."""
    fig = go.Figure()
    labels = [r[0] for r in rows]
    vals = [100.0 * r[1] for r in rows]
    errs = [None if r[2] is None else 100.0 * r[2] for r in rows]
    fig.add_trace(go.Bar(
        y=labels, x=vals, orientation="h", marker_color=[r[3] for r in rows], error_x=dict(type="data", array=[0 if e is None else e for e in errs], visible=True),
        text=[f"{v:.1f} %".replace(".", ",") for v in vals], textposition="outside", cliponaxis=False, hoverinfo="skip",
    ))
    if bayes is not None:
        fig.add_vline(x=100.0 * bayes, line_dash="dash", line_color=C.COLORS["bayes"], annotation_text="Bayes-Fehler", annotation_position="top")
    top = x_max if x_max is not None else max(vals + [100.0 * (bayes or 0.0)]) * 1.25 + 2
    fig.update_layout(title=title, xaxis_title="Testfehler in %", yaxis=dict(autorange="reversed"), height=60 + 46 * len(rows) + 40, margin=dict(l=10, r=30, t=50, b=40), showlegend=False, font=_FONT)
    fig.update_xaxes(range=[0, top])
    return _fixed(fig)


def fig_boundary(X, y, grid_extent, preds: dict, titles: dict, n_grid: int):
    """Zwei (oder eine) Entscheidungsflächen auf einer 2D-Welt: `preds` = {Schlüssel: Vorhersagen (0/1) auf dem Gitter (n_grid x n_grid, Zeile = y-Wert)}; Punkte = Trainingszeilen."""
    keys = list(preds)
    x0, x1, y0, y1 = grid_extent
    xs, ys = np.linspace(x0, x1, n_grid), np.linspace(y0, y1, n_grid)
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[titles[k] for k in keys], horizontal_spacing=0.06)
    for i, k in enumerate(keys, start=1):
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=preds[k], showscale=False, zmin=0, zmax=1, opacity=0.35, hoverinfo="skip",
                                 colorscale=[[0, C.COLORS["klasse0"]], [1, C.COLORS["klasse1"]]]), row=1, col=i)
        for cls, col in ((0, C.COLORS["klasse0"]), (1, C.COLORS["klasse1"])):
            m = y == cls
            fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", marker=dict(size=5, color=col, line=dict(width=0.5, color="white")), showlegend=False, hoverinfo="skip"), row=1, col=i)
    fig.update_xaxes(range=[x0, x1], scaleanchor=None)
    fig.update_yaxes(range=[y0, y1])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=30), font=_FONT)
    return _fixed(fig)


def fig_rotation(angles, series: dict, bayes: float):
    """Fehler gegen Drehwinkel: `series` = {Beschriftung: (Fehler-Liste als Anteil, Farbe)}."""
    fig = go.Figure()
    for name, (vals, color) in series.items():
        fig.add_trace(go.Scatter(x=list(angles), y=[100.0 * v for v in vals], mode="lines+markers", name=name, line=dict(color=color), hovertemplate="%{x}°: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=100.0 * bayes, line_dash="dash", line_color=C.COLORS["bayes"], annotation_text="Bayes-Fehler", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Drehwinkel der Merkmale in Grad", yaxis_title="Testfehler in %", height=340, margin=dict(l=10, r=10, t=30, b=40), font=_FONT, legend=dict(orientation="h", y=1.12))
    return _fixed(fig)


def fig_share(labels, shares):
    """Anteil des mittleren Abstandsquadrats je Spalte (Summe 1)."""
    fig = go.Figure(go.Bar(x=list(labels), y=[100.0 * s for s in shares], marker_color=C.COLORS["kern"], text=[f"{100 * s:.1f} %".replace(".", ",") for s in shares], textposition="outside", cliponaxis=False,
                           hoverinfo="skip"))
    fig.update_layout(yaxis_title="Anteil am mittleren Abstandsquadrat in %", height=320, margin=dict(l=10, r=10, t=30, b=60), font=_FONT)
    fig.update_yaxes(range=[0, 115])
    return _fixed(fig)


def fig_learning(lc: dict, title: str = ""):
    """Lernkurve: Fehler gegen die Zahl der Trainingszeilen (logarithmisch), Gauß-SVM und abgestimmter Wald mit Standardfehler, Bayes-Fehler als Linie."""
    fig = go.Figure()
    n = lc["n"]
    for key, name, color in (("rbf", "Gauß-SVM", C.COLORS["kern"]), ("rf", "abgestimmter Wald", C.COLORS["baum"])):
        fig.add_trace(go.Scatter(x=n, y=[100.0 * v for v in lc[key]], mode="lines+markers", name=name, line=dict(color=color),
                                 error_y=dict(type="data", array=[100.0 * v for v in lc[key + "_se"]], visible=True), hovertemplate="n = %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=n, y=[100.0 * v for v in lc["bayes"]], mode="lines", name="Bayes-Fehler", line=dict(color=C.COLORS["bayes"], dash="dash")))
    fig.update_layout(title=title, xaxis_title="Trainingszeilen n", yaxis_title="Testfehler in %", height=340, margin=dict(l=10, r=10, t=50, b=40), font=_FONT, legend=dict(orientation="h", y=1.15))
    fig.update_xaxes(type="log", tickvals=n)
    return _fixed(fig)
