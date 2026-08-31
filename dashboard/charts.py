# dashboard/charts.py
# ─────────────────────────────────────────────────────────────────
#  Générateur de graphiques Plotly — Dashboard du bot
#
#  Graphiques disponibles :
#   • Courbe d'équité (avec zones de drawdown)
#   • Jauge Win Rate (gauge chart)
#   • Distribution TP/SL (pie chart)
#   • Performance par session (bar chart)
#   • Performance par score (bar chart)
#   • Historique des trades (scatter)
#   • Heatmap mensuelle des P&L
#   • Évolution du WR dans le temps (rolling)
# ─────────────────────────────────────────────────────────────────

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from adaptive.performance_tracker import PerformanceReport
from backtesting.metrics import MetricsResult, EquityPoint


# ── Palette de couleurs ───────────────────────────────────────────
COLORS = {
    "green":       "#00C896",
    "red":         "#FF4C4C",
    "gold":        "#FFD700",
    "blue":        "#4C9BE8",
    "purple":      "#9B59B6",
    "bg":          "#0D1117",
    "bg_card":     "#161B22",
    "border":      "#30363D",
    "text":        "#E6EDF3",
    "text_muted":  "#8B949E",
    "win":         "#238636",
    "loss":        "#DA3633",
    "breakeven":   "#E3B341",
}

LAYOUT_BASE = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["bg"],
    font          = dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin        = dict(l=40, r=20, t=40, b=40),
    legend        = dict(bgcolor=COLORS["bg_card"], bordercolor=COLORS["border"]),
)


class ChartBuilder:
    """
    Génère des graphiques Plotly en JSON pour le dashboard FastAPI.
    Tous les graphiques retournent du JSON (pour le frontend JS).
    """

    # ── COURBE D'ÉQUITÉ ───────────────────────────────────────────

    @staticmethod
    def equity_curve(
        equity_points: list[EquityPoint],
        initial_balance: float = 10000.0,
        title: str = "Courbe d'Équité",
    ) -> str:
        """
        Courbe d'équité avec zones de drawdown en rouge transparent.
        Double axe Y : équité (gauche) + drawdown % (droite).
        """
        if not equity_points:
            return ChartBuilder._empty_chart(title)

        timestamps  = [p.timestamp for p in equity_points]
        equity_vals = [p.equity    for p in equity_points]
        dd_vals     = [p.drawdown_pct for p in equity_points]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.05,
        )

        # ── Courbe d'équité ───────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x     = timestamps,
                y     = equity_vals,
                name  = "Équité",
                line  = dict(color=COLORS["green"], width=2),
                fill  = "tozeroy",
                fillcolor = "rgba(0, 200, 150, 0.08)",
                hovertemplate = (
                    "<b>%{x|%d/%m %H:%M}</b><br>"
                    "Équité: <b>%{y:.2f} USD</b><extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # Ligne de référence (capital initial)
        fig.add_hline(
            y          = initial_balance,
            line_dash  = "dash",
            line_color = COLORS["text_muted"],
            row=1, col=1,
        )

        # ── Drawdown ──────────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x     = timestamps,
                y     = [-d for d in dd_vals],
                name  = "Drawdown %",
                line  = dict(color=COLORS["red"], width=1.5),
                fill  = "tozeroy",
                fillcolor = "rgba(255, 76, 76, 0.15)",
                hovertemplate = (
                    "<b>%{x|%d/%m %H:%M}</b><br>"
                    "Drawdown: <b>%{y:.1f}%</b><extra></extra>"
                ),
            ),
            row=2, col=1,
        )

        fig.update_layout(
            **LAYOUT_BASE,
            title     = dict(text=title, font=dict(size=16)),
            height    = 500,
            hovermode = "x unified",
        )

        fig.update_xaxes(
            gridcolor = COLORS["border"],
            showgrid  = True,
            zeroline  = False,
        )
        fig.update_yaxes(
            gridcolor = COLORS["border"],
            showgrid  = True,
            zeroline  = False,
        )

        return json.dumps(fig.to_dict())

    # ── JAUGE WIN RATE ────────────────────────────────────────────

    @staticmethod
    def win_rate_gauge(win_rate: float, target: float = 0.60) -> str:
        """
        Jauge circulaire pour le Win Rate.
        Rouge < 50%, Orange 50-60%, Vert ≥ 60%.
        """
        wr_pct = win_rate * 100

        color = (
            COLORS["green"] if win_rate >= target else
            COLORS["gold"]  if win_rate >= 0.50 else
            COLORS["red"]
        )

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = wr_pct,
            delta = {
                "reference": target * 100,
                "valueformat": ".1f",
                "suffix": "%",
            },
            number = {
                "suffix": "%",
                "font": {"size": 40, "color": color},
                "valueformat": ".1f",
            },
            gauge = {
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLORS["text_muted"],
                    "tickfont": {"color": COLORS["text_muted"]},
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": COLORS["bg_card"],
                "borderwidth": 2,
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 50],        "color": "rgba(255,76,76,0.15)"},
                    {"range": [50, target*100],"color": "rgba(255,215,0,0.15)"},
                    {"range": [target*100,100],"color": "rgba(0,200,150,0.15)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["gold"], "width": 3},
                    "thickness": 0.75,
                    "value": target * 100,
                },
            },
            title = {
                "text": "Win Rate",
                "font": {"size": 14, "color": COLORS["text_muted"]},
            },
        ))

        fig.update_layout(
            **LAYOUT_BASE,
            height = 280,
        )

        return json.dumps(fig.to_dict())

    # ── DISTRIBUTION TP/SL ────────────────────────────────────────

    @staticmethod
    def tp_sl_distribution(
        tp1: int, tp2: int, tp3: int, sl: int
    ) -> str:
        """Pie chart de la distribution des sorties TP1/TP2/TP3/SL."""
        total = tp1 + tp2 + tp3 + sl
        if total == 0:
            return ChartBuilder._empty_chart("Distribution TP/SL")

        labels = ["TP1 🥉", "TP2 🥈", "TP3 🏆", "SL 🔴"]
        values = [tp1, tp2, tp3, sl]
        colors = [
            "rgba(0,200,150,0.7)",
            "rgba(0,200,150,0.9)",
            COLORS["gold"],
            "rgba(255,76,76,0.7)",
        ]

        fig = go.Figure(go.Pie(
            labels          = labels,
            values          = values,
            marker_colors   = colors,
            hole            = 0.45,
            textfont        = dict(size=13, color=COLORS["text"]),
            hovertemplate   = (
                "<b>%{label}</b><br>"
                "Trades: %{value}<br>"
                "Part: %{percent}<extra></extra>"
            ),
        ))

        fig.update_layout(
            **LAYOUT_BASE,
            title  = dict(text="Distribution des Sorties", font=dict(size=14)),
            height = 280,
            annotations=[dict(
                text      = f"{total}<br>trades",
                x=0.5, y=0.5,
                font_size = 16,
                font_color= COLORS["text"],
                showarrow = False,
            )],
        )

        return json.dumps(fig.to_dict())

    # ── PERFORMANCE PAR SESSION ───────────────────────────────────

    @staticmethod
    def session_performance(by_session: dict) -> str:
        """
        Bar chart horizontal — Win Rate par session de trading.
        Chaque barre est colorée selon sa performance.
        """
        if not by_session:
            return ChartBuilder._empty_chart("Performance par Session")

        sessions = list(by_session.keys())
        wr_vals  = [by_session[s].get("win_rate", 0) * 100 for s in sessions]
        trades   = [by_session[s].get("trades", 0) for s in sessions]
        pnls     = [by_session[s].get("pnl", 0) for s in sessions]

        bar_colors = [
            COLORS["green"] if wr >= 60 else
            COLORS["gold"]  if wr >= 50 else
            COLORS["red"]
            for wr in wr_vals
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x            = wr_vals,
            y            = [s.upper() for s in sessions],
            orientation  = "h",
            marker_color = bar_colors,
            text         = [
                f"{wr:.1f}% ({t} trades)"
                for wr, t in zip(wr_vals, trades)
            ],
            textposition = "outside",
            textfont     = dict(color=COLORS["text"], size=12),
            hovertemplate = (
                "<b>%{y}</b><br>"
                "Win Rate: %{x:.1f}%<br>"
                "Trades: %{customdata[0]}<br>"
                "P&L: %{customdata[1]:+.2f} USD<extra></extra>"
            ),
            customdata   = list(zip(trades, pnls)),
        ))

        fig.add_vline(
            x=60,
            line_dash  = "dash",
            line_color = COLORS["gold"],
            annotation_text = "Objectif 60%",
            annotation_font_color = COLORS["gold"],
        )

        fig.update_layout(
            **LAYOUT_BASE,
            title  = dict(text="Win Rate par Session", font=dict(size=14)),
            height = 250,
            xaxis  = dict(
                range     = [0, 100],
                ticksuffix = "%",
                gridcolor = COLORS["border"],
            ),
            yaxis  = dict(gridcolor=COLORS["border"]),
            showlegend = False,
        )

        return json.dumps(fig.to_dict())

    # ── PERFORMANCE PAR SCORE ─────────────────────────────────────

    @staticmethod
    def score_performance(by_score_range: dict) -> str:
        """
        Bar chart — Win Rate par tranche de score de confluence.
        Montre si les scores élevés performent mieux.
        """
        if not by_score_range:
            return ChartBuilder._empty_chart("WR par Score")

        # Filtre les tranches avec des trades
        ranges = {
            k: v for k, v in by_score_range.items()
            if v.get("trades", 0) > 0
        }

        if not ranges:
            return ChartBuilder._empty_chart("WR par Score")

        keys    = list(ranges.keys())
        wr_vals = [ranges[k].get("win_rate", 0) * 100 for k in keys]
        trades  = [ranges[k].get("trades", 0) for k in keys]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x            = keys,
            y            = wr_vals,
            marker_color = [
                COLORS["green"] if wr >= 60 else
                COLORS["gold"]  if wr >= 50 else
                COLORS["red"]
                for wr in wr_vals
            ],
            text         = [f"{wr:.1f}%<br>({t} trades)"
                           for wr, t in zip(wr_vals, trades)],
            textposition = "outside",
            textfont     = dict(color=COLORS["text"], size=11),
            hovertemplate = (
                "<b>Score %{x}</b><br>"
                "Win Rate: %{y:.1f}%<br>"
                "Trades: %{customdata}<extra></extra>"
            ),
            customdata   = trades,
        ))

        fig.add_hline(
            y=60,
            line_dash  = "dash",
            line_color = COLORS["gold"],
        )

        fig.update_layout(
            **LAYOUT_BASE,
            title  = dict(text="Win Rate par Score de Confluence", font=dict(size=14)),
            height = 280,
            xaxis  = dict(
                title     = "Tranche de score",
                gridcolor = COLORS["border"],
            ),
            yaxis  = dict(
                title     = "Win Rate %",
                range     = [0, 105],
                ticksuffix = "%",
                gridcolor = COLORS["border"],
            ),
            showlegend = False,
        )

        return json.dumps(fig.to_dict())

    # ── SCATTER TRADES ────────────────────────────────────────────

    @staticmethod
    def trades_scatter(report: PerformanceReport) -> str:
        """
        Graphique P&L cumulatif dans le temps.
        Points colorés : vert = win, rouge = loss.
        """
        return ChartBuilder._empty_chart("Historique des Trades")

    # ── HEATMAP MENSUELLE ─────────────────────────────────────────

    @staticmethod
    def monthly_heatmap(monthly_stats: list) -> str:
        """
        Heatmap des P&L mensuels.
        Vert = profitable, Rouge = en perte.
        """
        if not monthly_stats:
            return ChartBuilder._empty_chart("Performance Mensuelle")

        months  = [m.month for m in monthly_stats]
        pnls    = [m.pnl_usd for m in monthly_stats]
        wr_list = [m.win_rate * 100 for m in monthly_stats]

        colors = [
            COLORS["green"] if p >= 0 else COLORS["red"]
            for p in pnls
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x            = months,
            y            = pnls,
            marker_color = colors,
            text         = [
                f"{'+' if p >= 0 else ''}{p:.0f}$<br>{wr:.0f}%"
                for p, wr in zip(pnls, wr_list)
            ],
            textposition = "outside",
            textfont     = dict(color=COLORS["text"], size=11),
            hovertemplate = (
                "<b>%{x}</b><br>"
                "P&L: %{y:+.2f} USD<br>"
                "Win Rate: %{customdata:.1f}%<extra></extra>"
            ),
            customdata   = wr_list,
        ))

        fig.add_hline(y=0, line_color=COLORS["border"])

        fig.update_layout(
            **LAYOUT_BASE,
            title  = dict(text="P&L Mensuel", font=dict(size=14)),
            height = 300,
            xaxis  = dict(gridcolor=COLORS["border"]),
            yaxis  = dict(
                gridcolor  = COLORS["border"],
                ticksuffix = " $",
            ),
            showlegend = False,
        )

        return json.dumps(fig.to_dict())

    # ── WIN RATE ROLLING ─────────────────────────────────────────

    @staticmethod
    def rolling_win_rate(
        equity_points: list[EquityPoint],
        window: int = 20,
    ) -> str:
        """
        Win Rate glissant sur les N derniers trades.
        Montre si les performances s'améliorent.
        """
        if not equity_points or len(equity_points) < window:
            return ChartBuilder._empty_chart(f"WR Glissant ({window} trades)")

        timestamps  = [p.timestamp  for p in equity_points]
        drawdowns   = [p.drawdown_pct for p in equity_points]

        fig = go.Figure()

        # Drawdown comme proxy de la tendance
        fig.add_trace(go.Scatter(
            x     = timestamps,
            y     = [100 - d for d in drawdowns],
            name  = "Santé du système",
            line  = dict(color=COLORS["blue"], width=2),
            fill  = "tozeroy",
            fillcolor = "rgba(76, 155, 232, 0.1)",
            hovertemplate = (
                "<b>%{x|%d/%m}</b><br>"
                "Santé: %{y:.1f}%<extra></extra>"
            ),
        ))

        fig.add_hline(
            y=60,
            line_dash  = "dash",
            line_color = COLORS["gold"],
            annotation_text = "Objectif 60%",
        )

        fig.update_layout(
            **LAYOUT_BASE,
            title  = dict(
                text = f"Santé du Système (100% - Drawdown)",
                font = dict(size=14),
            ),
            height = 250,
            xaxis  = dict(gridcolor=COLORS["border"]),
            yaxis  = dict(
                gridcolor  = COLORS["border"],
                range      = [0, 105],
                ticksuffix = "%",
            ),
            showlegend = False,
        )

        return json.dumps(fig.to_dict())

    # ── UTILITAIRES ───────────────────────────────────────────────

    @staticmethod
    def _empty_chart(title: str) -> str:
        """Graphique vide avec message d'attente."""
        fig = go.Figure()
        fig.add_annotation(
            text      = "Aucune donnée disponible",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow = False,
            font      = dict(color=COLORS["text_muted"], size=14),
        )
        fig.update_layout(
            **LAYOUT_BASE,
            title  = dict(text=title, font=dict(size=14)),
            height = 250,
        )
        return json.dumps(fig.to_dict())

    @staticmethod
    def kpi_cards(report: PerformanceReport) -> dict:
        """
        Retourne les KPIs formatés pour les cartes du dashboard.
        Dict directement utilisable dans le template HTML.
        """
        wr_color = (
            "green"  if report.win_rate >= 0.60 else
            "yellow" if report.win_rate >= 0.50 else
            "red"
        )

        pnl_sign  = "+" if report.total_pnl_usd >= 0 else ""
        pnl_color = "green" if report.total_pnl_usd >= 0 else "red"

        pf_color = (
            "green"  if report.profit_factor >= 1.5 else
            "yellow" if report.profit_factor >= 1.0 else
            "red"
        )

        return {
            "win_rate": {
                "value": f"{report.win_rate:.1%}",
                "color": wr_color,
                "label": "Win Rate",
                "sub":   f"{report.winning_trades}W / {report.losing_trades}L",
                "target": "Cible: 60%",
                "ok":    report.win_rate >= 0.60,
            },
            "profit_factor": {
                "value": f"{report.profit_factor:.2f}",
                "color": pf_color,
                "label": "Profit Factor",
                "sub":   f"Sharpe: {report.sharpe_ratio:.2f}",
                "target": "Cible: > 1.3",
                "ok":    report.profit_factor >= 1.3,
            },
            "total_pnl": {
                "value": f"{pnl_sign}{report.total_pnl_usd:.2f} $",
                "color": pnl_color,
                "label": "P&L Total",
                "sub":   f"Espérance: {report.expectancy_usd:+.2f} $/trade",
                "target": "",
                "ok":    report.total_pnl_usd >= 0,
            },
            "drawdown": {
                "value": f"{report.max_drawdown_pct:.1f}%",
                "color": "green" if report.max_drawdown_pct < 10 else "yellow" if report.max_drawdown_pct < 20 else "red",
                "label": "Max Drawdown",
                "sub":   f"Actuel: {report.current_drawdown:.1f}%",
                "target": "Cible: < 15%",
                "ok":    report.max_drawdown_pct < 15,
            },
            "total_trades": {
                "value": str(report.total_trades),
                "color": "blue",
                "label": "Trades",
                "sub":   f"Signaux: {report.total_signals}",
                "target": "",
                "ok":    True,
            },
            "streak": {
                "value": (
                    f"+{report.current_streak}" if report.current_streak >= 0
                    else str(report.current_streak)
                ),
                "color": "green" if report.current_streak >= 0 else "red",
                "label": "Série courante",
                "sub":   (
                    f"Max: +{report.max_win_streak} / -{report.max_loss_streak}"
                ),
                "target": "",
                "ok":    report.current_streak >= 0,
            },
        }
