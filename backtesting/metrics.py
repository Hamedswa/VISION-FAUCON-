# backtesting/metrics.py
# ─────────────────────────────────────────────────────────────────
#  Métriques de backtesting — Évaluation mathématique pure
#
#  Métriques calculées :
#   • Win Rate, Profit Factor, Sharpe Ratio, Sortino Ratio
#   • Calmar Ratio (rendement / max drawdown)
#   • Expectancy (gain moyen par trade)
#   • MAE / MFE (adverse/favorable excursion)
#   • Courbe d'équité + drawdown continu
#   • Analyse par période (mensuel)
# ─────────────────────────────────────────────────────────────────

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeRecord:
    """Représente un trade individual dans le backtest."""
    signal_id:    int
    instrument:   str
    direction:    str           # "bullish" | "bearish"
    entry:        float
    exit_price:   float
    sl:           float
    tp1:          float
    tp2:          float
    tp3:          float
    pnl_usd:      float
    pnl_pips:     float
    is_win:       bool
    tp_level_hit: int | None    # 1, 2, 3 ou None (SL)
    duration_min: int
    score:        int
    session:      str | None
    opened_at:    datetime
    closed_at:    datetime
    rr_ratio:     float
    rr_achieved:  float         # R:R réellement obtenu


@dataclass
class EquityPoint:
    """Point sur la courbe d'équité."""
    timestamp:     datetime
    equity:        float
    drawdown_pct:  float
    trade_count:   int


@dataclass
class MonthlyStats:
    """Statistiques mensuelles."""
    month:         str    # "2024-01"
    trades:        int
    wins:          int
    win_rate:      float
    pnl_usd:       float
    profit_factor: float


@dataclass
class MetricsResult:
    # ── Identification ────────────────────────────────────────────
    instrument:        str | None
    period_from:       datetime
    period_to:         datetime
    total_bars:        int

    # ── Volume ────────────────────────────────────────────────────
    total_signals:     int
    total_trades:      int
    winning_trades:    int
    losing_trades:     int

    # ── Ratios principaux ─────────────────────────────────────────
    win_rate:          float
    profit_factor:     float
    sharpe_ratio:      float
    sortino_ratio:     float
    calmar_ratio:      float

    # ── P&L ───────────────────────────────────────────────────────
    total_pnl_usd:     float
    avg_win_usd:       float
    avg_loss_usd:      float
    largest_win_usd:   float
    largest_loss_usd:  float
    expectancy_usd:    float
    total_pnl_pips:    float

    # ── Risque ────────────────────────────────────────────────────
    max_drawdown_pct:  float
    avg_drawdown_pct:  float
    recovery_factor:   float    # total_pnl / max_drawdown
    max_loss_streak:   int
    max_win_streak:    int

    # ── Durée ─────────────────────────────────────────────────────
    avg_duration_min:  float
    avg_win_dur_min:   float
    avg_loss_dur_min:  float

    # ── TP Distribution ───────────────────────────────────────────
    tp1_rate:          float    # % qui atteignent TP1
    tp2_rate:          float
    tp3_rate:          float
    sl_rate:           float

    # ── R:R ───────────────────────────────────────────────────────
    avg_rr_planned:    float
    avg_rr_achieved:   float

    # ── Courbe d'équité ───────────────────────────────────────────
    equity_curve:      list[EquityPoint] = field(default_factory=list)
    monthly_stats:     list[MonthlyStats] = field(default_factory=list)

    # ── Qualité globale ───────────────────────────────────────────
    grade:             str = ""   # A+, A, B, C, D
    summary:           str = ""


class BacktestMetrics:
    """
    Calcule toutes les métriques de performance
    à partir d'une liste de TradeRecord.
    100% mathématique, aucune dépendance externe.
    """

    def calculate(
        self,
        trades:      list[TradeRecord],
        instrument:  str | None = None,
        initial_balance: float = 10000.0,
    ) -> MetricsResult:
        """
        Calcule toutes les métriques sur une liste de trades.

        Args:
            trades:          Liste des trades du backtest
            instrument:      Instrument analysé (None = global)
            initial_balance: Capital de départ en USD

        Returns:
            MetricsResult complet
        """
        if not trades:
            return self._empty_result(instrument)

        trades_sorted = sorted(trades, key=lambda t: t.opened_at)

        wins   = [t for t in trades_sorted if t.is_win]
        losses = [t for t in trades_sorted if not t.is_win]

        total  = len(trades_sorted)
        n_wins = len(wins)

        # ── Ratios de base ────────────────────────────────────────
        win_rate = n_wins / total

        win_pnls  = [t.pnl_usd for t in wins]
        loss_pnls = [t.pnl_usd for t in losses]

        gross_profit = sum(p for p in win_pnls  if p > 0)
        gross_loss   = abs(sum(p for p in loss_pnls if p < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        total_pnl  = sum(t.pnl_usd for t in trades_sorted)
        total_pips = sum(t.pnl_pips for t in trades_sorted)

        avg_win  = sum(win_pnls)  / n_wins        if wins   else 0.0
        avg_loss = sum(loss_pnls) / len(losses)   if losses else 0.0

        largest_win  = max(win_pnls,  default=0.0)
        largest_loss = min(loss_pnls, default=0.0)

        expectancy = total_pnl / total

        # ── Sharpe & Sortino ──────────────────────────────────────
        all_pnls = [t.pnl_usd for t in trades_sorted]
        sharpe   = self._sharpe(all_pnls)
        sortino  = self._sortino(all_pnls)

        # ── Courbe d'équité & Drawdown ────────────────────────────
        equity_curve, max_dd, avg_dd = self._equity_curve(
            trades_sorted, initial_balance
        )

        # ── Calmar Ratio ──────────────────────────────────────────
        calmar = (
            (total_pnl / initial_balance * 100) / max_dd
            if max_dd > 0 else float("inf")
        )

        # ── Recovery Factor ───────────────────────────────────────
        recovery = (
            total_pnl / (initial_balance * max_dd / 100)
            if max_dd > 0 else float("inf")
        )

        # ── Séries ────────────────────────────────────────────────
        max_win_streak, max_loss_streak = self._streaks(trades_sorted)

        # ── Durées ────────────────────────────────────────────────
        avg_dur      = statistics.mean([t.duration_min for t in trades_sorted]) if trades_sorted else 0
        avg_win_dur  = statistics.mean([t.duration_min for t in wins])   if wins   else 0
        avg_loss_dur = statistics.mean([t.duration_min for t in losses]) if losses else 0

        # ── Distribution TP ───────────────────────────────────────
        tp1_count = sum(1 for t in trades_sorted if t.tp_level_hit == 1)
        tp2_count = sum(1 for t in trades_sorted if t.tp_level_hit == 2)
        tp3_count = sum(1 for t in trades_sorted if t.tp_level_hit == 3)
        sl_count  = sum(1 for t in trades_sorted if not t.is_win)

        tp1_rate = tp1_count / total
        tp2_rate = tp2_count / total
        tp3_rate = tp3_count / total
        sl_rate  = sl_count  / total

        # ── R:R ───────────────────────────────────────────────────
        avg_rr_planned  = statistics.mean([t.rr_ratio    for t in trades_sorted]) if trades_sorted else 0
        avg_rr_achieved = statistics.mean([t.rr_achieved for t in trades_sorted]) if trades_sorted else 0

        # ── Stats mensuelles ──────────────────────────────────────
        monthly = self._monthly_stats(trades_sorted)

        # ── Période ───────────────────────────────────────────────
        period_from = trades_sorted[0].opened_at
        period_to   = trades_sorted[-1].closed_at

        # ── Grade ─────────────────────────────────────────────────
        grade   = self._grade(win_rate, profit_factor, max_dd, sharpe)
        summary = self._build_summary(
            win_rate, profit_factor, total_pnl,
            max_dd, sharpe, grade, total
        )

        return MetricsResult(
            instrument       = instrument,
            period_from      = period_from,
            period_to        = period_to,
            total_bars       = 0,
            total_signals    = total,
            total_trades     = total,
            winning_trades   = n_wins,
            losing_trades    = len(losses),
            win_rate         = round(win_rate, 4),
            profit_factor    = round(profit_factor, 2),
            sharpe_ratio     = round(sharpe, 2),
            sortino_ratio    = round(sortino, 2),
            calmar_ratio     = round(calmar, 2),
            total_pnl_usd    = round(total_pnl, 2),
            avg_win_usd      = round(avg_win, 2),
            avg_loss_usd     = round(avg_loss, 2),
            largest_win_usd  = round(largest_win, 2),
            largest_loss_usd = round(largest_loss, 2),
            expectancy_usd   = round(expectancy, 2),
            total_pnl_pips   = round(total_pips, 1),
            max_drawdown_pct = round(max_dd, 2),
            avg_drawdown_pct = round(avg_dd, 2),
            recovery_factor  = round(recovery, 2),
            max_loss_streak  = max_loss_streak,
            max_win_streak   = max_win_streak,
            avg_duration_min = round(avg_dur, 1),
            avg_win_dur_min  = round(avg_win_dur, 1),
            avg_loss_dur_min = round(avg_loss_dur, 1),
            tp1_rate         = round(tp1_rate, 4),
            tp2_rate         = round(tp2_rate, 4),
            tp3_rate         = round(tp3_rate, 4),
            sl_rate          = round(sl_rate, 4),
            avg_rr_planned   = round(avg_rr_planned, 2),
            avg_rr_achieved  = round(avg_rr_achieved, 2),
            equity_curve     = equity_curve,
            monthly_stats    = monthly,
            grade            = grade,
            summary          = summary,
        )

    # ── COURBE D'ÉQUITÉ ───────────────────────────────────────────

    @staticmethod
    def _equity_curve(
        trades:          list[TradeRecord],
        initial_balance: float,
    ) -> tuple[list[EquityPoint], float, float]:
        """
        Construit la courbe d'équité et calcule le drawdown.

        Returns:
            (equity_curve, max_drawdown_pct, avg_drawdown_pct)
        """
        equity       = initial_balance
        peak         = initial_balance
        max_dd       = 0.0
        drawdowns    = []
        curve        = []

        for i, trade in enumerate(trades):
            equity += trade.pnl_usd
            peak    = max(peak, equity)

            dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
            drawdowns.append(dd)
            max_dd = max(max_dd, dd)

            curve.append(EquityPoint(
                timestamp    = trade.closed_at,
                equity       = round(equity, 2),
                drawdown_pct = round(dd, 2),
                trade_count  = i + 1,
            ))

        avg_dd = statistics.mean(drawdowns) if drawdowns else 0.0
        return curve, max_dd, avg_dd

    # ── SHARPE RATIO ──────────────────────────────────────────────

    @staticmethod
    def _sharpe(pnls: list[float], risk_free: float = 0.0) -> float:
        """
        Sharpe = (E[R] - Rf) / σ(R)
        Annualisé en supposant ~250 trades/an.
        """
        if len(pnls) < 2:
            return 0.0
        mean = statistics.mean(pnls)
        std  = statistics.stdev(pnls)
        if std == 0:
            return 0.0
        # Facteur d'annualisation (racine de 250 trades/an)
        annual_factor = math.sqrt(250)
        return ((mean - risk_free) / std) * annual_factor

    # ── SORTINO RATIO ─────────────────────────────────────────────

    @staticmethod
    def _sortino(pnls: list[float], risk_free: float = 0.0) -> float:
        """
        Sortino = (E[R] - Rf) / σ_downside
        Utilise uniquement la volatilité des pertes (downside).
        """
        if len(pnls) < 2:
            return 0.0

        mean         = statistics.mean(pnls)
        losses_only  = [p for p in pnls if p < risk_free]

        if not losses_only:
            return float("inf")

        # Downside deviation
        downside_var = sum((p - risk_free) ** 2 for p in losses_only) / len(pnls)
        downside_std = math.sqrt(downside_var)

        if downside_std == 0:
            return 0.0

        annual_factor = math.sqrt(250)
        return ((mean - risk_free) / downside_std) * annual_factor

    # ── SÉRIES ────────────────────────────────────────────────────

    @staticmethod
    def _streaks(trades: list[TradeRecord]) -> tuple[int, int]:
        """Calcule les séries max de gains et pertes consécutifs."""
        max_wins  = 0
        max_losses = 0
        curr_wins  = 0
        curr_losses = 0

        for t in trades:
            if t.is_win:
                curr_wins  += 1
                curr_losses = 0
                max_wins    = max(max_wins, curr_wins)
            else:
                curr_losses += 1
                curr_wins   = 0
                max_losses  = max(max_losses, curr_losses)

        return max_wins, max_losses

    # ── STATS MENSUELLES ──────────────────────────────────────────

    @staticmethod
    def _monthly_stats(trades: list[TradeRecord]) -> list[MonthlyStats]:
        """Groupe les trades par mois et calcule les métriques."""
        monthly: dict[str, list[TradeRecord]] = {}

        for t in trades:
            key = t.opened_at.strftime("%Y-%m")
            monthly.setdefault(key, []).append(t)

        result = []
        for month, month_trades in sorted(monthly.items()):
            wins      = [t for t in month_trades if t.is_win]
            losses    = [t for t in month_trades if not t.is_win]
            wr        = len(wins) / len(month_trades)
            total_pnl = sum(t.pnl_usd for t in month_trades)

            gross_profit = sum(t.pnl_usd for t in wins if t.pnl_usd > 0)
            gross_loss   = abs(sum(t.pnl_usd for t in losses if t.pnl_usd < 0))
            pf           = gross_profit / gross_loss if gross_loss > 0 else 999.0

            result.append(MonthlyStats(
                month         = month,
                trades        = len(month_trades),
                wins          = len(wins),
                win_rate      = round(wr, 4),
                pnl_usd       = round(total_pnl, 2),
                profit_factor = round(pf, 2),
            ))

        return result

    # ── GRADE ─────────────────────────────────────────────────────

    @staticmethod
    def _grade(
        wr:  float,
        pf:  float,
        dd:  float,
        sharpe: float,
    ) -> str:
        """
        Attribue une note globale au backtest.

        A+ : WR ≥ 70%, PF ≥ 2.0, DD < 10%, Sharpe ≥ 1.5
        A  : WR ≥ 65%, PF ≥ 1.7, DD < 15%, Sharpe ≥ 1.0
        B  : WR ≥ 60%, PF ≥ 1.4, DD < 20%
        C  : WR ≥ 55%, PF ≥ 1.2, DD < 25%
        D  : En dessous
        """
        score = 0

        # Win Rate
        if wr >= 0.70: score += 4
        elif wr >= 0.65: score += 3
        elif wr >= 0.60: score += 2
        elif wr >= 0.55: score += 1

        # Profit Factor
        if pf >= 2.0: score += 4
        elif pf >= 1.7: score += 3
        elif pf >= 1.4: score += 2
        elif pf >= 1.2: score += 1

        # Drawdown (inversé)
        if dd < 10: score += 3
        elif dd < 15: score += 2
        elif dd < 20: score += 1

        # Sharpe
        if sharpe >= 1.5: score += 2
        elif sharpe >= 1.0: score += 1

        if score >= 12: return "A+"
        if score >= 9:  return "A"
        if score >= 6:  return "B"
        if score >= 3:  return "C"
        return "D"

    # ── RÉSUMÉ ────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        wr: float, pf: float, total_pnl: float,
        max_dd: float, sharpe: float,
        grade: str, total: int,
    ) -> str:
        return (
            f"Grade {grade} — {total} trades\n"
            f"WR: {wr:.1%} | PF: {pf:.2f} | "
            f"Sharpe: {sharpe:.2f} | DD max: {max_dd:.1f}%\n"
            f"P&L total: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f} USD"
        )

    # ── RÉSULTAT VIDE ─────────────────────────────────────────────

    @staticmethod
    def _empty_result(instrument: str | None) -> MetricsResult:
        now = datetime.utcnow()
        return MetricsResult(
            instrument=instrument, period_from=now, period_to=now,
            total_bars=0, total_signals=0, total_trades=0,
            winning_trades=0, losing_trades=0,
            win_rate=0.0, profit_factor=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0,
            total_pnl_usd=0.0, avg_win_usd=0.0, avg_loss_usd=0.0,
            largest_win_usd=0.0, largest_loss_usd=0.0,
            expectancy_usd=0.0, total_pnl_pips=0.0,
            max_drawdown_pct=0.0, avg_drawdown_pct=0.0,
            recovery_factor=0.0, max_loss_streak=0, max_win_streak=0,
            avg_duration_min=0.0, avg_win_dur_min=0.0,
            avg_loss_dur_min=0.0,
            tp1_rate=0.0, tp2_rate=0.0, tp3_rate=0.0, sl_rate=0.0,
            avg_rr_planned=0.0, avg_rr_achieved=0.0,
            grade="D", summary="Aucun trade sur la période",
        )
