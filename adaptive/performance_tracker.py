# adaptive/performance_tracker.py
# ─────────────────────────────────────────────────────────────────
#  Tracker de performance — Mémoire du bot
#
#  Calcule en temps réel :
#   • Win Rate global et par instrument
#   • Profit Factor (gains bruts / pertes brutes)
#   • Sharpe Ratio (rendement ajusté au risque)
#   • Drawdown maximum
#   • Séries de gains/pertes consécutifs
#   • Performance par session (London, NY, Overlap)
#   • Performance par score de confluence
# ─────────────────────────────────────────────────────────────────

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from database.db_manager import DatabaseManager
from database.models import Signal, SignalStatus


@dataclass
class PerformanceReport:
    # ── Période ───────────────────────────────────────────────────
    period_days:       int
    from_date:         datetime
    to_date:           datetime
    instrument:        str | None     # None = global

    # ── Métriques principales ─────────────────────────────────────
    total_signals:     int
    total_trades:      int
    winning_trades:    int
    losing_trades:     int
    breakeven_trades:  int
    partial_trades:    int

    win_rate:          float    # Ex: 0.65 = 65%
    profit_factor:     float    # Gains bruts / Pertes brutes
    sharpe_ratio:      float    # Rendement / Volatilité

    # ── P&L ───────────────────────────────────────────────────────
    total_pnl_usd:     float
    avg_win_usd:       float
    avg_loss_usd:      float
    largest_win_usd:   float
    largest_loss_usd:  float
    expectancy_usd:    float    # Gain moyen par trade

    # ── Drawdown ──────────────────────────────────────────────────
    max_drawdown_pct:  float
    current_drawdown:  float

    # ── Séries ────────────────────────────────────────────────────
    max_win_streak:    int
    max_loss_streak:   int
    current_streak:    int      # Positif = gains, négatif = pertes

    # ── TP Distribution ───────────────────────────────────────────
    tp1_hit_count:     int
    tp2_hit_count:     int
    tp3_hit_count:     int
    sl_hit_count:      int

    # ── Par session ───────────────────────────────────────────────
    by_session:        dict = field(default_factory=dict)
    # ex: {"london": {"trades": 10, "wr": 0.7}, "ny": {...}}

    # ── Par score de confluence ───────────────────────────────────
    by_score_range:    dict = field(default_factory=dict)
    # ex: {"75-80": {"trades": 5, "wr": 0.6}, "81-90": {...}}

    # ── État ──────────────────────────────────────────────────────
    target_wr_reached: bool     # True si WR >= 60%
    generated_at:      datetime = field(default_factory=datetime.utcnow)


class PerformanceTracker:
    """
    Calcule les métriques de performance depuis la base de données.
    Utilisé par le dashboard, l'optimizer et les alertes Telegram.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    # ── RAPPORT PRINCIPAL ─────────────────────────────────────────

    async def generate_report(
        self,
        days:       int = 30,
        instrument: str | None = None,
    ) -> PerformanceReport:
        """
        Génère un rapport de performance complet.

        Args:
            days:       Période d'analyse en jours
            instrument: Filtrer sur un instrument (None = global)
        """
        since  = datetime.utcnow() - timedelta(days=days)
        signals = await self._db.get_signals_for_backtest(
            instrument = instrument,
            months     = max(1, days // 30),
        )

        # Filtrage sur la période exacte
        signals = [s for s in signals if s.created_at >= since]

        if not signals:
            return self._empty_report(days, instrument)

        # ── Calculs de base ───────────────────────────────────────
        finished = [
            s for s in signals
            if s.status in (
                SignalStatus.TP1_HIT,
                SignalStatus.TP2_HIT,
                SignalStatus.TP3_HIT,
                SignalStatus.SL_HIT,
            )
        ]

        wins = [
            s for s in finished
            if s.status in (
                SignalStatus.TP1_HIT,
                SignalStatus.TP2_HIT,
                SignalStatus.TP3_HIT,
            )
        ]
        losses = [s for s in finished if s.status == SignalStatus.SL_HIT]

        total_trades    = len(finished)
        winning_trades  = len(wins)
        losing_trades   = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # ── P&L ───────────────────────────────────────────────────
        win_pnls  = [s.pnl_usd for s in wins   if s.pnl_usd is not None]
        loss_pnls = [s.pnl_usd for s in losses  if s.pnl_usd is not None]

        gross_profit = sum(p for p in win_pnls  if p > 0)
        gross_loss   = abs(sum(p for p in loss_pnls if p < 0))

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        avg_win  = sum(win_pnls)  / len(win_pnls)   if win_pnls  else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls)  if loss_pnls else 0.0

        total_pnl = sum(
            s.pnl_usd for s in finished
            if s.pnl_usd is not None
        )

        expectancy = total_pnl / total_trades if total_trades > 0 else 0.0

        largest_win  = max(win_pnls,  default=0.0)
        largest_loss = min(loss_pnls, default=0.0)

        # ── Sharpe Ratio ──────────────────────────────────────────
        all_pnls = [
            s.pnl_usd for s in finished
            if s.pnl_usd is not None
        ]
        sharpe = self._calc_sharpe(all_pnls)

        # ── Drawdown ──────────────────────────────────────────────
        max_dd, curr_dd = self._calc_drawdown(finished)

        # ── Séries ────────────────────────────────────────────────
        max_win_streak, max_loss_streak, curr_streak = (
            self._calc_streaks(finished)
        )

        # ── Distribution TP/SL ────────────────────────────────────
        tp1_count = sum(1 for s in finished if s.status == SignalStatus.TP1_HIT)
        tp2_count = sum(1 for s in finished if s.status == SignalStatus.TP2_HIT)
        tp3_count = sum(1 for s in finished if s.status == SignalStatus.TP3_HIT)
        sl_count  = sum(1 for s in finished if s.status == SignalStatus.SL_HIT)

        # ── Par session ───────────────────────────────────────────
        by_session = self._calc_by_session(finished)

        # ── Par score de confluence ───────────────────────────────
        by_score = self._calc_by_score_range(finished)

        # ── Signaux PENDING / CANCELLED ───────────────────────────
        partial   = sum(1 for s in finished if getattr(s, "result", None) == "PARTIAL")
        breakeven = sum(1 for s in finished if getattr(s, "result", None) == "BREAKEVEN")

        report = PerformanceReport(
            period_days       = days,
            from_date         = since,
            to_date           = datetime.utcnow(),
            instrument        = instrument,
            total_signals     = len(signals),
            total_trades      = total_trades,
            winning_trades    = winning_trades,
            losing_trades     = losing_trades,
            breakeven_trades  = breakeven,
            partial_trades    = partial,
            win_rate          = round(win_rate, 4),
            profit_factor     = round(profit_factor, 2),
            sharpe_ratio      = round(sharpe, 2),
            total_pnl_usd     = round(total_pnl, 2),
            avg_win_usd       = round(avg_win, 2),
            avg_loss_usd      = round(avg_loss, 2),
            largest_win_usd   = round(largest_win, 2),
            largest_loss_usd  = round(largest_loss, 2),
            expectancy_usd    = round(expectancy, 2),
            max_drawdown_pct  = round(max_dd, 2),
            current_drawdown  = round(curr_dd, 2),
            max_win_streak    = max_win_streak,
            max_loss_streak   = max_loss_streak,
            current_streak    = curr_streak,
            tp1_hit_count     = tp1_count,
            tp2_hit_count     = tp2_count,
            tp3_hit_count     = tp3_count,
            sl_hit_count      = sl_count,
            by_session        = by_session,
            by_score_range    = by_score,
            target_wr_reached = win_rate >= 0.60,
        )

        logger.info(
            f"📊 Rapport {days}j {instrument or 'GLOBAL'} — "
            f"WR: {win_rate:.1%} | PF: {profit_factor:.2f} | "
            f"Trades: {total_trades} | P&L: {total_pnl:.2f} USD"
        )

        return report

    # ── MÉTRIQUES RAPIDES ─────────────────────────────────────────

    async def get_win_rate(
        self,
        days:       int = 30,
        instrument: str | None = None,
    ) -> float:
        """Win rate rapide sans rapport complet."""
        return await self._db.get_win_rate(instrument, days)

    async def is_performing(self, days: int = 7) -> bool:
        """
        True si le bot performe bien sur les 7 derniers jours.
        Critères : WR >= 60% ET PF >= 1.3
        """
        report = await self.generate_report(days=days)
        return (
            report.win_rate >= 0.60
            and report.profit_factor >= 1.3
            and report.total_trades >= 5
        )

    # ── CALCULS STATISTIQUES ──────────────────────────────────────

    @staticmethod
    def _calc_sharpe(pnls: list[float], risk_free: float = 0.0) -> float:
        """
        Sharpe Ratio simplifié.
        Sharpe = (moyenne - risk_free) / écart-type
        """
        if len(pnls) < 2:
            return 0.0

        mean = sum(pnls) / len(pnls)
        variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std_dev  = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return 0.0

        return (mean - risk_free) / std_dev

    @staticmethod
    def _calc_drawdown(signals: list) -> tuple[float, float]:
        """
        Calcule le drawdown maximum et courant en %.
        Basé sur la courbe des P&L cumulés.
        """
        if not signals:
            return 0.0, 0.0

        equity       = 0.0
        peak         = 0.0
        max_dd       = 0.0

        for sig in sorted(signals, key=lambda x: x.created_at):
            pnl    = sig.pnl_usd or 0.0
            equity += pnl
            peak   = max(peak, equity)

            if peak > 0:
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)

        # Drawdown courant
        curr_dd = (peak - equity) / peak * 100 if peak > 0 else 0.0

        return max_dd, curr_dd

    @staticmethod
    def _calc_streaks(signals: list) -> tuple[int, int, int]:
        """
        Calcule les séries max de gains et pertes.
        Returns: (max_win_streak, max_loss_streak, current_streak)
        """
        if not signals:
            return 0, 0, 0

        win_statuses = {
            SignalStatus.TP1_HIT,
            SignalStatus.TP2_HIT,
            SignalStatus.TP3_HIT,
        }

        ordered = sorted(signals, key=lambda x: x.created_at)
        max_wins  = 0
        max_losses = 0
        curr_wins  = 0
        curr_losses = 0

        for sig in ordered:
            if sig.status in win_statuses:
                curr_wins  += 1
                curr_losses = 0
                max_wins    = max(max_wins, curr_wins)
            elif sig.status == SignalStatus.SL_HIT:
                curr_losses += 1
                curr_wins    = 0
                max_losses   = max(max_losses, curr_losses)

        # Streak courant (+ = gains, - = pertes)
        current_streak = curr_wins if curr_wins > 0 else -curr_losses

        return max_wins, max_losses, current_streak

    @staticmethod
    def _calc_by_session(signals: list) -> dict:
        """Performance par session de trading."""
        sessions: dict[str, dict] = {}
        win_statuses = {
            SignalStatus.TP1_HIT,
            SignalStatus.TP2_HIT,
            SignalStatus.TP3_HIT,
        }

        for sig in signals:
            sess = sig.session or "unknown"
            if sess not in sessions:
                sessions[sess] = {"trades": 0, "wins": 0, "pnl": 0.0}

            sessions[sess]["trades"] += 1
            if sig.status in win_statuses:
                sessions[sess]["wins"] += 1
            sessions[sess]["pnl"] += sig.pnl_usd or 0.0

        # Calcul WR par session
        for sess_data in sessions.values():
            t = sess_data["trades"]
            w = sess_data["wins"]
            sess_data["win_rate"] = round(w / t, 4) if t > 0 else 0.0
            sess_data["pnl"]      = round(sess_data["pnl"], 2)

        return sessions

    @staticmethod
    def _calc_by_score_range(signals: list) -> dict:
        """
        Performance par tranche de score de confluence.
        Tranches : 75-79, 80-84, 85-89, 90-94, 95-100
        """
        ranges: dict[str, dict] = {
            "75-79":  {"trades": 0, "wins": 0},
            "80-84":  {"trades": 0, "wins": 0},
            "85-89":  {"trades": 0, "wins": 0},
            "90-94":  {"trades": 0, "wins": 0},
            "95-100": {"trades": 0, "wins": 0},
        }
        win_statuses = {
            SignalStatus.TP1_HIT,
            SignalStatus.TP2_HIT,
            SignalStatus.TP3_HIT,
        }

        for sig in signals:
            score = sig.confluence_score or 0
            if score < 75:
                continue

            key = (
                "95-100" if score >= 95 else
                "90-94"  if score >= 90 else
                "85-89"  if score >= 85 else
                "80-84"  if score >= 80 else
                "75-79"
            )
            ranges[key]["trades"] += 1
            if sig.status in win_statuses:
                ranges[key]["wins"] += 1

        for r, data in ranges.items():
            t = data["trades"]
            data["win_rate"] = round(data["wins"] / t, 4) if t > 0 else 0.0

        return ranges

    # ── RAPPORT VIDE ──────────────────────────────────────────────

    @staticmethod
    def _empty_report(days: int, instrument: str | None) -> PerformanceReport:
        """Retourne un rapport vide (aucun trade sur la période)."""
        now = datetime.utcnow()
        return PerformanceReport(
            period_days=days,
            from_date=now - timedelta(days=days),
            to_date=now,
            instrument=instrument,
            total_signals=0, total_trades=0,
            winning_trades=0, losing_trades=0,
            breakeven_trades=0, partial_trades=0,
            win_rate=0.0, profit_factor=0.0, sharpe_ratio=0.0,
            total_pnl_usd=0.0, avg_win_usd=0.0, avg_loss_usd=0.0,
            largest_win_usd=0.0, largest_loss_usd=0.0, expectancy_usd=0.0,
            max_drawdown_pct=0.0, current_drawdown=0.0,
            max_win_streak=0, max_loss_streak=0, current_streak=0,
            tp1_hit_count=0, tp2_hit_count=0,
            tp3_hit_count=0, sl_hit_count=0,
            target_wr_reached=False,
        )

    # ── SNAPSHOT JOURNALIER ───────────────────────────────────────

    async def save_daily_snapshot(self):
        """
        Sauvegarde un snapshot de performance journalier en DB.
        À appeler à minuit UTC.
        """
        report = await self.generate_report(days=1)

        await self._db.save_performance_snapshot({
            "period":       "daily",
            "instrument":   None,
            "total_signals":  report.total_signals,
            "total_trades":   report.total_trades,
            "winning_trades": report.winning_trades,
            "losing_trades":  report.losing_trades,
            "win_rate":       report.win_rate,
            "profit_factor":  report.profit_factor,
            "sharpe_ratio":   report.sharpe_ratio,
            "total_pnl_usd":  report.total_pnl_usd,
            "avg_win_usd":    report.avg_win_usd,
            "avg_loss_usd":   report.avg_loss_usd,
            "max_drawdown":   report.max_drawdown_pct,
        })

        logger.info("💾 Snapshot journalier sauvegardé")
