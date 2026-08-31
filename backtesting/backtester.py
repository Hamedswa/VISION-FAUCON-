# backtesting/backtester.py
# ─────────────────────────────────────────────────────────────────
#  Moteur de backtesting — Simulation historique complète
#
#  Stratégie de simulation :
#   1. Charge les données OHLCV historiques (Twelve Data via DB)
#   2. Rejoue le pipeline SMC/ICT sur chaque bougie
#   3. Simule l'exécution (market order au open de la bougie suivante)
#   4. Suit l'évolution TP1/TP2/TP3/SL bougie par bougie
#   5. Applique la gestion : clôture 50% à TP1, SL → BE
#   6. Calcule toutes les métriques via BacktestMetrics
#
#  Mode walk-forward :
#   Divise la période en segments et optimise sur chacun
#   pour tester si l'optimizer généralise bien.
# ─────────────────────────────────────────────────────────────────

import asyncio
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from config import settings
from config.instruments import get_instrument, INSTRUMENTS
from core import (
    TechnicalIndicators,
    SMCDetector,
    ScoringEngine,
    TPSLCalculator,
    EntryValidator,
)
from data.data_manager import DataManager
from database.db_manager import DatabaseManager
from .metrics import BacktestMetrics, MetricsResult, TradeRecord


@dataclass
class BacktestConfig:
    """Configuration d'un backtest."""
    instrument:         str
    months:             int = 6
    initial_balance:    float = 10000.0
    risk_pct:           float = 1.0

    # Paramètres du moteur (peuvent être surchargés par l'optimizer)
    min_score:          int   = 75
    atr_sl_multiplier:  float = 1.5
    tp1_ratio:          float = 1.0
    tp2_ratio:          float = 1.618
    tp3_ratio:          float = 2.5
    min_rr:             float = 1.5
    swing_lookback:     int   = 5

    # Options avancées
    partial_close_tp1:  bool  = True   # Clôture 50% à TP1
    move_sl_be:         bool  = True   # SL → BE après TP1
    walk_forward:       bool  = False  # Mode walk-forward
    wf_splits:          int   = 3      # Segments pour walk-forward


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""
    config:         BacktestConfig
    metrics:        MetricsResult
    trades:         list[TradeRecord]
    started_at:     datetime
    finished_at:    datetime
    duration_sec:   float
    bars_analyzed:  int
    signals_found:  int

    # Walk-forward segments
    wf_segments:    list[MetricsResult] = field(default_factory=list)

    @property
    def is_profitable(self) -> bool:
        return self.metrics.total_pnl_usd > 0

    @property
    def meets_target(self) -> bool:
        """True si le backtest atteint l'objectif ≥60% WR."""
        return (
            self.metrics.win_rate >= 0.60
            and self.metrics.profit_factor >= 1.3
        )

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Backtest {self.config.instrument} "
            f"({self.config.months} mois)\n"
            f"Grade: {m.grade} | "
            f"WR: {m.win_rate:.1%} | "
            f"PF: {m.profit_factor:.2f} | "
            f"Sharpe: {m.sharpe_ratio:.2f}\n"
            f"P&L: {'+' if m.total_pnl_usd >= 0 else ''}"
            f"{m.total_pnl_usd:.2f} USD | "
            f"DD max: {m.max_drawdown_pct:.1f}% | "
            f"Trades: {m.total_trades}\n"
            f"{'✅ OBJECTIF 60% WR ATTEINT' if self.meets_target else '❌ Objectif non atteint'}\n"
            f"Durée analyse: {self.duration_sec:.1f}s"
        )


class Backtester:
    """
    Moteur de backtesting — Simule le bot sur données historiques.
    Utilise le même pipeline d'analyse que le bot live.
    """

    WARMUP_BARS = 60   # Bougies de préchauffage avant d'analyser

    def __init__(
        self,
        data: DataManager,
        db:   DatabaseManager,
    ):
        self._data    = data
        self._db      = db
        self._metrics = BacktestMetrics()

        # Réutilise les mêmes modules que le bot live
        self._indicators = TechnicalIndicators()
        self._scoring    = ScoringEngine()
        self._tpsl       = TPSLCalculator()
        self._validator  = EntryValidator()

    # ── BACKTEST PRINCIPAL ────────────────────────────────────────

    async def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Lance un backtest complet sur un instrument.

        Args:
            config: Configuration du backtest

        Returns:
            BacktestResult avec métriques et courbe d'équité
        """
        started_at = datetime.utcnow()
        logger.info(
            f"🔬 Backtest démarré — {config.instrument} "
            f"({config.months} mois) | "
            f"Balance: {config.initial_balance} USD"
        )

        instrument_cfg = get_instrument(config.instrument)

        # ── Chargement des données historiques ────────────────────
        tf_data = await self._load_historical_data(config)
        if not tf_data:
            logger.error(f"Backtest annulé — données insuffisantes")
            return self._empty_result(config, started_at)

        df_h4  = tf_data["4h"]
        df_h1  = tf_data["1h"]
        df_m15 = tf_data["15min"]
        df_m5  = tf_data["5min"]

        total_bars = len(df_m15)
        logger.info(
            f"📊 Données chargées — "
            f"H4:{len(df_h4)} | H1:{len(df_h1)} | "
            f"M15:{len(df_m15)} | M5:{len(df_m5)}"
        )

        # ── Simulation bougie par bougie ──────────────────────────
        smc = SMCDetector(swing_lookback=config.swing_lookback)
        trades: list[TradeRecord] = []
        signals_found = 0

        # Parcourt M15 (timeframe d'entrée)
        # Commence après le warmup
        for i in range(self.WARMUP_BARS, len(df_m15) - 1):

            # Sous-ensembles de données jusqu'à la bougie courante
            h4_slice  = self._align_tf_slice(df_h4,  df_m15.index[i])
            h1_slice  = self._align_tf_slice(df_h1,  df_m15.index[i])
            m15_slice = df_m15.iloc[:i + 1]
            m5_slice  = self._align_tf_slice(df_m5,  df_m15.index[i])

            if len(h4_slice) < 30 or len(h1_slice) < 30:
                continue

            # ── Analyse SMC ───────────────────────────────────────
            try:
                smc_h4  = smc.analyze(h4_slice,  direction="auto")
                direction = smc_h4.current_trend

                if direction == "neutral":
                    continue

                smc_h1  = smc.analyze(h1_slice,  direction=direction)
                smc_m15 = smc.analyze(m15_slice, direction=direction)

                # ── Indicateurs ───────────────────────────────────
                if len(h1_slice) < 60:
                    continue
                indicators = TechnicalIndicators.analyze(h1_slice)

                # ── Score ─────────────────────────────────────────
                score = self._scoring.calculate(
                    smc_h4     = smc_h4,
                    smc_h1     = smc_h1,
                    smc_m15    = smc_m15,
                    indicators = indicators,
                    direction  = direction,
                    min_score  = config.min_score,
                )

                if not score.is_valid:
                    continue

                # ── Prix d'entrée (open de la bougie suivante) ───
                entry = float(df_m15["open"].iloc[i + 1])
                signals_found += 1

                # ── TP/SL ─────────────────────────────────────────
                tpsl = self._tpsl.calculate(
                    entry          = entry,
                    direction      = direction,
                    atr            = indicators.atr,
                    order_block    = smc_m15.order_block,
                    swing_highs    = smc_h1.swing_highs,
                    swing_lows     = smc_h1.swing_lows,
                    atr_multiplier = config.atr_sl_multiplier,
                    tp1_ratio      = config.tp1_ratio,
                    tp2_ratio      = config.tp2_ratio,
                    tp3_ratio      = config.tp3_ratio,
                )

                # ── Validation R:R ────────────────────────────────
                if tpsl.rr_tp2 < config.min_rr:
                    continue

                # ── Simulation du trade ───────────────────────────
                trade = self._simulate_trade(
                    config     = config,
                    df_m15     = df_m15,
                    start_idx  = i + 1,
                    direction  = direction,
                    entry      = entry,
                    tpsl       = tpsl,
                    score      = score.total,
                    session    = self._get_session(df_m15.index[i]),
                    signal_id  = signals_found,
                )

                if trade:
                    trades.append(trade)
                    logger.debug(
                        f"  Trade #{len(trades)} — {direction} @ {entry} "
                        f"| {'WIN' if trade.is_win else 'LOSS'} "
                        f"| P&L: {trade.pnl_usd:+.2f} USD"
                    )

            except Exception as e:
                logger.debug(f"Analyse erreur bar {i}: {e}")
                continue

        # ── Calcul des métriques ──────────────────────────────────
        metrics = self._metrics.calculate(
            trades           = trades,
            instrument       = config.instrument,
            initial_balance  = config.initial_balance,
        )
        metrics.total_bars = total_bars

        finished_at  = datetime.utcnow()
        duration_sec = (finished_at - started_at).total_seconds()

        logger.success(
            f"✅ Backtest terminé — {len(trades)} trades | "
            f"WR: {metrics.win_rate:.1%} | "
            f"PF: {metrics.profit_factor:.2f} | "
            f"P&L: {metrics.total_pnl_usd:+.2f} USD | "
            f"{duration_sec:.1f}s"
        )

        result = BacktestResult(
            config        = config,
            metrics       = metrics,
            trades        = trades,
            started_at    = started_at,
            finished_at   = finished_at,
            duration_sec  = duration_sec,
            bars_analyzed = total_bars,
            signals_found = signals_found,
        )

        # ── Walk-forward optionnel ────────────────────────────────
        if config.walk_forward and len(trades) >= 30:
            result.wf_segments = await self._walk_forward(
                config, trades, config.wf_splits
            )

        return result

    # ── SIMULATION D'UN TRADE ─────────────────────────────────────

    def _simulate_trade(
        self,
        config:    BacktestConfig,
        df_m15:    pd.DataFrame,
        start_idx: int,
        direction: str,
        entry:     float,
        tpsl:      object,
        score:     int,
        session:   str | None,
        signal_id: int,
    ) -> TradeRecord | None:
        """
        Simule l'évolution d'un trade bougie par bougie.

        Logique :
          • Vérifie SL/TP1/TP2/TP3 sur chaque HIGH et LOW
          • À TP1 : ferme 50% + déplace SL au BE (si configuré)
          • Clôture à la fin de la période si toujours ouvert
        """
        is_long    = direction == "bullish"
        sl         = tpsl.sl
        tp1        = tpsl.tp1
        tp2        = tpsl.tp2
        tp3        = tpsl.tp3
        opened_at  = df_m15.index[start_idx]

        tp1_hit     = False
        sl_at_be    = False
        result_pnl  = None
        exit_price  = None
        tp_level    = None
        is_win      = False
        closed_idx  = start_idx

        # Max 500 bougies = ~5 jours en M15 (timeout trade)
        max_bars = min(start_idx + 500, len(df_m15))

        for j in range(start_idx, max_bars):
            candle = df_m15.iloc[j]
            high   = float(candle["high"])
            low    = float(candle["low"])

            if is_long:
                # ── SL ────────────────────────────────────────────
                if low <= sl:
                    exit_price = sl
                    is_win     = tp1_hit and sl_at_be   # BE = breakeven
                    tp_level   = None if not tp1_hit else 1
                    result_pnl = 0.0 if sl_at_be else entry - sl
                    closed_idx = j
                    break

                # ── TP3 ───────────────────────────────────────────
                if high >= tp3:
                    exit_price = tp3
                    is_win     = True
                    tp_level   = 3
                    result_pnl = tp3 - entry
                    closed_idx = j
                    break

                # ── TP2 ───────────────────────────────────────────
                if high >= tp2:
                    exit_price = tp2
                    is_win     = True
                    tp_level   = 2
                    result_pnl = tp2 - entry
                    closed_idx = j
                    break

                # ── TP1 ───────────────────────────────────────────
                if not tp1_hit and high >= tp1:
                    tp1_hit = True
                    if config.move_sl_be:
                        sl       = entry   # SL → breakeven
                        sl_at_be = True

            else:   # SHORT
                # ── SL ────────────────────────────────────────────
                if high >= sl:
                    exit_price = sl
                    is_win     = tp1_hit and sl_at_be
                    tp_level   = None if not tp1_hit else 1
                    result_pnl = 0.0 if sl_at_be else sl - entry
                    result_pnl = -result_pnl if result_pnl > 0 else result_pnl
                    closed_idx = j
                    break

                # ── TP3 ───────────────────────────────────────────
                if low <= tp3:
                    exit_price = tp3
                    is_win     = True
                    tp_level   = 3
                    result_pnl = entry - tp3
                    closed_idx = j
                    break

                # ── TP2 ───────────────────────────────────────────
                if low <= tp2:
                    exit_price = tp2
                    is_win     = True
                    tp_level   = 2
                    result_pnl = entry - tp2
                    closed_idx = j
                    break

                # ── TP1 ───────────────────────────────────────────
                if not tp1_hit and low <= tp1:
                    tp1_hit = True
                    if config.move_sl_be:
                        sl       = entry
                        sl_at_be = True

        # Trade toujours ouvert à la fin → clôture au dernier prix
        if exit_price is None:
            last_close = float(df_m15["close"].iloc[max_bars - 1])
            exit_price = last_close
            result_pnl = (
                last_close - entry if is_long
                else entry - last_close
            )
            is_win     = result_pnl > 0
            tp_level   = None
            closed_idx = max_bars - 1

        # ── Calcul P&L en USD ─────────────────────────────────────
        instr_cfg  = get_instrument(config.instrument)
        pip_val    = instr_cfg.pip_value
        pnl_pips   = result_pnl / pip_val if pip_val > 0 else result_pnl

        # Taille de position fixe basée sur le risque
        risk_usd   = config.initial_balance * (config.risk_pct / 100)
        sl_dist    = abs(entry - tpsl.sl)
        if sl_dist > 0:
            if instr_cfg.broker == "oanda":
                lot_size = risk_usd / (sl_dist / pip_val * pip_val) / 100000
                lot_size = max(instr_cfg.min_lot, min(instr_cfg.max_lot, lot_size))
                pnl_usd  = pnl_pips * pip_val * lot_size
            else:
                units    = risk_usd / sl_dist
                pnl_usd  = result_pnl * units
        else:
            pnl_usd = 0.0

        # Spread cost
        pnl_usd -= instr_cfg.avg_spread_pips * pip_val * 0.01

        # ── Durée ─────────────────────────────────────────────────
        try:
            close_ts     = df_m15.index[closed_idx]
            duration_min = int((close_ts - opened_at).total_seconds() / 60)
        except Exception:
            duration_min = (closed_idx - start_idx) * 15

        # ── R:R achievé ───────────────────────────────────────────
        rr_achieved = (
            abs(exit_price - entry) / abs(tpsl.sl - entry)
            if abs(tpsl.sl - entry) > 0 else 0.0
        )
        rr_achieved = rr_achieved if is_win else -rr_achieved

        return TradeRecord(
            signal_id    = signal_id,
            instrument   = config.instrument,
            direction    = direction,
            entry        = entry,
            exit_price   = exit_price,
            sl           = tpsl.sl,
            tp1          = tp1,
            tp2          = tp2,
            tp3          = tp3,
            pnl_usd      = round(pnl_usd, 2),
            pnl_pips     = round(pnl_pips, 1),
            is_win       = is_win,
            tp_level_hit = tp_level,
            duration_min = duration_min,
            score        = score,
            session      = session,
            opened_at    = opened_at,
            closed_at    = df_m15.index[min(closed_idx, len(df_m15) - 1)],
            rr_ratio     = round(tpsl.rr_tp2, 2),
            rr_achieved  = round(abs(rr_achieved), 2),
        )

    # ── BACKTEST MULTI-INSTRUMENTS ────────────────────────────────

    async def run_all(
        self,
        months:          int = 6,
        initial_balance: float = 10000.0,
        risk_pct:        float = 1.0,
    ) -> dict[str, BacktestResult]:
        """
        Lance un backtest sur tous les instruments configurés.
        Exécution séquentielle pour éviter les limites API.
        """
        results: dict[str, BacktestResult] = {}

        for symbol in INSTRUMENTS.keys():
            config = BacktestConfig(
                instrument      = symbol,
                months          = months,
                initial_balance = initial_balance,
                risk_pct        = risk_pct,
                min_score       = settings.MIN_CONFLUENCE_SCORE,
                atr_sl_multiplier = settings.ATR_SL_MULTIPLIER,
                tp1_ratio       = settings.TP1_RATIO,
                tp2_ratio       = settings.TP2_RATIO,
                tp3_ratio       = settings.TP3_RATIO,
                min_rr          = settings.MIN_RR_RATIO,
            )

            try:
                result = await self.run(config)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Backtest {symbol} error: {e}")

            # Pause pour respecter les limites API
            await asyncio.sleep(2.0)

        # ── Résumé global ─────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("RÉSUMÉ BACKTEST GLOBAL")
        logger.info("=" * 60)
        for sym, res in results.items():
            m = res.metrics
            status = "✅" if res.meets_target else "❌"
            logger.info(
                f"{status} {sym:<10} | "
                f"WR: {m.win_rate:.1%} | "
                f"PF: {m.profit_factor:.2f} | "
                f"Grade: {m.grade} | "
                f"P&L: {m.total_pnl_usd:+.2f} USD"
            )

        return results

    # ── WALK-FORWARD ──────────────────────────────────────────────

    async def _walk_forward(
        self,
        config:  BacktestConfig,
        trades:  list[TradeRecord],
        n_splits: int,
    ) -> list[MetricsResult]:
        """
        Divise les trades en N segments et calcule les métriques
        sur chaque segment séparément.

        Valide que les performances sont stables dans le temps.
        """
        if len(trades) < n_splits * 10:
            return []

        segment_size = len(trades) // n_splits
        segments     = []

        for i in range(n_splits):
            start = i * segment_size
            end   = start + segment_size if i < n_splits - 1 else len(trades)
            seg   = trades[start:end]

            m = self._metrics.calculate(
                trades          = seg,
                instrument      = config.instrument,
                initial_balance = config.initial_balance,
            )
            segments.append(m)

            logger.debug(
                f"  WF Segment {i+1}/{n_splits} — "
                f"WR: {m.win_rate:.1%} | "
                f"PF: {m.profit_factor:.2f} | "
                f"Trades: {m.total_trades}"
            )

        return segments

    # ── CHARGEMENT DES DONNÉES ────────────────────────────────────

    async def _load_historical_data(
        self,
        config: BacktestConfig,
    ) -> dict[str, pd.DataFrame] | None:
        """Charge les données OHLCV historiques pour le backtest."""
        # Calcule le nombre de bougies nécessaires
        bars_map = {
            "4h":   config.months * 30 * 6,     # 6 bougies/jour
            "1h":   config.months * 30 * 24,    # 24 bougies/jour
            "15min": config.months * 30 * 96,   # 96 bougies/jour
            "5min":  config.months * 30 * 288,  # 288 bougies/jour
        }

        # Twelve Data limite à 5000 bougies par appel
        for tf in bars_map:
            bars_map[tf] = min(bars_map[tf], 5000)

        try:
            tf_data: dict[str, pd.DataFrame] = {}
            for tf, output_size in bars_map.items():
                df = await self._data.get_candles(
                    symbol      = config.instrument,
                    timeframe   = tf,
                    output_size = output_size,
                    force_fetch = True,
                )
                if len(df) < self.WARMUP_BARS + 10:
                    logger.error(
                        f"Données insuffisantes {config.instrument} {tf}: "
                        f"{len(df)} bougies"
                    )
                    return None
                tf_data[tf] = df

            return tf_data

        except Exception as e:
            logger.error(f"Chargement données backtest error: {e}")
            return None

    # ── UTILITAIRES ───────────────────────────────────────────────

    @staticmethod
    def _align_tf_slice(
        df:          pd.DataFrame,
        until:       pd.Timestamp,
    ) -> pd.DataFrame:
        """Retourne le slice du DataFrame jusqu'au timestamp donné."""
        if df is None or df.empty:
            return pd.DataFrame()
        mask = df.index <= until
        return df[mask]

    @staticmethod
    def _get_session(timestamp: pd.Timestamp) -> str | None:
        """Retourne la session correspondant au timestamp."""
        hour = timestamp.hour
        if 13 <= hour < 17:
            return "overlap"
        if 7 <= hour < 13:
            return "london"
        if 17 <= hour < 20:
            return "ny"
        return None

    @staticmethod
    def _empty_result(
        config:     BacktestConfig,
        started_at: datetime,
    ) -> BacktestResult:
        """Retourne un résultat vide en cas d'erreur."""
        empty_metrics = BacktestMetrics._empty_result(config.instrument)
        return BacktestResult(
            config        = config,
            metrics       = empty_metrics,
            trades        = [],
            started_at    = started_at,
            finished_at   = datetime.utcnow(),
            duration_sec  = 0.0,
            bars_analyzed = 0,
            signals_found = 0,
        )
