# database/models.py
# ─────────────────────────────────────────────────────────────────
#  Modèles SQLAlchemy — Tables de la base de données
#  Toutes les données du bot sont stockées ici pour backtesting
#  et analyse de performance
# ─────────────────────────────────────────────────────────────────

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, JSON, Enum, ForeignKey, Index, Text
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


# ── ENUMS ─────────────────────────────────────────────────────────

class SignalDirection(str, enum.Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


class SignalStatus(str, enum.Enum):
    PENDING   = "PENDING"    # Signal envoyé, trade non ouvert
    ACTIVE    = "ACTIVE"     # Trade ouvert
    TP1_HIT   = "TP1_HIT"    # TP1 atteint
    TP2_HIT   = "TP2_HIT"    # TP2 atteint
    TP3_HIT   = "TP3_HIT"    # TP3 atteint (full win)
    SL_HIT    = "SL_HIT"     # Stop loss touché (loss)
    CANCELLED = "CANCELLED"  # Annulé avant entrée
    EXPIRED   = "EXPIRED"    # Expiré sans exécution


class TradeResult(str, enum.Enum):
    WIN       = "WIN"
    LOSS      = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    PARTIAL   = "PARTIAL"    # TP1 ou TP2 atteint, pas TP3


# ── TABLE : SIGNALS ───────────────────────────────────────────────

class Signal(Base):
    """
    Chaque signal généré par le bot — validé ou non.
    Source principale pour le backtesting et l'optimisation.
    """
    __tablename__ = "signals"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── Identification ────────────────────────────────────────────
    instrument      = Column(String(20), nullable=False, index=True)
    direction       = Column(Enum(SignalDirection), nullable=False)
    status          = Column(Enum(SignalStatus), default=SignalStatus.PENDING, index=True)
    timeframe       = Column(String(10), nullable=False)   # ex: "15min"

    # ── Prix ──────────────────────────────────────────────────────
    entry_price     = Column(Float, nullable=False)
    sl_price        = Column(Float, nullable=False)
    tp1_price       = Column(Float, nullable=False)
    tp2_price       = Column(Float, nullable=False)
    tp3_price       = Column(Float, nullable=False)
    actual_entry    = Column(Float, nullable=True)         # Prix d'entrée réel

    # ── Risk/Reward ───────────────────────────────────────────────
    rr_ratio        = Column(Float, nullable=False)
    risk_pips       = Column(Float, nullable=False)
    reward_pips_tp1 = Column(Float, nullable=False)
    reward_pips_tp2 = Column(Float, nullable=False)
    reward_pips_tp3 = Column(Float, nullable=False)

    # ── Score de confluence ───────────────────────────────────────
    confluence_score      = Column(Integer, nullable=False)   # /100
    score_bos             = Column(Integer, default=0)        # 0 ou 20
    score_choch           = Column(Integer, default=0)        # 0 ou 15
    score_ob              = Column(Integer, default=0)        # 0 ou 20
    score_fvg             = Column(Integer, default=0)        # 0 ou 10
    score_liquidity       = Column(Integer, default=0)        # 0 ou 10
    score_mtf_alignment   = Column(Integer, default=0)        # 5, 10 ou 15
    score_rsi             = Column(Integer, default=0)        # 0 ou 5
    score_macd            = Column(Integer, default=0)        # 0 ou 5

    # ── SMC/ICT Détails ───────────────────────────────────────────
    smc_details     = Column(JSON, nullable=True)
    # Exemple: {"ob_level": 1920.5, "fvg_range": [1918, 1922], "swept_ssl": 1915.0}

    # ── Indicateurs au moment du signal ──────────────────────────
    rsi_value       = Column(Float, nullable=True)
    macd_value      = Column(Float, nullable=True)
    macd_signal     = Column(Float, nullable=True)
    atr_value       = Column(Float, nullable=True)
    ema20           = Column(Float, nullable=True)
    ema50           = Column(Float, nullable=True)

    # ── Session & Contexte ────────────────────────────────────────
    session         = Column(String(20), nullable=True)    # "london" | "ny" | "overlap"
    news_nearby     = Column(Boolean, default=False)       # News < 30min

    # ── Résultat ──────────────────────────────────────────────────
    pnl_pips        = Column(Float, nullable=True)         # P&L en pips
    pnl_usd         = Column(Float, nullable=True)         # P&L en USD
    exit_price      = Column(Float, nullable=True)
    exit_at         = Column(DateTime, nullable=True)
    duration_minutes= Column(Integer, nullable=True)       # Durée du trade

    # ── Telegram ─────────────────────────────────────────────────
    telegram_msg_id = Column(Integer, nullable=True)       # ID du message Telegram

    # ── Relation vers Trade ───────────────────────────────────────
    trade           = relationship("Trade", back_populates="signal", uselist=False)

    __table_args__ = (
        Index("ix_signals_instrument_created", "instrument", "created_at"),
        Index("ix_signals_status_instrument", "status", "instrument"),
    )

    def __repr__(self):
        return (
            f"<Signal {self.id} | {self.instrument} {self.direction} "
            f"@ {self.entry_price} | Score: {self.confluence_score} "
            f"| Status: {self.status}>"
        )


# ── TABLE : TRADES ────────────────────────────────────────────────

class Trade(Base):
    """
    Trade réellement exécuté sur OANDA ou CCXT.
    Lié à un Signal parent.
    """
    __tablename__ = "trades"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    signal_id       = Column(Integer, ForeignKey("signals.id"), nullable=False, unique=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── Broker ────────────────────────────────────────────────────
    broker          = Column(String(20), nullable=False)   # "oanda" | "ccxt"
    broker_order_id = Column(String(100), nullable=True)   # ID ordre chez le broker
    broker_trade_id = Column(String(100), nullable=True)   # ID trade chez le broker

    # ── Exécution ─────────────────────────────────────────────────
    lot_size        = Column(Float, nullable=False)
    open_price      = Column(Float, nullable=False)
    close_price     = Column(Float, nullable=True)
    open_at         = Column(DateTime, nullable=False)
    close_at        = Column(DateTime, nullable=True)

    # ── Résultat ──────────────────────────────────────────────────
    result          = Column(Enum(TradeResult), nullable=True)
    pnl_usd         = Column(Float, nullable=True)
    pnl_pips        = Column(Float, nullable=True)
    commission      = Column(Float, default=0.0)
    swap            = Column(Float, default=0.0)
    net_pnl_usd     = Column(Float, nullable=True)         # Après commission + swap

    # ── TP atteint ────────────────────────────────────────────────
    tp_level_hit    = Column(Integer, nullable=True)       # 1, 2 ou 3

    # ── Capital ───────────────────────────────────────────────────
    balance_before  = Column(Float, nullable=True)
    balance_after   = Column(Float, nullable=True)

    # ── Relation ──────────────────────────────────────────────────
    signal          = relationship("Signal", back_populates="trade")

    __table_args__ = (
        Index("ix_trades_broker_order", "broker", "broker_order_id"),
    )

    def __repr__(self):
        return (
            f"<Trade {self.id} | Signal {self.signal_id} "
            f"| {self.result} | P&L: {self.net_pnl_usd} USD>"
        )


# ── TABLE : CANDLES ───────────────────────────────────────────────

class Candle(Base):
    """
    Cache OHLCV pour éviter les appels API répétés.
    Utilisé par le backtesting engine.
    """
    __tablename__ = "candles"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    instrument  = Column(String(20), nullable=False)
    timeframe   = Column(String(10), nullable=False)
    timestamp   = Column(DateTime, nullable=False)
    open        = Column(Float, nullable=False)
    high        = Column(Float, nullable=False)
    low         = Column(Float, nullable=False)
    close       = Column(Float, nullable=False)
    volume      = Column(Float, default=0.0)

    __table_args__ = (
        Index(
            "ix_candles_unique",
            "instrument", "timeframe", "timestamp",
            unique=True
        ),
        Index("ix_candles_instrument_tf", "instrument", "timeframe"),
    )

    def __repr__(self):
        return (
            f"<Candle {self.instrument} {self.timeframe} "
            f"| {self.timestamp} O:{self.open} H:{self.high} "
            f"L:{self.low} C:{self.close}>"
        )


# ── TABLE : OPTIMIZER RUNS ────────────────────────────────────────

class OptimizerRun(Base):
    """
    Historique des optimisations Optuna.
    Permet de tracker l'évolution des paramètres dans le temps.
    """
    __tablename__ = "optimizer_runs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_at          = Column(DateTime, default=datetime.utcnow, nullable=False)
    instrument      = Column(String(20), nullable=True)    # None = global
    trials_count    = Column(Integer, nullable=False)
    best_score      = Column(Float, nullable=False)        # Meilleur WR obtenu

    # Paramètres optimisés (stockés en JSON pour flexibilité)
    best_params     = Column(JSON, nullable=False)
    # Exemple: {"min_confluence_score": 78, "atr_sl_multiplier": 1.3, ...}

    # Métriques sur la période de test
    win_rate        = Column(Float, nullable=False)
    profit_factor   = Column(Float, nullable=False)
    sharpe_ratio    = Column(Float, nullable=True)
    total_trades    = Column(Integer, nullable=False)
    period_months   = Column(Integer, nullable=False)

    notes           = Column(Text, nullable=True)

    def __repr__(self):
        return (
            f"<OptimizerRun {self.id} | {self.run_at} "
            f"| WR: {self.win_rate:.1%} | PF: {self.profit_factor:.2f}>"
        )


# ── TABLE : PERFORMANCE SNAPSHOTS ────────────────────────────────

class PerformanceSnapshot(Base):
    """
    Snapshot journalier/hebdomadaire des performances.
    Alimente le dashboard et le suivi de la couche adaptative.
    """
    __tablename__ = "performance_snapshots"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_at     = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    period          = Column(String(10), nullable=False)   # "daily" | "weekly" | "monthly"
    instrument      = Column(String(20), nullable=True)    # None = global (tous instruments)

    # Métriques de performance
    total_signals   = Column(Integer, default=0)
    total_trades    = Column(Integer, default=0)
    winning_trades  = Column(Integer, default=0)
    losing_trades   = Column(Integer, default=0)
    win_rate        = Column(Float, nullable=True)         # Ex: 0.65 = 65%
    profit_factor   = Column(Float, nullable=True)
    sharpe_ratio    = Column(Float, nullable=True)

    # P&L
    total_pnl_usd   = Column(Float, default=0.0)
    avg_win_usd     = Column(Float, nullable=True)
    avg_loss_usd    = Column(Float, nullable=True)
    max_drawdown    = Column(Float, nullable=True)         # % max drawdown

    # Capital
    balance         = Column(Float, nullable=True)
    equity          = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_perf_period_instrument", "period", "instrument"),
    )

    def __repr__(self):
        return (
            f"<PerfSnapshot {self.period} {self.snapshot_at} "
            f"| WR: {self.win_rate:.1%} | P&L: {self.total_pnl_usd} USD>"
        )
