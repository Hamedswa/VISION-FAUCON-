# config/instruments.py
# ─────────────────────────────────────────────────────────────────
#  Configuration par instrument
#  Chaque instrument a ses propres paramètres de trading
# ─────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InstrumentConfig:
    symbol: str                          # Symbole interne du bot
    twelve_data_symbol: str              # Symbole pour Twelve Data API
    oanda_symbol: str | None             # Symbole OANDA (None si crypto)
    ccxt_symbol: str | None              # Symbole CCXT (None si Forex)
    broker: Literal["oanda", "ccxt"]     # Broker d'exécution
    pip_value: float                     # Valeur d'un pip en USD
    min_lot: float                       # Lot minimum
    max_lot: float                       # Lot maximum
    lot_step: float                      # Pas de lot
    avg_spread_pips: float               # Spread moyen en pips
    atr_period: int = 14                 # Période ATR
    sessions_24h: bool = False           # True = trade 24/7 (BTC)
    priority: int = 1                    # 1 = haute priorité


INSTRUMENTS: dict[str, InstrumentConfig] = {

    # ── OR / XAU/USD ──────────────────────────────────────────────
    "XAUUSD": InstrumentConfig(
        symbol="XAUUSD",
        twelve_data_symbol="XAU/USD",
        oanda_symbol="XAU_USD",
        ccxt_symbol=None,
        broker="oanda",
        pip_value=0.01,
        min_lot=0.01,
        max_lot=10.0,
        lot_step=0.01,
        avg_spread_pips=3.0,
        sessions_24h=False,
        priority=1,
    ),

    # ── BITCOIN / BTC/USD ─────────────────────────────────────────
    "BTCUSD": InstrumentConfig(
        symbol="BTCUSD",
        twelve_data_symbol="BTC/USD",
        oanda_symbol=None,
        ccxt_symbol="BTC/USDT",
        broker="ccxt",
        pip_value=1.0,
        min_lot=0.001,
        max_lot=1.0,
        lot_step=0.001,
        avg_spread_pips=50.0,
        sessions_24h=True,
        priority=1,
    ),

    # ── EUR/USD ───────────────────────────────────────────────────
    "EURUSD": InstrumentConfig(
        symbol="EURUSD",
        twelve_data_symbol="EUR/USD",
        oanda_symbol="EUR_USD",
        ccxt_symbol=None,
        broker="oanda",
        pip_value=10.0,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
        avg_spread_pips=1.2,
        sessions_24h=False,
        priority=2,
    ),

    # ── GBP/USD ───────────────────────────────────────────────────
    "GBPUSD": InstrumentConfig(
        symbol="GBPUSD",
        twelve_data_symbol="GBP/USD",
        oanda_symbol="GBP_USD",
        ccxt_symbol=None,
        broker="oanda",
        pip_value=10.0,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
        avg_spread_pips=1.5,
        sessions_24h=False,
        priority=2,
    ),

    # ── USD/JPY ───────────────────────────────────────────────────
    "USDJPY": InstrumentConfig(
        symbol="USDJPY",
        twelve_data_symbol="USD/JPY",
        oanda_symbol="USD_JPY",
        ccxt_symbol=None,
        broker="oanda",
        pip_value=9.0,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
        avg_spread_pips=1.0,
        sessions_24h=False,
        priority=3,
    ),
}


def get_instrument(symbol: str) -> InstrumentConfig:
    """Retourne la config d'un instrument ou lève une erreur claire."""
    if symbol not in INSTRUMENTS:
        raise ValueError(
            f"Instrument '{symbol}' inconnu. "
            f"Disponibles : {list(INSTRUMENTS.keys())}"
        )
    return INSTRUMENTS[symbol]


def get_active_instruments() -> list[InstrumentConfig]:
    """Retourne tous les instruments triés par priorité."""
    return sorted(INSTRUMENTS.values(), key=lambda x: x.priority)
