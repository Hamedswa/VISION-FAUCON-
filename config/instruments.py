# config/instruments.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InstrumentConfig:
    symbol:          str
    twelve_data_symbol: str
    mt5_symbol:      str | None        # Symbole MT5 Exness (None si crypto)
    ccxt_symbol:     str | None        # Symbole Binance (None si Forex)
    broker:          Literal["mt5", "ccxt"]
    pip_value:       float
    min_lot:         float
    max_lot:         float
    lot_step:        float
    avg_spread_pips: float
    atr_period:      int  = 14
    sessions_24h:    bool = False
    priority:        int  = 1


INSTRUMENTS: dict[str, InstrumentConfig] = {

    # ── OR / XAU/USD ──────────────────────────────────────────────
    "XAUUSD": InstrumentConfig(
        symbol             = "XAUUSD",
        twelve_data_symbol = "XAU/USD",
        mt5_symbol         = "XAUUSD",
        ccxt_symbol        = None,
        broker             = "mt5",
        pip_value          = 0.01,
        min_lot            = 0.01,
        max_lot            = 50.0,
        lot_step           = 0.01,
        avg_spread_pips    = 3.0,
        sessions_24h       = False,
        priority           = 1,
    ),

    # ── BITCOIN / BTC/USD ─────────────────────────────────────────
    "BTCUSD": InstrumentConfig(
        symbol             = "BTCUSD",
        twelve_data_symbol = "BTC/USD",
        mt5_symbol         = None,
        ccxt_symbol        = "BTC/USDT",
        broker             = "ccxt",
        pip_value          = 1.0,
        min_lot            = 0.001,
        max_lot            = 1.0,
        lot_step           = 0.001,
        avg_spread_pips    = 50.0,
        sessions_24h       = True,
        priority           = 1,
    ),

    # ── EUR/USD ───────────────────────────────────────────────────
    "EURUSD": InstrumentConfig(
        symbol             = "EURUSD",
        twelve_data_symbol = "EUR/USD",
        mt5_symbol         = "EURUSDm",   # Exness utilise "m" pour micro
        ccxt_symbol        = None,
        broker             = "mt5",
        pip_value          = 0.0001,
        min_lot            = 0.01,
        max_lot            = 200.0,
        lot_step           = 0.01,
        avg_spread_pips    = 1.2,
        sessions_24h       = False,
        priority           = 2,
    ),

    # ── GBP/USD ───────────────────────────────────────────────────
    "GBPUSD": InstrumentConfig(
        symbol             = "GBPUSD",
        twelve_data_symbol = "GBP/USD",
        mt5_symbol         = "GBPUSDm",
        ccxt_symbol        = None,
        broker             = "mt5",
        pip_value          = 0.0001,
        min_lot            = 0.01,
        max_lot            = 200.0,
        lot_step           = 0.01,
        avg_spread_pips    = 1.5,
        sessions_24h       = False,
        priority           = 2,
    ),

    # ── USD/JPY ───────────────────────────────────────────────────
    "USDJPY": InstrumentConfig(
        symbol             = "USDJPY",
        twelve_data_symbol = "USD/JPY",
        mt5_symbol         = "USDJPYm",
        ccxt_symbol        = None,
        broker             = "mt5",
        pip_value          = 0.01,
        min_lot            = 0.01,
        max_lot            = 200.0,
        lot_step           = 0.01,
        avg_spread_pips    = 1.0,
        sessions_24h       = False,
        priority           = 3,
    ),
}


def get_instrument(symbol: str) -> InstrumentConfig:
    if symbol not in INSTRUMENTS:
        raise ValueError(
            f"Instrument '{symbol}' inconnu. "
            f"Disponibles : {list(INSTRUMENTS.keys())}"
        )
    return INSTRUMENTS[symbol]


def get_active_instruments() -> list[InstrumentConfig]:
    return sorted(INSTRUMENTS.values(), key=lambda x: x.priority)
