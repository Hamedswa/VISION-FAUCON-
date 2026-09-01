# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):

    # ── ENVIRONNEMENT ─────────────────────────────────────────────
    BOT_ENV:   Literal["development", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── MT5 / EXNESS ──────────────────────────────────────────────
    MT5_LOGIN:    int
    MT5_PASSWORD: str
    MT5_SERVER:   str

    # ── BINANCE (BTC) ─────────────────────────────────────────────
    CCXT_EXCHANGE:  str  = "binance"
    CCXT_API_KEY:   str  = ""
    CCXT_API_SECRET:str  = ""
    CCXT_SANDBOX:   bool = False

    # ── TWELVE DATA ───────────────────────────────────────────────
    TWELVE_DATA_KEY: str

    # ── NEWS API ──────────────────────────────────────────────────
    NEWS_API_KEY: str

    # ── GROQ ──────────────────────────────────────────────────────
    GROQ_API_KEY: str
    GROQ_MODEL:   str = "mixtral-8x7b-32768"

    # ── TELEGRAM ──────────────────────────────────────────────────
    TELEGRAM_TOKEN:     str
    TELEGRAM_CHANNEL_ID:str
    TELEGRAM_ADMIN_ID:  str = ""

    # ── BASE DE DONNÉES ───────────────────────────────────────────
    DATABASE_URL: str

    # ── GESTION DU RISQUE ─────────────────────────────────────────
    RISK_PER_TRADE:  float = Field(default=1.0,  ge=0.1, le=5.0)
    MAX_DAILY_LOSS:  float = Field(default=3.0,  ge=0.5, le=10.0)
    MAX_OPEN_TRADES: int   = Field(default=5,    ge=1,   le=20)

    # ── SCORING ───────────────────────────────────────────────────
    MIN_CONFLUENCE_SCORE: int = Field(default=75, ge=50, le=100)

    # ── TIMEFRAMES ────────────────────────────────────────────────
    TIMEFRAMES:  list[str] = ["4h", "1h", "15min", "5min"]
    PRIMARY_TF:  str = "4h"
    CONFIRM_TF:  str = "1h"
    ENTRY_TF:    str = "15min"
    TRIGGER_TF:  str = "5min"

    # ── DASHBOARD ─────────────────────────────────────────────────
    DASHBOARD_HOST:       str = "0.0.0.0"
    DASHBOARD_PORT:       int = 8000
    DASHBOARD_SECRET_KEY: str = "changeme"

    # ── BACKTESTING ───────────────────────────────────────────────
    BACKTEST_MONTHS: int = Field(default=6, ge=1, le=24)

    # ── OPTIMISATION ADAPTATIVE ───────────────────────────────────
    OPTIMIZER_ENABLED:      bool = True
    OPTIMIZER_TRIALS:       int  = Field(default=100, ge=10, le=500)
    OPTIMIZER_INTERVAL_DAYS:int  = Field(default=7,   ge=1,  le=30)

    # ── FILTRES SESSIONS ──────────────────────────────────────────
    LONDON_OPEN_UTC:  int = 7
    LONDON_CLOSE_UTC: int = 17
    NY_OPEN_UTC:      int = 13
    NY_CLOSE_UTC:     int = 22

    # ── ANTI-SPAM ─────────────────────────────────────────────────
    MAX_SIGNALS_PER_INSTRUMENT_PER_DAY: int = 3

    # ── TP/SL PARAMÈTRES ──────────────────────────────────────────
    ATR_SL_MULTIPLIER: float = 1.5
    TP1_RATIO:         float = 1.0
    TP2_RATIO:         float = 1.618
    TP3_RATIO:         float = 2.5
    MIN_RR_RATIO:      float = 1.5

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = True


settings = Settings()
