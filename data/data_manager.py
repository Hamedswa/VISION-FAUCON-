# data/data_manager.py
# ─────────────────────────────────────────────────────────────────
#  DataManager — Orchestrateur central des données
#  Adapté pour MT5 (Exness) + Binance
#
#  Responsabilités :
#   • Fetch OHLCV via Twelve Data (tous les instruments)
#   • Cache mémoire pour limiter les appels API
#   • Prix en temps réel (MT5 pour Forex/XAU, CCXT pour BTC)
#   • Routing exécution → MT5 ou CCXT selon l'instrument
# ─────────────────────────────────────────────────────────────────

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from twelve_data import TDClient
from loguru import logger

from config import settings
from config.instruments import get_instrument, INSTRUMENTS
from .mt5_client import MT5Client
from .ccxt_client import CCXTClient


TF_MAP = {
    "4h":    "4h",
    "1h":    "1h",
    "15min": "15min",
    "5min":  "5min",
    "1min":  "1min",
    "1day":  "1day",
}


class DataManager:
    """
    Point d'entrée unique pour toutes les données marché.
    MT5 → Exness (Forex + XAU)
    CCXT → Binance (BTC)
    Twelve Data → Données OHLCV pour tous
    """

    def __init__(self):
        self._td    = TDClient(apikey=settings.TWELVE_DATA_KEY)
        self._mt5   = MT5Client()
        self._ccxt  = CCXTClient()

        # Cache mémoire : {(symbol, timeframe): (timestamp, DataFrame)}
        self._cache: dict[tuple, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = {
            "5min":  timedelta(minutes=4),
            "15min": timedelta(minutes=13),
            "1h":    timedelta(minutes=55),
            "4h":    timedelta(hours=3, minutes=50),
            "1day":  timedelta(hours=23),
        }

    # ── INITIALISATION ────────────────────────────────────────────

    async def initialize(self):
        """Connexion MT5 au démarrage."""
        connected = await self._mt5.connect()
        if not connected:
            logger.warning(
                "⚠️ MT5 non connecté — "
                "Vérifie que MetaTrader 5 est ouvert et connecté"
            )
        return connected

    async def close(self):
        """Fermeture propre des connexions."""
        await self._mt5.disconnect()
        await self._ccxt.close()
        logger.info("DataManager fermé")

    # ── OHLCV — FETCH PRINCIPAL ───────────────────────────────────

    async def get_candles(
        self,
        symbol:      str,
        timeframe:   str,
        output_size: int = 200,
        force_fetch: bool = False,
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame OHLCV via Twelve Data.
        Cache mémoire avec TTL par timeframe.
        """
        cache_key = (symbol, timeframe)
        ttl       = self._cache_ttl.get(timeframe, timedelta(minutes=15))

        if not force_fetch and cache_key in self._cache:
            ts, df = self._cache[cache_key]
            if datetime.utcnow() - ts < ttl:
                return df.copy()

        instrument  = get_instrument(symbol)
        td_symbol   = instrument.twelve_data_symbol
        td_tf       = TF_MAP.get(timeframe, timeframe)

        df = await self._fetch_twelve_data(td_symbol, td_tf, output_size)
        self._cache[cache_key] = (datetime.utcnow(), df)
        return df.copy()

    async def _fetch_twelve_data(
        self,
        symbol:      str,
        interval:    str,
        output_size: int,
    ) -> pd.DataFrame:
        """Appel API Twelve Data dans un thread pool."""
        loop = asyncio.get_event_loop()

        def _fetch():
            ts = self._td.time_series(
                symbol     = symbol,
                interval   = interval,
                outputsize = output_size,
                order      = "ASC",
                timezone   = "UTC",
            )
            return ts.as_pandas()

        try:
            df = await loop.run_in_executor(None, _fetch)
            df.columns = [c.lower() for c in df.columns]
            df.index   = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            if "volume" not in df.columns:
                df["volume"] = 0.0

            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df.sort_index()

            logger.debug(
                f"📡 Twelve Data — {symbol} {interval} "
                f"→ {len(df)} bougies"
            )
            return df

        except Exception as e:
            logger.error(f"Twelve Data fetch error ({symbol} {interval}): {e}")
            raise

    # ── MULTI-TIMEFRAME ───────────────────────────────────────────

    async def get_multi_tf(
        self,
        symbol:      str,
        timeframes:  list[str] | None = None,
        output_size: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """Fetch H4, H1, M15, M5 en parallèle."""
        tfs = timeframes or settings.TIMEFRAMES

        tasks = {
            tf: asyncio.create_task(
                self.get_candles(symbol, tf, output_size)
            )
            for tf in tfs
        }

        results: dict[str, pd.DataFrame] = {}
        for tf, task in tasks.items():
            try:
                results[tf] = await task
            except Exception as e:
                logger.error(f"get_multi_tf error — {symbol} {tf}: {e}")
                results[tf] = pd.DataFrame()

        return results

    # ── PRIX TEMPS RÉEL ───────────────────────────────────────────

    async def get_current_price(self, symbol: str) -> float:
        """
        Prix mid actuel.
        → MT5 pour Forex/XAU (Exness)
        → CCXT pour BTC (Binance)
        """
        instrument = get_instrument(symbol)

        if instrument.broker == "mt5" and instrument.mt5_symbol:
            try:
                data = await self._mt5.get_price(instrument.mt5_symbol)
                return data["mid"]
            except Exception as e:
                logger.warning(f"MT5 price error {symbol}: {e}")

        elif instrument.broker == "ccxt" and instrument.ccxt_symbol:
            try:
                data = await self._ccxt.get_price(instrument.ccxt_symbol)
                return data["mid"]
            except Exception as e:
                logger.warning(f"CCXT price error {symbol}: {e}")

        # Fallback : dernière clôture Twelve Data
        df = await self.get_candles(symbol, "5min", output_size=5, force_fetch=True)
        return float(df["close"].iloc[-1])

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Prix actuels de plusieurs instruments en parallèle."""
        tasks = {
            s: asyncio.create_task(self.get_current_price(s))
            for s in symbols
        }
        results: dict[str, float] = {}
        for symbol, task in tasks.items():
            try:
                results[symbol] = await task
            except Exception as e:
                logger.error(f"get_current_prices error — {symbol}: {e}")
                results[symbol] = 0.0
        return results

    # ── SOLDE & COMPTE ────────────────────────────────────────────

    async def get_balance(self, broker: str = "mt5") -> float:
        """Retourne le solde disponible."""
        if broker == "mt5":
            return await self._mt5.get_balance()
        elif broker == "ccxt":
            return await self._ccxt.get_balance()
        return 0.0

    # ── ROUTING EXÉCUTION ────────────────────────────────────────

    def get_mt5(self) -> MT5Client:
        """Accès direct au client MT5."""
        return self._mt5

    def get_ccxt(self) -> CCXTClient:
        """Accès direct au client CCXT (Binance)."""
        return self._ccxt

    # ── VALIDATION CONNEXIONS ─────────────────────────────────────

    async def validate_connection(self) -> dict[str, bool]:
        """Vérifie toutes les connexions API au démarrage."""
        status = {
            "twelve_data": False,
            "mt5":         False,
            "ccxt":        False,
        }

        # Twelve Data
        try:
            await self.get_candles("XAUUSD", "1h", output_size=5)
            status["twelve_data"] = True
            logger.info("✅ Twelve Data — connexion OK")
        except Exception as e:
            logger.error(f"❌ Twelve Data — FAILED: {e}")

        # MT5
        try:
            status["mt5"] = await self._mt5.is_connected()
            if status["mt5"]:
                logger.info("✅ MT5 Exness — connexion OK")
            else:
                logger.error("❌ MT5 Exness — non connecté")
        except Exception as e:
            logger.error(f"❌ MT5 Exness — FAILED: {e}")

        # CCXT Binance
        try:
            await self._ccxt.get_balance()
            status["ccxt"] = True
            logger.info("✅ Binance — connexion OK")
        except Exception as e:
            logger.warning(f"⚠️ Binance — FAILED (non bloquant): {e}")

        return status

    # ── UTILITAIRES ───────────────────────────────────────────────

    def clear_cache(self, symbol: str | None = None):
        if symbol:
            keys = [k for k in self._cache if k[0] == symbol]
            for k in keys:
                del self._cache[k]
        else:
            self._cache.clear()
