# data/data_manager.py
# ─────────────────────────────────────────────────────────────────
#  DataManager — Orchestrateur central des données
#
#  Responsabilités :
#   • Fetch OHLCV via Twelve Data (tous les instruments)
#   • Cache en mémoire + base de données pour limiter les appels API
#   • Conversion vers DataFrame pandas standardisé
#   • Prix en temps réel (route vers OANDA ou CCXT selon instrument)
#   • Fetch multi-timeframe en une seule opération
# ─────────────────────────────────────────────────────────────────

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from twelve_data import TDClient
from loguru import logger

from config import settings
from config.instruments import INSTRUMENTS, InstrumentConfig, get_instrument
from .oanda_client import OandaClient
from .ccxt_client import CCXTClient


# ── Mapping timeframes → format Twelve Data ────────────────────────
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
    Instancier une fois au démarrage et réutiliser partout.
    """

    def __init__(self):
        self._td      = TDClient(apikey=settings.TWELVE_DATA_KEY)
        self._oanda   = OandaClient()
        self._ccxt    = CCXTClient()

        # Cache mémoire : {(symbol, timeframe): (timestamp, DataFrame)}
        self._cache: dict[tuple, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = {
            "5min":  timedelta(minutes=4),
            "15min": timedelta(minutes=13),
            "1h":    timedelta(minutes=55),
            "4h":    timedelta(hours=3, minutes=50),
            "1day":  timedelta(hours=23),
        }

    # ── CLEANUP ───────────────────────────────────────────────────

    async def close(self):
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
        Retourne un DataFrame OHLCV pour un instrument et timeframe.

        Stratégie de cache :
          1. Cache mémoire (TTL selon timeframe) → ultra-rapide
          2. Twelve Data API                     → fallback

        Args:
            symbol:      Ex "XAUUSD", "BTCUSD", "EURUSD"
            timeframe:   Ex "4h", "1h", "15min", "5min"
            output_size: Nombre de bougies (max 5000)
            force_fetch: Ignore le cache et force un appel API

        Returns:
            DataFrame avec index DatetimeIndex et colonnes :
            open, high, low, close + volume (0 si non dispo)
        """
        cache_key = (symbol, timeframe)
        ttl = self._cache_ttl.get(timeframe, timedelta(minutes=15))

        # ── Cache mémoire ─────────────────────────────────────────
        if not force_fetch and cache_key in self._cache:
            ts, df = self._cache[cache_key]
            if datetime.utcnow() - ts < ttl:
                logger.debug(f"📦 Cache hit — {symbol} {timeframe}")
                return df.copy()

        # ── Twelve Data API ───────────────────────────────────────
        instrument = get_instrument(symbol)
        td_symbol  = instrument.twelve_data_symbol
        td_tf      = TF_MAP.get(timeframe, timeframe)

        df = await self._fetch_twelve_data(td_symbol, td_tf, output_size)

        # Mise en cache
        self._cache[cache_key] = (datetime.utcnow(), df)
        return df.copy()

    async def _fetch_twelve_data(
        self,
        symbol:      str,
        interval:    str,
        output_size: int,
    ) -> pd.DataFrame:
        """
        Appel API Twelve Data — exécuté dans un thread pool
        (bibliothèque synchrone).
        """
        loop = asyncio.get_event_loop()

        def _fetch():
            ts = self._td.time_series(
                symbol      = symbol,
                interval    = interval,
                outputsize  = output_size,
                order       = "ASC",
                timezone    = "UTC",
            )
            return ts.as_pandas()

        try:
            df = await loop.run_in_executor(None, _fetch)

            # Standardisation des colonnes
            df.columns = [c.lower() for c in df.columns]
            df.index   = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"

            # Colonnes obligatoires
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

    # ── MULTI-TIMEFRAME EN UNE OPÉRATION ──────────────────────────

    async def get_multi_tf(
        self,
        symbol:      str,
        timeframes:  list[str] | None = None,
        output_size: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch H4, H1, M15, M5 en parallèle pour un instrument.

        Returns:
            {"4h": df_h4, "1h": df_h1, "15min": df_m15, "5min": df_m5}
        """
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

        logger.info(
            f"📊 Multi-TF {symbol} — "
            + " | ".join(f"{tf}: {len(df)} bougies"
                         for tf, df in results.items())
        )
        return results

    # ── PRIX TEMPS RÉEL ───────────────────────────────────────────

    async def get_current_price(self, symbol: str) -> float:
        """
        Retourne le prix mid actuel d'un instrument.
        Route vers OANDA ou CCXT selon le broker de l'instrument.
        """
        instrument = get_instrument(symbol)

        if instrument.broker == "oanda" and instrument.oanda_symbol:
            data  = await self._oanda.get_price(instrument.oanda_symbol)
            return data["mid"]
        elif instrument.broker == "ccxt" and instrument.ccxt_symbol:
            data  = await self._ccxt.get_price(instrument.ccxt_symbol)
            return data["mid"]
        else:
            # Fallback : dernière clôture Twelve Data
            df = await self.get_candles(symbol, "5min", output_size=5,
                                        force_fetch=True)
            return float(df["close"].iloc[-1])

    async def get_current_prices(
        self, symbols: list[str]
    ) -> dict[str, float]:
        """Retourne les prix actuels de plusieurs instruments en parallèle."""
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

    async def get_balance(self, broker: str = "oanda") -> float:
        """Retourne le solde disponible sur un broker."""
        if broker == "oanda":
            return await self._oanda.get_balance()
        elif broker == "ccxt":
            return await self._ccxt.get_balance()
        return 0.0

    # ── ROUTING EXÉCUTION ────────────────────────────────────────

    def get_oanda(self) -> OandaClient:
        """Accès direct au client OANDA (pour order_manager)."""
        return self._oanda

    def get_ccxt(self) -> CCXTClient:
        """Accès direct au client CCXT (pour order_manager)."""
        return self._ccxt

    # ── UTILITAIRES ───────────────────────────────────────────────

    def clear_cache(self, symbol: str | None = None):
        """Vide le cache mémoire (global ou par instrument)."""
        if symbol:
            keys = [k for k in self._cache if k[0] == symbol]
            for k in keys:
                del self._cache[k]
            logger.debug(f"🗑️ Cache vidé pour {symbol}")
        else:
            self._cache.clear()
            logger.debug("🗑️ Cache complet vidé")

    async def validate_connection(self) -> dict[str, bool]:
        """
        Vérifie que toutes les connexions API fonctionnent.
        Utile au démarrage du bot.
        """
        status = {
            "twelve_data": False,
            "oanda":       False,
            "ccxt":        False,
        }

        # Twelve Data
        try:
            await self.get_candles("XAUUSD", "1h", output_size=5)
            status["twelve_data"] = True
            logger.info("✅ Twelve Data — connexion OK")
        except Exception as e:
            logger.error(f"❌ Twelve Data — connexion FAILED: {e}")

        # OANDA
        try:
            await self._oanda.get_balance()
            status["oanda"] = True
            logger.info("✅ OANDA — connexion OK")
        except Exception as e:
            logger.error(f"❌ OANDA — connexion FAILED: {e}")

        # CCXT
        try:
            await self._ccxt.get_balance()
            status["ccxt"] = True
            logger.info("✅ CCXT — connexion OK")
        except Exception as e:
            logger.warning(f"⚠️ CCXT — connexion FAILED (non bloquant): {e}")

        return status
