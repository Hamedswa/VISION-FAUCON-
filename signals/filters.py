# signals/filters.py
# ─────────────────────────────────────────────────────────────────
#  Filtres pré-signal — Gardiens de la qualité
#
#  Filtres appliqués avant chaque analyse :
#   1. Session     — London, NY, Overlap uniquement
#   2. News        — Bloque 30 min avant/après news HIGH IMPACT
#   3. Anti-spam   — Max N signaux par instrument par jour
#   4. Weekend     — Pas de trade Forex le weekend
#   5. Spread      — Spread trop large = marché illiquide
# ─────────────────────────────────────────────────────────────────

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from newsapi import NewsApiClient
from loguru import logger

from config import settings
from config.instruments import InstrumentConfig


@dataclass
class FilterResult:
    passed:      bool
    session:     str | None       # "london" | "ny" | "overlap" | "asia" | None
    blocked_by:  str | None       # Raison du blocage
    news_nearby: bool             # News HIGH IMPACT dans les 30 min
    news_title:  str | None       # Titre de la news si proche


class SignalFilters:
    """
    Applique tous les filtres pré-signal.
    Un seul filtre raté = signal bloqué.
    """

    # ── Sessions haute probabilité (heures UTC) ────────────────────
    SESSIONS = {
        "london":  {"open": 7,  "close": 12},
        "ny":      {"open": 13, "close": 17},
        "overlap": {"open": 13, "close": 17},   # London/NY overlap
    }

    # Mots-clés pour détecter les news à fort impact
    HIGH_IMPACT_KEYWORDS = [
        "fed", "federal reserve", "fomc", "interest rate", "rate decision",
        "inflation", "cpi", "pce", "gdp", "nfp", "non-farm payroll",
        "unemployment", "ecb", "bank of england", "boe", "boj",
        "gold", "xau", "bitcoin", "btc", "crypto", "geopolit",
        "war", "conflict", "sanction", "recession", "crisis",
    ]

    def __init__(self):
        self._news_client    = NewsApiClient(api_key=settings.NEWS_API_KEY)
        self._news_cache:    list[dict] = []
        self._news_cache_ts: datetime | None = None
        self._news_cache_ttl = timedelta(minutes=30)

    # ── FILTRE PRINCIPAL ──────────────────────────────────────────

    async def check(
        self,
        instrument:    str,
        instrument_cfg: InstrumentConfig,
        current_spread: float | None = None,
        signals_today:  int = 0,
    ) -> FilterResult:
        """
        Exécute tous les filtres dans l'ordre.
        S'arrête au premier filtre raté.

        Args:
            instrument:      Symbole (ex: "XAUUSD")
            instrument_cfg:  Config de l'instrument
            current_spread:  Spread actuel en pips (optionnel)
            signals_today:   Nombre de signaux déjà émis aujourd'hui
        """
        now = datetime.utcnow()

        # ── 1. Filtre weekend ─────────────────────────────────────
        if not instrument_cfg.sessions_24h:   # Forex/XAU uniquement
            if now.weekday() >= 5:             # 5=Samedi, 6=Dimanche
                return FilterResult(
                    passed=False, session=None,
                    blocked_by="Weekend — marché Forex fermé",
                    news_nearby=False, news_title=None,
                )

        # ── 2. Filtre session ─────────────────────────────────────
        if not instrument_cfg.sessions_24h:
            session = self._get_session(now)
            if session is None:
                return FilterResult(
                    passed=False, session=None,
                    blocked_by=(
                        f"Hors session haute probabilité "
                        f"(heure UTC: {now.hour}h{now.minute:02d})"
                    ),
                    news_nearby=False, news_title=None,
                )
        else:
            session = "24h"   # BTC trade H24

        # ── 3. Filtre anti-spam ───────────────────────────────────
        max_daily = settings.MAX_SIGNALS_PER_INSTRUMENT_PER_DAY
        if signals_today >= max_daily:
            return FilterResult(
                passed=False, session=session,
                blocked_by=(
                    f"Anti-spam: {signals_today}/{max_daily} signaux "
                    f"déjà émis aujourd'hui sur {instrument}"
                ),
                news_nearby=False, news_title=None,
            )

        # ── 4. Filtre news HIGH IMPACT ─────────────────────────────
        news_nearby, news_title = await self._check_news(instrument)
        if news_nearby:
            return FilterResult(
                passed=False, session=session,
                blocked_by=f"News HIGH IMPACT proche: {news_title}",
                news_nearby=True, news_title=news_title,
            )

        # ── 5. Filtre spread (optionnel) ──────────────────────────
        if current_spread is not None:
            max_spread = instrument_cfg.avg_spread_pips * 3.0  # 3× la moyenne
            if current_spread > max_spread:
                return FilterResult(
                    passed=False, session=session,
                    blocked_by=(
                        f"Spread trop élevé: {current_spread:.1f} pips "
                        f"(max: {max_spread:.1f} pips)"
                    ),
                    news_nearby=False, news_title=None,
                )

        # ── Tous les filtres passés ───────────────────────────────
        logger.debug(
            f"✅ Filtres passés — {instrument} | "
            f"Session: {session} | "
            f"Signaux aujourd'hui: {signals_today}/{max_daily}"
        )

        return FilterResult(
            passed=True, session=session,
            blocked_by=None,
            news_nearby=False, news_title=None,
        )

    # ── SESSION ───────────────────────────────────────────────────

    def _get_session(self, now: datetime) -> str | None:
        """
        Retourne la session active ou None si hors session.

        Priorité : overlap > london > ny
        """
        hour   = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60.0

        # London/NY Overlap (13h-17h UTC) — meilleure session
        if 13.0 <= time_decimal < 17.0:
            return "overlap"

        # London Open (7h-12h UTC)
        if 7.0 <= time_decimal < 13.0:
            return "london"

        # NY (17h-20h UTC — fin de session NY)
        if 17.0 <= time_decimal < 20.0:
            return "ny"

        return None   # Hors session (Asia)

    def get_session_info(self) -> dict:
        """Retourne des infos complètes sur la session actuelle."""
        now     = datetime.utcnow()
        session = self._get_session(now)
        return {
            "current":     session,
            "utc_hour":    now.hour,
            "utc_minute":  now.minute,
            "is_trading":  session is not None,
            "next_session": self._next_session_open(now),
        }

    def _next_session_open(self, now: datetime) -> str:
        """Calcule l'ouverture de la prochaine session."""
        hour = now.hour
        if hour < 7:
            return f"London dans {7 - hour}h"
        if hour < 13:
            return f"Overlap dans {13 - hour}h"
        if hour < 17:
            return "Overlap en cours"
        return f"London tomorrow dans {31 - hour}h"

    # ── NEWS FILTER ───────────────────────────────────────────────

    async def _check_news(
        self,
        instrument: str,
    ) -> tuple[bool, str | None]:
        """
        Vérifie s'il y a une news HIGH IMPACT dans les 30 min.
        Utilise NewsAPI + cache de 30 min pour limiter les appels.

        Returns:
            (news_nearby: bool, news_title: str | None)
        """
        now = datetime.utcnow()

        # Rafraîchit le cache si nécessaire
        if (
            self._news_cache_ts is None
            or now - self._news_cache_ts > self._news_cache_ttl
        ):
            await self._refresh_news_cache()

        # Fenêtre de 30 min autour du moment actuel
        window_start = now - timedelta(minutes=30)
        window_end   = now + timedelta(minutes=30)

        for article in self._news_cache:
            try:
                pub_at = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except (KeyError, ValueError):
                continue

            if not (window_start <= pub_at <= window_end):
                continue

            title = (article.get("title") or "").lower()
            desc  = (article.get("description") or "").lower()
            text  = f"{title} {desc}"

            # Vérifie les mots-clés HIGH IMPACT
            for keyword in self.HIGH_IMPACT_KEYWORDS:
                if keyword in text:
                    logger.warning(
                        f"⚠️ News HIGH IMPACT détectée: "
                        f"{article.get('title', '')[:80]}"
                    )
                    return True, article.get("title", "")[:100]

        return False, None

    async def _refresh_news_cache(self):
        """Rafraîchit le cache des news depuis NewsAPI."""
        loop = asyncio.get_event_loop()

        def _fetch():
            return self._news_client.get_everything(
                q           = "gold forex bitcoin interest rate inflation federal reserve",
                language    = "en",
                sort_by     = "publishedAt",
                page_size   = 50,
                from_param  = (
                    datetime.utcnow() - timedelta(hours=2)
                ).strftime("%Y-%m-%dT%H:%M:%S"),
            )

        try:
            response = await loop.run_in_executor(None, _fetch)
            self._news_cache    = response.get("articles", [])
            self._news_cache_ts = datetime.utcnow()
            logger.debug(
                f"📰 Cache news rafraîchi — "
                f"{len(self._news_cache)} articles"
            )
        except Exception as e:
            logger.warning(f"NewsAPI error (non bloquant): {e}")
            self._news_cache = []

    # ── UTILITAIRES ───────────────────────────────────────────────

    @staticmethod
    def is_high_volatility_period(now: datetime | None = None) -> bool:
        """
        Détecte les périodes de haute volatilité connues :
          • 30 min autour du London Open (7h UTC)
          • 30 min autour du NY Open (13h UTC)
          • 30 min autour de la clôture NY (20h UTC)
        Ces périodes ont des mouvements erratiques = éviter les entrées.
        """
        if now is None:
            now = datetime.utcnow()

        volatile_hours = [(7, 0), (13, 0), (20, 0)]   # (heure, minute)
        for h, m in volatile_hours:
            open_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
            diff      = abs((now - open_time).total_seconds()) / 60
            if diff <= 30:
                return True
        return False
