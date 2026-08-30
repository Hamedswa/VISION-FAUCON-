# execution/risk_manager.py
# ─────────────────────────────────────────────────────────────────
#  Gestionnaire du risque — Gardien du capital
#
#  Responsabilités :
#   • Calcul de la taille de position (sizing)
#   • Vérification des limites de risque avant chaque trade
#   • Contrôle de la perte journalière maximale
#   • Contrôle du drawdown global
#   • Anti-corrélation (pas 2 trades identiques simultanés)
# ─────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from config import settings
from config.instruments import get_instrument, InstrumentConfig
from database.db_manager import DatabaseManager
from database.models import SignalStatus


@dataclass
class PositionSize:
    units:        float    # Nombre d'unités à trader
    risk_usd:     float    # Risque en USD pour ce trade
    risk_pct:     float    # % du capital risqué
    sl_pips:      float    # Distance SL en pips
    lot_size:     float    # Taille en lots standard
    valid:        bool
    reason:       str | None


@dataclass
class RiskCheck:
    approved:          bool
    rejection_reason:  str | None

    # Détail des vérifications
    daily_loss_ok:     bool
    max_trades_ok:     bool
    balance_ok:        bool
    correlation_ok:    bool
    drawdown_ok:       bool

    # Contexte
    current_balance:   float
    daily_loss_usd:    float
    daily_loss_pct:    float
    open_trades_count: int


class RiskManager:
    """
    Évalue si un nouveau trade peut être pris
    en fonction des règles de gestion du risque.
    Toutes les décisions sont mathématiques et déterministes.
    """

    def __init__(self, db: DatabaseManager):
        self._db           = db
        self._balance_open = 0.0   # Balance au début de la journée
        self._day_start    = datetime.utcnow().date()

    # ── VÉRIFICATION GLOBALE ──────────────────────────────────────

    async def check(
        self,
        instrument: str,
        direction:  str,
        balance:    float,
    ) -> RiskCheck:
        """
        Vérifie toutes les règles de risque avant d'autoriser un trade.

        Returns:
            RiskCheck avec approved=True si le trade peut être pris.
        """

        # ── 1. Balance minimum ────────────────────────────────────
        balance_ok = balance >= 50.0   # 50 USD minimum absolu

        # ── 2. Trades ouverts maximum ─────────────────────────────
        active_signals = await self._db.get_active_signals()
        open_count     = len(active_signals)
        max_trades_ok  = open_count < settings.MAX_OPEN_TRADES

        # ── 3. Perte journalière ──────────────────────────────────
        daily_loss_usd, daily_loss_pct = await self._get_daily_loss(balance)
        daily_loss_ok = daily_loss_pct < settings.MAX_DAILY_LOSS

        # ── 4. Drawdown global ────────────────────────────────────
        # On bloque si drawdown > 10% par rapport au balance de début de journée
        drawdown_ok = True
        if self._balance_open > 0:
            drawdown_pct = (self._balance_open - balance) / self._balance_open * 100
            drawdown_ok  = drawdown_pct < 10.0
        else:
            self._balance_open = balance  # Init première fois

        # ── 5. Anti-corrélation ───────────────────────────────────
        # Pas 2 trades dans la même direction sur le même instrument
        correlation_ok = True
        active_same = await self._db.get_active_signals(instrument)
        if any(
            s.direction.value.lower() == direction.lower()
            for s in active_same
        ):
            correlation_ok = False

        # ── Résultat final ────────────────────────────────────────
        approved = all([
            balance_ok,
            max_trades_ok,
            daily_loss_ok,
            drawdown_ok,
            correlation_ok,
        ])

        rejection = None
        if not approved:
            reasons = []
            if not balance_ok:      reasons.append(f"Balance trop faible ({balance:.2f} USD)")
            if not max_trades_ok:   reasons.append(f"Max trades atteint ({open_count}/{settings.MAX_OPEN_TRADES})")
            if not daily_loss_ok:   reasons.append(f"Perte journalière dépassée ({daily_loss_pct:.1f}% / {settings.MAX_DAILY_LOSS}%)")
            if not drawdown_ok:     reasons.append("Drawdown journalier > 10%")
            if not correlation_ok:  reasons.append(f"Trade {direction} déjà ouvert sur {instrument}")
            rejection = " | ".join(reasons)

        if approved:
            logger.info(f"✅ Risk check APPROUVÉ — {instrument} {direction}")
        else:
            logger.warning(f"🚫 Risk check REFUSÉ — {rejection}")

        return RiskCheck(
            approved          = approved,
            rejection_reason  = rejection,
            daily_loss_ok     = daily_loss_ok,
            max_trades_ok     = max_trades_ok,
            balance_ok        = balance_ok,
            correlation_ok    = correlation_ok,
            drawdown_ok       = drawdown_ok,
            current_balance   = balance,
            daily_loss_usd    = daily_loss_usd,
            daily_loss_pct    = daily_loss_pct,
            open_trades_count = open_count,
        )

    # ── SIZING ────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        balance:        float,
        entry:          float,
        sl_price:       float,
        instrument_cfg: InstrumentConfig,
        risk_pct:       float | None = None,
    ) -> PositionSize:
        """
        Calcule la taille de position optimale.

        Formule :
            risk_usd = balance × risk_pct / 100
            sl_pips  = |entry - sl| / pip_value
            units    = risk_usd / (sl_pips × pip_value)

        Applique les limites min/max de l'instrument.
        """
        risk_pct = risk_pct or settings.RISK_PER_TRADE

        if balance <= 0 or entry <= 0 or sl_price <= 0:
            return PositionSize(
                units=0, risk_usd=0, risk_pct=0,
                sl_pips=0, lot_size=0,
                valid=False, reason="Paramètres invalides"
            )

        # Montant à risquer
        risk_usd = balance * (risk_pct / 100.0)

        # Distance SL en pips
        sl_distance = abs(entry - sl_price)
        pip_val     = instrument_cfg.pip_value
        sl_pips     = sl_distance / pip_val if pip_val > 0 else 0

        if sl_pips == 0:
            return PositionSize(
                units=0, risk_usd=0, risk_pct=risk_pct,
                sl_pips=0, lot_size=0,
                valid=False, reason="SL distance nulle"
            )

        # Calcul des unités
        # Pour OANDA : 1 lot standard = 100 000 unités
        # pip_value pour EUR/USD = 10 USD par pip pour 1 lot
        # units = risk_usd / (sl_pips × pip_value_per_unit)
        pip_value_per_unit = pip_val / 100000.0  # pip value pour 1 unité
        units = risk_usd / (sl_pips * pip_value_per_unit) if pip_value_per_unit > 0 else 0

        # Pour crypto (CCXT) : unités en BTC
        if instrument_cfg.broker == "ccxt":
            # risk_usd / (sl_distance_en_USD)
            units = risk_usd / sl_distance if sl_distance > 0 else 0

        # Conversion en lots
        lot_size = units / 100000.0 if instrument_cfg.broker == "oanda" else units

        # Application des limites instrument
        lot_size = max(instrument_cfg.min_lot,
                       min(instrument_cfg.max_lot, lot_size))
        lot_size = self._round_to_step(lot_size, instrument_cfg.lot_step)
        units    
