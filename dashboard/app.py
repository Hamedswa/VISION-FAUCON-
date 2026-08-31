# dashboard/app.py
# ─────────────────────────────────────────────────────────────────
#  Dashboard FastAPI — Interface web du bot de trading
#
#  Routes :
#   GET  /              — Page principale (HTML)
#   GET  /api/overview  — KPIs globaux (JSON)
#   GET  /api/charts/equity     — Courbe d'équité (JSON Plotly)
#   GET  /api/charts/winrate    — Jauge Win Rate
#   GET  /api/charts/tpsl       — Distribution TP/SL
#   GET  /api/charts/session    — Performance par session
#   GET  /api/charts/score      — Performance par score
#   GET  /api/charts/monthly    — P&L mensuel
#   GET  /api/signals           — Historique signaux (JSON)
#   GET  /api/signals/active    — Signaux actifs
#   GET  /api/status            — Statut du bot
#   POST /api/backtest          — Lance un backtest
#   GET  /api/backtest/{id}     — Résultat backtest
#   WebSocket /ws/live          — Mises à jour temps réel
# ─────────────────────────────────────────────────────────────────

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from config import settings
from adaptive.performance_tracker import PerformanceTracker
from backtesting.backtester import Backtester, BacktestConfig
from database.db_manager import DatabaseManager
from database.models import SignalStatus
from data.data_manager import DataManager
from .charts import ChartBuilder


# ── State global ──────────────────────────────────────────────────
_db:      DatabaseManager | None = None
_data:    DataManager     | None = None
_tracker: PerformanceTracker | None = None
_backtester: Backtester   | None = None

# Cache des graphiques (TTL 60s pour ne pas recalculer à chaque requête)
_chart_cache: dict[str, tuple[str, datetime]] = {}
CHART_CACHE_TTL = 60   # secondes

# WebSocket connections actives
_ws_connections: list[WebSocket] = []


def create_app(
    db:      DatabaseManager,
    data:    DataManager,
    tracker: PerformanceTracker,
) -> FastAPI:
    """
    Factory — crée l'application FastAPI avec toutes les dépendances.
    Appelée depuis main.py.
    """
    global _db, _data, _tracker, _backtester
    _db       = db
    _data     = data
    _tracker  = tracker
    _backtester = Backtester(data, db)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Démarrage : lance la diffusion live aux WebSocket
        task = asyncio.create_task(_live_broadcast_loop())
        logger.info("✅ Dashboard FastAPI démarré")
        yield
        task.cancel()
        logger.info("⏹️ Dashboard arrêté")

    app = FastAPI(
        title       = "Trading Bot Dashboard",
        description = "Bot SMC/ICT — Monitoring en temps réel",
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── PAGE PRINCIPALE ───────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Page principale du dashboard."""
        return HTMLResponse(_build_html())

    # ── API — VUE D'ENSEMBLE ──────────────────────────────────────

    @app.get("/api/overview")
    async def get_overview():
        """
        Retourne tous les KPIs globaux en JSON.
        Utilisé pour les cartes du dashboard.
        """
        try:
            report_30d = await _tracker.generate_report(days=30)
            report_7d  = await _tracker.generate_report(days=7)
            report_1d  = await _tracker.generate_report(days=1)

            # Prix actuels
            prices = {}
            try:
                prices = await _data.get_current_prices(
                    ["XAUUSD", "BTCUSD", "EURUSD"]
                )
            except Exception:
                pass

            # Signaux actifs
            active_signals = await _db.get_active_signals()

            return JSONResponse({
                "kpis_30d":      ChartBuilder.kpi_cards(report_30d),
                "kpis_7d":       ChartBuilder.kpi_cards(report_7d),
                "kpis_1d":       ChartBuilder.kpi_cards(report_1d),
                "active_trades": len(active_signals),
                "prices":        prices,
                "bot_status": {
                    "env":        settings.BOT_ENV,
                    "min_score":  settings.MIN_CONFLUENCE_SCORE,
                    "risk_pct":   settings.RISK_PER_TRADE,
                    "max_trades": settings.MAX_OPEN_TRADES,
                },
                "generated_at": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            logger.error(f"API /overview error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── API — GRAPHIQUES ──────────────────────────────────────────

    @app.get("/api/charts/equity")
    async def chart_equity(days: int = 30, instrument: str | None = None):
        """Courbe d'équité."""
        cache_key = f"equity_{days}_{instrument}"
        cached    = _get_cache(cache_key)
        if cached:
            return JSONResponse({"chart": cached})

        report = await _tracker.generate_report(days=days, instrument=instrument)
        chart  = ChartBuilder.equity_curve(
            equity_points   = report.equity_curve if hasattr(report, "equity_curve") else [],
            initial_balance = 10000.0,
            title           = f"Équité ({days}j) {instrument or ''}",
        )
        _set_cache(cache_key, chart)
        return JSONResponse({"chart": json.loads(chart)})

    @app.get("/api/charts/winrate")
    async def chart_winrate(days: int = 30):
        """Jauge Win Rate."""
        cache_key = f"winrate_{days}"
        cached    = _get_cache(cache_key)
        if cached:
            return JSONResponse({"chart": cached})

        wr    = await _tracker.get_win_rate(days=days)
        chart = ChartBuilder.win_rate_gauge(wr)
        _set_cache(cache_key, chart)
        return JSONResponse({"chart": json.loads(chart)})

    @app.get("/api/charts/tpsl")
    async def chart_tpsl(days: int = 30):
        """Distribution TP/SL."""
        report = await _tracker.generate_report(days=days)
        chart  = ChartBuilder.tp_sl_distribution(
            tp1 = report.tp1_hit_count,
            tp2 = report.tp2_hit_count,
            tp3 = report.tp3_hit_count,
            sl  = report.sl_hit_count,
        )
        return JSONResponse({"chart": json.loads(chart)})

    @app.get("/api/charts/session")
    async def chart_session(days: int = 30):
        """Performance par session."""
        report = await _tracker.generate_report(days=days)
        chart  = ChartBuilder.session_performance(report.by_session)
        return JSONResponse({"chart": json.loads(chart)})

    @app.get("/api/charts/score")
    async def chart_score(days: int = 30):
        """Win Rate par score de confluence."""
        report = await _tracker.generate_report(days=days)
        chart  = ChartBuilder.score_performance(report.by_score_range)
        return JSONResponse({"chart": json.loads(chart)})

    @app.get("/api/charts/monthly")
    async def chart_monthly():
        """P&L mensuel."""
        cache_key = "monthly"
        cached    = _get_cache(cache_key, ttl=300)
        if cached:
            return JSONResponse({"chart": cached})

        # Données sur 6 mois
        signals = await _db.get_signals_for_backtest(months=6)

        from backtesting.metrics import BacktestMetrics, TradeRecord
        # Construit des TradeRecord simplifiés depuis les signals DB
        trade_records = []
        for s in signals:
            if s.exit_at and s.pnl_usd is not None:
                is_win = s.status in (
                    SignalStatus.TP1_HIT,
                    SignalStatus.TP2_HIT,
                    SignalStatus.TP3_HIT,
                )
                trade_records.append(TradeRecord(
                    signal_id    = s.id,
                    instrument   = s.instrument,
                    direction    = s.direction.value,
                    entry        = s.entry_price or 0,
                    exit_price   = s.exit_price or 0,
                    sl           = s.sl_price or 0,
                    tp1          = s.tp1_price or 0,
                    tp2          = s.tp2_price or 0,
                    tp3          = s.tp3_price or 0,
                    pnl_usd      = s.pnl_usd or 0,
                    pnl_pips     = s.pnl_pips or 0,
                    is_win       = is_win,
                    tp_level_hit = None,
                    duration_min = s.duration_minutes or 0,
                    score        = s.confluence_score or 0,
                    session      = s.session,
                    opened_at    = s.created_at,
                    closed_at    = s.exit_at,
                    rr_ratio     = s.rr_ratio or 0,
                    rr_achieved  = 0.0,
                ))

        metrics = BacktestMetrics().calculate(trade_records)
        chart   = ChartBuilder.monthly_heatmap(metrics.monthly_stats)
        _set_cache(cache_key, chart, ttl=300)
        return JSONResponse({"chart": json.loads(chart)})

    # ── API — SIGNAUX ────────────────────────────────────────────

    @app.get("/api/signals")
    async def get_signals(
        limit:      int = 50,
        instrument: str | None = None,
        status:     str | None = None,
    ):
        """Historique des signaux avec filtres."""
        signals = await _db.get_signals_for_backtest(
            instrument = instrument,
            months     = 1,
        )

        # Filtre par statut
        if status:
            signals = [s for s in signals if s.status.value == status.upper()]

        # Limite et tri
        signals = sorted(signals, key=lambda s: s.created_at, reverse=True)[:limit]

        return JSONResponse({
            "signals": [
                {
                    "id":          s.id,
                    "instrument":  s.instrument,
                    "direction":   s.direction.value,
                    "status":      s.status.value,
                    "score":       s.confluence_score,
                    "entry":       s.entry_price,
                    "sl":          s.sl_price,
                    "tp1":         s.tp1_price,
                    "tp2":         s.tp2_price,
                    "tp3":         s.tp3_price,
                    "rr":          s.rr_ratio,
                    "pnl_usd":     s.pnl_usd,
                    "pnl_pips":    s.pnl_pips,
                    "session":     s.session,
                    "created_at":  s.created_at.isoformat(),
                    "exit_at":     s.exit_at.isoformat() if s.exit_at else None,
                    "duration":    s.duration_minutes,
                }
                for s in signals
            ],
            "total": len(signals),
        })

    @app.get("/api/signals/active")
    async def get_active_signals():
        """Signaux et trades actuellement actifs."""
        signals = await _db.get_active_signals()
        prices  = await _data.get_current_prices(
            list({s.instrument for s in signals})
        )

        result = []
        for s in signals:
            current = prices.get(s.instrument, 0)
            entry   = s.entry_price or s.actual_entry or 0
            is_long = s.direction.value.upper() == "LONG"

            if entry > 0 and current > 0:
                pnl_live = current - entry if is_long else entry - current
            else:
                pnl_live = 0.0

            result.append({
                "id":          s.id,
                "instrument":  s.instrument,
                "direction":   s.direction.value,
                "entry":       entry,
                "sl":          s.sl_price,
                "tp1":         s.tp1_price,
                "tp2":         s.tp2_price,
                "tp3":         s.tp3_price,
                "current":     current,
                "pnl_live":    round(pnl_live, 4),
                "score":       s.confluence_score,
                "session":     s.session,
                "opened_at":   s.created_at.isoformat(),
            })

        return JSONResponse({"active": result, "count": len(result)})

    # ── API — STATUT DU BOT ───────────────────────────────────────

    @app.get("/api/status")
    async def get_status():
        """Statut global du bot."""
        connections = await _data.validate_connection()
        wr_7d       = await _tracker.get_win_rate(days=7)
        wr_30d      = await _tracker.get_win_rate(days=30)
        active      = await _db.get_active_signals()

        return JSONResponse({
            "status":       "running",
            "env":          settings.BOT_ENV,
            "connections":  connections,
            "performance": {
                "wr_7d":    f"{wr_7d:.1%}",
                "wr_30d":   f"{wr_30d:.1%}",
                "on_target": wr_30d >= 0.60,
            },
            "active_trades":  len(active),
            "config": {
                "min_score":    settings.MIN_CONFLUENCE_SCORE,
                "risk_pct":     settings.RISK_PER_TRADE,
                "max_trades":   settings.MAX_OPEN_TRADES,
                "max_dd":       settings.MAX_DAILY_LOSS,
                "min_rr":       settings.MIN_RR_RATIO,
                "optimizer":    settings.OPTIMIZER_ENABLED,
            },
            "timestamp": datetime.utcnow().isoformat(),
        })

    # ── API — BACKTEST ────────────────────────────────────────────

    @app.post("/api/backtest")
    async def run_backtest(payload: dict):
        """
        Lance un backtest à la demande.
        Body: {"instrument": "XAUUSD", "months": 3}
        """
        instrument = payload.get("instrument", "XAUUSD")
        months     = int(payload.get("months", 3))
        months     = max(1, min(months, 12))

        config = BacktestConfig(
            instrument        = instrument,
            months            = months,
            initial_balance   = 10000.0,
            risk_pct          = settings.RISK_PER_TRADE,
            min_score         = settings.MIN_CONFLUENCE_SCORE,
            atr_sl_multiplier = settings.ATR_SL_MULTIPLIER,
            tp1_ratio         = settings.TP1_RATIO,
            tp2_ratio         = settings.TP2_RATIO,
            tp3_ratio         = settings.TP3_RATIO,
            min_rr            = settings.MIN_RR_RATIO,
        )

        try:
            result = await _backtester.run(config)
            m      = result.metrics

            return JSONResponse({
                "success":   True,
                "instrument": instrument,
                "months":    months,
                "metrics": {
                    "grade":          m.grade,
                    "win_rate":       f"{m.win_rate:.1%}",
                    "profit_factor":  m.profit_factor,
                    "sharpe":         m.sharpe_ratio,
                    "sortino":        m.sortino_ratio,
                    "calmar":         m.calmar_ratio,
                    "total_pnl":      m.total_pnl_usd,
                    "max_drawdown":   m.max_drawdown_pct,
                    "total_trades":   m.total_trades,
                    "tp1_rate":       f"{m.tp1_rate:.1%}",
                    "tp2_rate":       f"{m.tp2_rate:.1%}",
                    "tp3_rate":       f"{m.tp3_rate:.1%}",
                    "sl_rate":        f"{m.sl_rate:.1%}",
                    "avg_rr":         m.avg_rr_achieved,
                    "expectancy":     m.expectancy_usd,
                    "win_streak":     m.max_win_streak,
                    "loss_streak":    m.max_loss_streak,
                    "meets_target":   result.meets_target,
                },
                "equity_chart": json.loads(
                    ChartBuilder.equity_curve(
                        m.equity_curve,
                        title=f"Backtest {instrument} {months}m",
                    )
                ),
                "monthly_chart": json.loads(
                    ChartBuilder.monthly_heatmap(m.monthly_stats)
                ),
                "summary":   result.summary(),
                "duration":  f"{result.duration_sec:.1f}s",
            })

        except Exception as e:
            logger.error(f"Backtest API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── WEBSOCKET — LIVE ──────────────────────────────────────────

    @app.websocket("/ws/live")
    async def websocket_live(ws: WebSocket):
        """
        WebSocket pour les mises à jour en temps réel.
        Envoie toutes les 10s : prix, trades actifs, WR live.
        """
        await ws.accept()
        _ws_connections.append(ws)
        logger.info(f"WebSocket connecté ({len(_ws_connections)} total)")

        try:
            while True:
                # Attend un message du client (ping/pong ou commande)
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)

        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            if ws in _ws_connections:
                _ws_connections.remove(ws)
            logger.info(f"WebSocket déconnecté ({len(_ws_connections)} restants)")

    return app


# ── BROADCAST LOOP ────────────────────────────────────────────────

async def _live_broadcast_loop():
    """
    Diffuse les mises à jour en temps réel à tous les WebSocket.
    S'exécute toutes les 10 secondes.
    """
    while True:
        try:
            await asyncio.sleep(10)

            if not _ws_connections or not _db or not _data:
                continue

            # Construit le payload live
            active  = await _db.get_active_signals()
            symbols = list({s.instrument for s in active}) or ["XAUUSD"]
            prices  = await _data.get_current_prices(symbols)
            wr_live = await _tracker.get_win_rate(days=7)

            payload = json.dumps({
                "type":          "live_update",
                "prices":        prices,
                "active_count":  len(active),
                "wr_7d":         f"{wr_live:.1%}",
                "timestamp":     datetime.utcnow().isoformat(),
            })

            # Diffuse à tous les clients connectés
            dead = []
            for ws in _ws_connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                _ws_connections.remove(ws)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Broadcast loop error: {e}")


# ── CACHE ─────────────────────────────────────────────────────────

def _get_cache(key: str, ttl: int = CHART_CACHE_TTL) -> Any | None:
    if key in _chart_cache:
        data, ts = _chart_cache[key]
        if (datetime.utcnow() - ts).total_seconds() < ttl:
            return json.loads(data)
    return None


def _set_cache(key: str, data: str, ttl: int = CHART_CACHE_TTL):
    _chart_cache[key] = (data, datetime.utcnow())


# ── HTML TEMPLATE ─────────────────────────────────────────────────

def _build_html() -> str:
    """
    Page HTML complète du dashboard.
    Design sombre, responsive, Plotly + WebSocket live.
    """
    return """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg:       #0D1117;
    --bg-card:  #161B22;
    --border:   #30363D;
    --text:     #E6EDF3;
    --muted:    #8B949E;
    --green:    #00C896;
    --red:      #FF4C4C;
    --gold:     #FFD700;
    --blue:     #4C9BE8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: Inter, -apple-system, sans-serif;
    font-size: 14px;
  }

  /* ── HEADER ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-card);
  }
  .logo { font-size: 18px; font-weight: 700; }
  .logo span { color: var(--gold); }
  .live-badge {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: var(--muted);
  }
  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
  }

  /* ── LAYOUT ── */
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 20px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 16px; margin-bottom: 20px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px; margin-bottom: 20px;
  }

  /* ── CARTES KPI ── */
  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }
  .kpi-card.green::before { background: var(--green); }
  .kpi-card.red::before   { background: var(--red); }
  .kpi-card.gold::before  { background: var(--gold); }
  .kpi-card.blue::before  { background: var(--blue); }

  .kpi-label {
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px; color: var(--muted); margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 28px; font-weight: 700; margin-bottom: 4px;
  }
  .kpi-value.green { color: var(--green); }
  .kpi-value.red   { color: var(--red); }
  .kpi-value.gold  { color: var(--gold); }
  .kpi-value.blue  { color: var(--blue); }
  .kpi-sub  { font-size: 11px; color: var(--muted); }
  .kpi-target { font-size: 10px; color: var(--muted); margin-top: 4px; }

  /* ── CHART CARDS ── */
  .chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  .chart-card h3 {
    font-size: 13px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }

  /* ── TABLE SIGNAUX ── */
  .signals-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .signals-table th {
    text-align: left;
    padding: 10px 12px;
    color: var(--muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .signals-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
  }
  .signals-table tr:last-child td { border-bottom: none; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-long   { background: rgba(0,200,150,0.15); color: var(--green); }
  .badge-short  { background: rgba(255,76,76,0.15);  color: var(--red); }
  .badge-win    { background: rgba(0,200,150,0.2);  color: var(--green); }
  .badge-loss   { background: rgba(255,76,76,0.2);   color: var(--red); }
  .badge-active { background: rgba(76,155,232,0.2);  color: var(--blue); }
  .badge-pending { background: rgba(255,215,0,0.2); color: var(--gold); }

  /* ── TABS ── */
  .tabs {
    display: flex; gap: 4px; margin-bottom: 16px;
  }
  .tab {
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    color: var(--muted);
    background: transparent;
    border: 1px solid transparent;
    transition: all 0.2s;
  }
  .tab.active {
    background: var(--bg-card);
    border-color: var(--border);
    color: var(--text);
  }
  .tab:hover { color: var(--text); }

  /* ── BACKTEST ── */
  .backtest-form {
    display: flex; gap: 12px; align-items: flex-end;
    margin-bottom: 16px;
  }
  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group label { font-size: 11px; color: var(--muted); }
  select, .btn {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    cursor: pointer;
  }
  .btn {
    background: var(--green);
    color: #000;
    font-weight: 600;
    border-color: var(--green);
    transition: opacity 0.2s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── LIVE PRICE BAR ── */
  .price-bar {
    display: flex; gap: 24px;
    padding: 10px 0; margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .price-item { display: flex; align-items: center; gap: 8px; }
  .price-symbol { font-size: 11px; color: var(--muted); font-weight: 600; }
  .price-val { font-size: 14px; font-weight: 700; }

  /* ── RESPONSIVE ── */
  @media (max-width: 768px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
    .backtest-form { flex-direction: column; }
  }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">🤖 Trading Bot <span>SMC/ICT</span></div>
  <div class="live-badge">
    <div class="dot" id="ws-dot"></div>
    <span id="ws-status">Connexion...</span>
    <span id="last-update" style="margin-left:12px"></span>
  </div>
</header>

<div class="container">

  <!-- PRIX LIVE -->
  <div class="price-bar" id="price-bar">
    <div class="price-item">
      <span class="price-symbol">XAU/USD</span>
      <span class="price-val" id="price-XAUUSD">—</span>
    </div>
    <div class="price-item">
      <span class="price-symbol">BTC/USD</span>
      <span class="price-val" id="price-BTCUSD">—</span>
    </div>
    <div class="price-item">
      <span class="price-symbol">EUR/USD</span>
      <span class="price-val" id="price-EURUSD">—</span>
    </div>
    <div class="price-item" style="margin-left:auto">
      <span class="price-symbol">TRADES ACTIFS</span>
      <span class="price-val" id="active-count" style="color:var(--blue)">—</span>
    </div>
    <div class="price-item">
      <span class="price-symbol">WR 7 JOURS</span>
      <span class="price-val" id="wr-live">—</span>
    </div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab('overview')">Vue d'ensemble</button>
    <button class="tab" onclick="switchTab('signals')">Signaux</button>
    <button class="tab" onclick="switchTab('backtest')">Backtest</button>
  </div>

  <!-- TAB OVERVIEW -->
  <div id="tab-overview">

    <!-- KPI CARDS (30 jours) -->
    <div class="grid-4" id="kpi-grid">
      <!-- Rempli par JS -->
    </div>

    <!-- GRAPHIQUES PRINCIPAUX -->
    <div class="chart-card" style="margin-bottom:16px">
      <h3>Courbe d'équité — 30 jours</h3>
      <div id="chart-equity"></div>
    </div>

    <div class="grid-3">
      <div class="chart-card">
        <div id="chart-winrate"></div>
      </div>
      <div class="chart-card">
        <div id="chart-tpsl"></div>
      </div>
      <div class="chart-card">
        <div id="chart-session"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="chart-card">
        <div id="chart-score"></div>
      </div>
      <div class="chart-card">
        <div id="chart-monthly"></div>
      </div>
    </div>

  </div>

  <!-- TAB SIGNALS -->
  <div id="tab-signals" style="display:none">
    <div class="chart-card">
      <h3>Signaux récents</h3>
      <div style="overflow-x:auto">
        <table class="signals-table" id="signals-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Instrument</th>
              <th>Direction</th>
              <th>Statut</th>
              <th>Score</th>
              <th>Entrée</th>
              <th>SL</th>
              <th>TP2</th>
              <th>R:R</th>
              <th>P&L</th>
              <th>Session</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody id="signals-body">
            <tr><td colspan="12" style="text-align:center;color:var(--muted);padding:24px">Chargement...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB BACKTEST -->
  <div id="tab-backtest" style="display:none">
    <div class="chart-card">
      <h3>Lancer un Backtest</h3>
      <div class="backtest-form">
        <div class="form-group">
          <label>Instrument</label>
          <select id="bt-instrument">
            <option>XAUUSD</option>
            <option>BTCUSD</option>
            <option>EURUSD</option>
            <option>GBPUSD</option>
          </select>
        </div>
        <div class="form-group">
          <label>Période</label>
          <select id="bt-months">
            <option value="1">1 mois</option>
            <option value="3" selected>3 mois</option>
            <option value="6">6 mois</option>
          </select>
        </div>
        <button class="btn" onclick="runBacktest()" id="bt-btn">
          ▶ Lancer
        </button>
      </div>
      <div id="bt-results" style="display:none">
        <div class="grid-4" id="bt-kpis" style="margin-bottom:16px"></div>
        <div id="chart-bt-equity"></div>
        <div id="chart-bt-monthly" style="margin-top:16px"></div>
        <pre id="bt-summary" style="background:var(--bg);padding:16px;border-radius:8px;margin-top:16px;font-size:12px;color:var(--muted);white-space:pre-wrap"></pre>
      </div>
    </div>
  </div>

</div><!-- /container -->

<script>
// ── WebSocket Live ────────────────────────────────────────────────
const wsUrl = `ws://${location.host}/ws/live`;
let ws;

function connectWs() {
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    document.getElementById('ws-dot').style.background = 'var(--green)';
    document.getElementById('ws-status').textContent = 'Live';
  };

  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type !== 'live_update') return;

    // Prix
    for (const [sym, price] of Object.entries(d.prices || {})) {
      const el = document.getElementById('price-' + sym);
      if (el) el.textContent = Number(price).toLocaleString('fr-FR', {minimumFractionDigits: 2});
    }

    document.getElementById('active-count').textContent = d.active_count ?? '—';
    document.getElementById('wr-live').textContent = d.wr_7d || '—';
    document.getElementById('last-update').textContent =
      new Date().toLocaleTimeString('fr-FR');
  };

  ws.onclose = () => {
    document.getElementById('ws-dot').style.background = 'var(--red)';
    document.getElementById('ws-status').textContent = 'Reconnexion...';
    setTimeout(connectWs, 3000);
  };
}
connectWs();

// ── Chargement initial ────────────────────────────────────────────
async function loadOverview() {
  const res = await fetch('/api/overview');
  const d   = await res.json();

  // KPI Cards
  const kpis = d.kpis_30d;
  const grid  = document.getElementById('kpi-grid');
  grid.innerHTML = Object.entries(kpis).map(([key, kpi]) => `
    <div class="kpi-card ${kpi.color}">
      <div class="kpi-label">${kpi.label}</div>
      <div class="kpi-value ${kpi.color}">${kpi.value}</div>
      <div class="kpi-sub">${kpi.sub}</div>
      <div class="kpi-target">${kpi.target}</div>
    </div>
  `).join('');
}

async function loadCharts() {
  // Charge les graphiques en parallèle
  const endpoints = [
    {id: 'chart-equity',   url: '/api/charts/equity'},
    {id: 'chart-winrate',  url: '/api/charts/winrate'},
    {id: 'chart-tpsl',     url: '/api/charts/tpsl'},
    {id: 'chart-session',  url: '/api/charts/session'},
    {id: 'chart-score',    url: '/api/charts/score'},
    {id: 'chart-monthly',  url: '/api/charts/monthly'},
  ];

  await Promise.all(endpoints.map(async ({id, url}) => {
    try {
      const res  = await fetch(url);
      const data = await res.json();
      if (data.chart) {
        Plotly.newPlot(id, data.chart.data, data.chart.layout, {
          responsive: true, displayModeBar: false
        });
      }
    } catch (e) {
      console.error('Chart error', id, e);
    }
  }));
}

async function loadSignals() {
  const res  = await fetch('/api/signals?limit=50');
  const data = await res.json();
  const tbody = document.getElementById('signals-body');

  if (!data.signals.length) {
    tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:var(--muted);padding:24px">Aucun signal</td></tr>';
    return;
  }

  tbody.innerHTML = data.signals.map(s => {
    const dir     = s.direction === 'LONG' ? 'badge-long' : 'badge-short';
    const status  = getStatusBadge(s.status);
    const pnl     = s.pnl_usd != null
      ? `<span style="color:${s.pnl_usd >= 0 ? 'var(--green)' : 'var(--red)'}">${s.pnl_usd >= 0 ? '+' : ''}${s.pnl_usd.toFixed(2)}$</span>`
      : '—';
    const date = new Date(s.created_at).toLocaleString('fr-FR', {
      day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit'
    });
    return `
      <tr>
        <td style="color:var(--muted)">#${s.id}</td>
        <td><b>${s.instrument}</b></td>
        <td><span class="badge ${dir}">${s.direction}</span></td>
        <td>${status}</td>
        <td><b style="color:${s.score >= 85 ? 'var(--gold)' : 'inherit'}">${s.score}</b></td>
        <td>${s.entry?.toFixed(4) ?? '—'}</td>
        <td>${s.sl?.toFixed(4) ?? '—'}</td>
        <td>${s.tp2?.toFixed(4) ?? '—'}</td>
        <td>${s.rr?.toFixed(1) ?? '—'}</td>
        <td>${pnl}</td>
        <td style="color:var(--muted)">${s.session?.toUpperCase() ?? '—'}</td>
        <td style="color:var(--muted)">${date}</td>
      </tr>
    `;
  }).join('');
}

function getStatusBadge(status) {
  const map = {
    'ACTIVE':   '<span class="badge badge-active">ACTIF</span>',
    'TP1_HIT':  '<span class="badge badge-win">TP1 ✅</span>',
    'TP2_HIT':  '<span class="badge badge-win">TP2 🥈</span>',
    'TP3_HIT':  '<span class="badge badge-win">TP3 🏆</span>',
    'SL_HIT':   '<span class="badge badge-loss">SL 🔴</span>',
    'PENDING':  '<span class="badge badge-pending">EN ATTENTE</span>',
    'CANCELLED':'<span class="badge" style="opacity:.5">ANNULÉ</span>',
  };
  return map[status] || status;
}

// ── TABS ──────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-overview').style.display  = tab === 'overview'  ? '' : 'none';
  document.getElementById('tab-signals').style.display   = tab === 'signals'   ? '' : 'none';
  document.getElementById('tab-backtest').style.display  = tab === 'backtest'  ? '' : 'none';

  if (tab === 'signals')  loadSignals();
}

// ── BACKTEST ──────────────────────────────────────────────────────
async function runBacktest() {
  const btn    = document.getElementById('bt-btn');
  const instr  = document.getElementById('bt-instrument').value;
  const months = parseInt(document.getElementById('bt-months').value);

  btn.disabled    = true;
  btn.textContent = '⏳ En cours...';

  try {
    const res  = await fetch('/api/backtest', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({instrument: instr, months}),
    });
    const data = await res.json();

    // KPIs backtest
    const m    = data.metrics;
    const kpis = [
      {label:'Grade',       value: m.grade,         color: m.grade.startsWith('A') ? 'green' : m.grade === 'B' ? 'gold' : 'red'},
      {label:'Win Rate',    value: m.win_rate,       color: parseFloat(m.win_rate) >= 60 ? 'green' : 'red'},
      {label:'Profit Factor', value: m.profit_factor.toFixed(2), color: m.profit_factor >= 1.3 ? 'green' : 'red'},
      {label:'P&L Total',   value: `${m.total_pnl >= 0 ? '+' : ''}${m.total_pnl.toFixed(2)}$`, color: m.total_pnl >= 0 ? 'green' : 'red'},
      {label:'Drawdown Max', value: `${m.max_drawdown.toFixed(1)}%`, color: m.max_drawdown < 15 ? 'green' : 'red'},
      {label:'Trades',      value: m.total_trades,  color: 'blue'},
      {label:'Sharpe',      value: m.sharpe.toFixed(2), color: m.sharpe >= 1 ? 'green' : 'red'},
      {label:'Objectif',    value: m.meets_target ? '✅ ATTEINT' : '❌ NON', color: m.meets_target ? 'green' : 'red'},
    ];

    document.getElementById('bt-kpis').innerHTML = kpis.map(k => `
      <div class="kpi-card ${k.color}">
        <div class="kpi-label">${k.label}</div>
        <div class="kpi-value ${k.color}">${k.value}</div>
      </div>
    `).join('');

    // Graphiques backtest
    if (data.equity_chart) {
      Plotly.newPlot('chart-bt-equity', data.equity_chart.data, data.equity_chart.layout, {
        responsive: true, displayModeBar: false
      });
    }
    if (data.monthly_chart) {
      Plotly.newPlot('chart-bt-monthly', data.monthly_chart.data, data.monthly_chart.layout, {
        responsive: true, displayModeBar: false
      });
    }

    document.getElementById('bt-summary').textContent = data.summary;
    document.getElementById('bt-results').style.display = '';

  } catch (e) {
    alert('Erreur backtest: ' + e.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = '▶ Lancer';
  }
}

// ── INIT ──────────────────────────────────────────────────────────
loadOverview();
loadCharts();
</script>
</body>
</html>"""
