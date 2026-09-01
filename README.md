# 🤖 Trading Bot SMC/ICT

Bot de trading algorithmique **math-first + adaptatif** basé sur les
concepts Smart Money Concepts (SMC) et ICT.

**Objectif : ≥ 60% Win Rate (3 trades gagnants sur 5)**

---

## 🏗️ Architecture

- **Math Engine** — SMC/ICT Detection, Scoring /100, TP/SL, Validation
- **Adaptive Layer** — Performance Tracker, Pattern Analyzer, Optuna Optimizer
- **Infrastructure** — Twelve Data, OANDA, CCXT, NewsAPI, Groq, Telegram, PostgreSQL, Dashboard

---

## ⚡ Installation rapide

### 1. Cloner et configurer

```bash
git clone https://github.com/ton-username/trading-bot.git
cd trading-bot

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
