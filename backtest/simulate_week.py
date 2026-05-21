#!/usr/bin/env python3
"""
Simulate the current validated config on recent trading days.
Outputs a detailed HTML report with every trade.
"""
import sys, os, copy, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import pytz
from datetime import datetime
from pathlib import Path

from src.engine import TradingEngine, _build_strategy
from src.risk.manager import RiskManager
from src.data.fetcher import DataFetcher

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]


def simulate_day(fetcher, config, sym_overrides, sym, date_str):
    """Simulate one day for one symbol. Returns list of trade dicts."""
    sym_cfg = copy.deepcopy(config)
    ov = sym_overrides.get(sym, {})
    if "stop_loss_pct" in ov:
        sym_cfg.setdefault("risk", {})["stop_loss_pct"] = float(ov["stop_loss_pct"])
    if "reward_risk_ratio" in ov:
        sym_cfg.setdefault("risk", {})["reward_risk_ratio"] = float(ov["reward_risk_ratio"])
    if "volume_multiplier" in ov:
        sym_cfg.setdefault("strategy", {})["volume_multiplier"] = float(ov["volume_multiplier"])
    if "strategy" in ov:
        sym_cfg["active_strategy"] = ov["strategy"]
        sym_cfg.setdefault("strategy", {})["active_strategy"] = ov["strategy"]

    strat_name = sym_cfg.get("active_strategy", "orb")
    strat_params = sym_cfg.get("strategy_params", {}).get(strat_name, {})
    for key in ["afternoon_entries", "entry_window_minutes", "afternoon_entry_enabled"]:
        if key in strat_params:
            sym_cfg[key] = strat_params[key]

    strat = _build_strategy(sym_cfg)
    rm = RiskManager(sym_cfg)
    balance = float(sym_cfg.get("account", {}).get("balance", 500))

    try:
        bars = fetcher.fetch_historical(sym, date_str, date_str, filter_windows=False)
    except Exception:
        return []
    if bars is None or bars.empty:
        return []

    trades = []
    position = None
    trade_count = 0

    for i in range(len(bars)):
        row = bars.iloc[i]
        bar_price = float(row["close"])
        bar_time = row["timestamp"].astimezone(_ET)
        bar_vol = float(row["volume"])

        if position is None and trade_count < 100:
            signal = strat.generate_signal(bars, i)
            if signal:
                signal["symbol"] = sym
                bp = math.floor(balance * 0.95 * 100) / 100
                pos = rm.calculate_position(signal, balance, buying_power=bp, bars_df=bars.iloc[:i+1])
                if pos:
                    position = pos
                    position["entry_time"] = bar_time
                    position["entry_bar"] = i
                    trade_count += 1

        elif position is not None:
            exit_info = rm.check_exit(position, bar_price)
            if exit_info and exit_info.get("exit"):
                pnl = (bar_price - position["entry_price"]) * position["shares"]
                duration = (i - position["entry_bar"])
                trades.append({
                    "symbol": sym, "strategy": strat_name, "date": date_str,
                    "entry_time": position["entry_time"].strftime("%H:%M"),
                    "exit_time": bar_time.strftime("%H:%M"),
                    "shares": position["shares"],
                    "entry_price": position["entry_price"],
                    "exit_price": bar_price,
                    "stop_loss": position["stop_loss"],
                    "target": position["target"],
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / (position["entry_price"] * position["shares"]) * 100, 2),
                    "reason": exit_info.get("reason", "unknown"),
                    "duration_bars": duration,
                    "sl_pct": ov.get("stop_loss_pct", "?"),
                    "rr": ov.get("reward_risk_ratio", "?"),
                })
                balance += pnl
                position = None

    # EOD close
    if position is not None:
        last_price = float(bars.iloc[-1]["close"])
        pnl = (last_price - position["entry_price"]) * position["shares"]
        duration = len(bars) - 1 - position["entry_bar"]
        trades.append({
            "symbol": sym, "strategy": strat_name, "date": date_str,
            "entry_time": position["entry_time"].strftime("%H:%M"),
            "exit_time": bars.iloc[-1]["timestamp"].astimezone(_ET).strftime("%H:%M"),
            "shares": position["shares"],
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "stop_loss": position["stop_loss"],
            "target": position["target"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (position["entry_price"] * position["shares"]) * 100, 2),
            "reason": "eod_exit",
            "duration_bars": duration,
            "sl_pct": ov.get("stop_loss_pct", "?"),
            "rr": ov.get("reward_risk_ratio", "?"),
        })

    return trades


def generate_html(all_trades, dates, config):
    """Generate a beautiful HTML report."""
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

    # Per-symbol stats
    sym_stats = {}
    for t in all_trades:
        s = t["symbol"]
        if s not in sym_stats:
            sym_stats[s] = {"trades": 0, "wins": 0, "pnl": 0, "strategy": t["strategy"]}
        sym_stats[s]["trades"] += 1
        if t["pnl"] > 0:
            sym_stats[s]["wins"] += 1
        sym_stats[s]["pnl"] += t["pnl"]

    # Per-day stats
    day_stats = {}
    for t in all_trades:
        d = t["date"]
        if d not in day_stats:
            day_stats[d] = {"trades": 0, "wins": 0, "pnl": 0}
        day_stats[d]["trades"] += 1
        if t["pnl"] > 0:
            day_stats[d]["wins"] += 1
        day_stats[d]["pnl"] += t["pnl"]

    trade_rows = ""
    for t in all_trades:
        color = "#00ff88" if t["pnl"] > 0 else "#ff3131"
        reason_badge = {
            "stop_loss": '<span style="color:#ff3131">STOP</span>',
            "target_hit": '<span style="color:#00ff88">TARGET</span>',
            "eod_exit": '<span style="color:#ffaa00">EOD</span>',
        }.get(t["reason"], t["reason"])

        trade_rows += f"""
        <tr>
            <td>{t['date']}</td>
            <td><strong>{t['symbol']}</strong></td>
            <td>{t['strategy']}</td>
            <td>{t['entry_time']}</td>
            <td>{t['exit_time']}</td>
            <td>${t['entry_price']:.2f}</td>
            <td>${t['exit_price']:.2f}</td>
            <td>${t['stop_loss']:.2f}</td>
            <td>${t['target']:.2f}</td>
            <td>{t['shares']}</td>
            <td style="color:{color};font-weight:700">${t['pnl']:+.2f}</td>
            <td style="color:{color}">{t['pnl_pct']:+.1f}%</td>
            <td>{reason_badge}</td>
            <td>{t['duration_bars']}m</td>
        </tr>"""

    sym_rows = ""
    for s, st in sorted(sym_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr_s = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        color = "#00ff88" if st["pnl"] > 0 else "#ff3131"
        sym_rows += f"""
        <tr>
            <td><strong>{s}</strong></td>
            <td>{st['strategy']}</td>
            <td>{st['trades']}</td>
            <td>{st['wins']}</td>
            <td>{wr_s:.0f}%</td>
            <td style="color:{color};font-weight:700">${st['pnl']:+.2f}</td>
        </tr>"""

    day_rows = ""
    cumulative = 0
    for d in sorted(day_stats.keys()):
        ds = day_stats[d]
        cumulative += ds["pnl"]
        color = "#00ff88" if ds["pnl"] > 0 else "#ff3131"
        day_rows += f"""
        <tr>
            <td>{d}</td>
            <td>{ds['trades']}</td>
            <td>{ds['wins']}/{ds['trades'] - ds['wins']}</td>
            <td style="color:{color};font-weight:700">${ds['pnl']:+.2f}</td>
            <td>${cumulative:+.2f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Simulation Report</title>
<style>
  :root {{ --bg:#0e1117; --card:#1a1f2e; --green:#00ff88; --red:#ff3131; --amber:#ffaa00; --text:#e0e8f0; --muted:#7888a0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; margin:0; padding:2rem; }}
  .hero {{ background:linear-gradient(135deg,#0d1535,#1a0f2e 50%,#0e1117); border:1px solid var(--green); padding:2.5rem; text-align:center; margin-bottom:2rem; border-radius:8px; }}
  .hero h1 {{ color:var(--green); font-weight:700; letter-spacing:3px; margin:0; }}
  .hero p {{ color:var(--muted); margin:0.5rem 0 0; }}
  .card {{ background:var(--card); border:1px solid #1e2d3d; border-radius:8px; padding:1.5rem; margin-bottom:1.5rem; }}
  .stat-row {{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem; }}
  .stat {{ background:var(--card); border:1px solid #1e2d3d; border-radius:8px; padding:1.2rem; flex:1; min-width:120px; text-align:center; }}
  .stat-value {{ font-size:1.8rem; font-weight:700; }}
  .stat-label {{ font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-top:0.3rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  th {{ color:var(--muted); font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:0.6rem; text-align:left; border-bottom:1px solid var(--green); }}
  td {{ padding:0.5rem 0.6rem; border-bottom:1px solid #1e2d3d; color:var(--text); }}
  tr:hover {{ background:rgba(0,255,136,0.03); }}
  h2 {{ color:var(--green); font-size:1.1rem; letter-spacing:2px; margin:0 0 1rem; }}
  .green {{ color:var(--green); }} .red {{ color:var(--red); }} .amber {{ color:var(--amber); }}
</style>
</head>
<body>

<div class="hero">
  <h1>TRADING SIMULATION</h1>
  <p>Validated Config &bull; {dates[0]} to {dates[-1]} &bull; {len(all_trades)} trades</p>
</div>

<div class="stat-row">
  <div class="stat"><div class="stat-value green">{len(all_trades)}</div><div class="stat-label">Total Trades</div></div>
  <div class="stat"><div class="stat-value {'green' if total_pnl > 0 else 'red'}">${total_pnl:+.2f}</div><div class="stat-label">Total P&L</div></div>
  <div class="stat"><div class="stat-value {'green' if wr > 50 else 'amber'}">{wr:.0f}%</div><div class="stat-label">Win Rate</div></div>
  <div class="stat"><div class="stat-value green">${avg_win:+.2f}</div><div class="stat-label">Avg Win</div></div>
  <div class="stat"><div class="stat-value red">${avg_loss:+.2f}</div><div class="stat-label">Avg Loss</div></div>
  <div class="stat"><div class="stat-value">{len(wins)}/{len(losses)}</div><div class="stat-label">Wins/Losses</div></div>
</div>

<div class="card">
  <h2>BY SYMBOL</h2>
  <table>
    <thead><tr><th>Symbol</th><th>Strategy</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>P&L</th></tr></thead>
    <tbody>{sym_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>BY DAY</h2>
  <table>
    <thead><tr><th>Date</th><th>Trades</th><th>W/L</th><th>Day P&L</th><th>Cumulative</th></tr></thead>
    <tbody>{day_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>ALL TRADES</h2>
  <table>
    <thead><tr><th>Date</th><th>Symbol</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>Stop</th><th>Target</th><th>Shares</th><th>P&L</th><th>%</th><th>Reason</th><th>Duration</th></tr></thead>
    <tbody>{trade_rows}</tbody>
  </table>
</div>

<div style="text-align:center;color:var(--muted);font-size:0.65rem;padding:2rem 0;border-top:1px solid #1e2d3d">
  Generated by Day Trading Bot Simulator &bull; {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</body>
</html>"""
    return html


if __name__ == "__main__":
    with open(_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    fetcher = DataFetcher()
    sym_overrides = config.get("symbol_overrides", {})
    symbols = config.get("symbols", [])

    dates = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"]

    print(f"Simulating {len(symbols)} symbols x {len(dates)} days...")
    all_trades = []

    for date_str in dates:
        for sym in symbols:
            trades = simulate_day(fetcher, config, sym_overrides, sym, date_str)
            all_trades.extend(trades)
            if trades:
                for t in trades:
                    print(f"  {t['date']} {t['symbol']:<6} {t['strategy']:<16} "
                          f"entry={t['entry_time']} exit={t['exit_time']} "
                          f"${t['entry_price']:.2f}→${t['exit_price']:.2f} "
                          f"pnl=${t['pnl']:+.2f} {t['reason']}")

    print(f"\nTotal: {len(all_trades)} trades, P&L=${sum(t['pnl'] for t in all_trades):+.2f}")

    html = generate_html(all_trades, dates, config)
    out_path = _ROOT / "logs" / "research" / "simulation_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved to {out_path}")
