import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)

_TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"

_REASON_LABEL = {
    "stop_loss":  "Stop Loss Hit",
    "target_hit": "Target Hit",
    "eod_exit":   "End-of-Day Exit",
}

_MODE_TAG = {
    "backtest": "BACKTEST",
    "paper":    "PAPER",
    "live":     "LIVE",
}


class Notifier:
    """Sends trade notifications via Telegram Bot API.

    All methods silently no-op when:
    - notifications.telegram.enabled is False in config, OR
    - TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set in .env

    API failures never propagate — they are logged locally and swallowed.
    """

    def __init__(self, config: Dict):
        tg_cfg = config.get("notifications", {}).get("telegram", {})
        self.enabled  = tg_cfg.get("enabled", False)
        self.token    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id  = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.mode_tag = _MODE_TAG.get(config.get("mode", "backtest"), "BACKTEST")

        if self.enabled and (not self.token or not self.chat_id):
            logger.warning(
                "Telegram enabled in config but BOT_TOKEN/CHAT_ID not set — "
                "notifications will be skipped"
            )
            self.enabled = False

    # ------------------------------------------------------------------
    # Public notification methods
    # ------------------------------------------------------------------

    def notify_trade_opened(self, position: Dict, mode: Optional[str] = None) -> None:
        tag = _MODE_TAG.get(mode, self.mode_tag) if mode else self.mode_tag
        entry_dt = position.get("entry_time")
        time_str = entry_dt.strftime("%H:%M ET") if entry_dt else "—"
        date_str = entry_dt.strftime("%Y-%m-%d")  if entry_dt else "—"

        msg = (
            f"🟢 <b>[{tag}] TRADE OPENED</b>\n"
            f"Symbol: <b>{position['symbol']}</b>\n"
            f"Side: BUY\n"
            f"Shares: {position['shares']}\n"
            f"Entry: ${position['entry_price']:.2f}\n"
            f"Stop-Loss: ${position['stop_loss']:.2f}\n"
            f"Target: ${position['target']:.2f}\n"
            f"Risk: ${position['risk_amount']:.2f} | "
            f"Reward: ${position['reward']:.2f}\n"
            f"Time: {time_str} | {date_str}"
        )
        self._send(msg)

    def notify_trade_closed(self, trade_record: Dict, account_balance: float,
                            mode: Optional[str] = None) -> None:
        tag    = _MODE_TAG.get(mode, self.mode_tag) if mode else self.mode_tag
        pnl    = trade_record["pnl"]
        icon   = "✅" if pnl >= 0 else "❌"
        reason = _REASON_LABEL.get(trade_record["reason"], trade_record["reason"])
        pnl_sign = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        msg = (
            f"🔴 <b>[{tag}] TRADE CLOSED</b>\n"
            f"Symbol: <b>{trade_record['symbol']}</b>\n"
            f"Exit: ${trade_record['exit_price']:.2f}\n"
            f"P&amp;L: {pnl_sign} {icon}\n"
            f"Reason: {reason}\n"
            f"Duration: {trade_record['duration_minutes']} min\n"
            f"Account: ${account_balance:.2f}\n"
            f"Daily P&amp;L: {pnl_sign}"
        )
        self._send(msg)

    def notify_daily_summary(self, summary: Dict, account_balance: float,
                             mode: Optional[str] = None) -> None:
        tag      = _MODE_TAG.get(mode, self.mode_tag) if mode else self.mode_tag
        pnl      = summary["total_pnl"]
        pnl_str  = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        msg = (
            f"📊 <b>[{tag}] DAILY SUMMARY</b>\n"
            f"Date: {summary['date']}\n"
            f"Trades: {summary['total_trades']} | "
            f"Wins: {summary['wins']} | "
            f"Losses: {summary['losses']}\n"
            f"Win Rate: {summary['win_rate']:.0f}%\n"
            f"Daily P&amp;L: {pnl_str}\n"
            f"Account Balance: ${account_balance:.2f}"
        )
        self._send(msg)

    def notify_halt(self, reason: str, daily_pnl: float,
                    mode: Optional[str] = None) -> None:
        tag     = _MODE_TAG.get(mode, self.mode_tag) if mode else self.mode_tag
        pnl_str = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"

        msg = (
            f"⚠️ <b>[{tag}] TRADING HALTED</b>\n"
            f"Reason: {reason}\n"
            f"Daily P&amp;L: {pnl_str}\n"
            f"Bot paused for today."
        )
        self._send(msg)

    def notify_heartbeat(self, date: str, balance: float,
                         mode: Optional[str] = None) -> None:
        tag = _MODE_TAG.get(mode, self.mode_tag) if mode else self.mode_tag

        msg = (
            f"💓 <b>[{tag}] BOT HEARTBEAT</b>\n"
            f"Simulating: {date}\n"
            f"Balance: ${balance:.2f}\n"
            f"Status: Running ✅"
        )
        self._send(msg)

    def notify_error(self, error: str, mode: Optional[str] = None) -> None:
        tag = _MODE_TAG.get(mode, self.mode_tag) if mode else self.mode_tag

        msg = (
            f"❌ <b>[{tag}] ERROR</b>\n"
            f"{error}"
        )
        self._send(msg)

    # ------------------------------------------------------------------
    # Internal send with retry
    # ------------------------------------------------------------------

    def _send(self, text: str) -> None:
        """POST a message to the Telegram Bot API.

        One automatic retry after 1 second on failure.
        All exceptions are caught — notification failures never crash the engine.
        """
        if not self.enabled:
            return

        url     = _TELEGRAM_URL.format(token=self.token)
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}

        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code == 200:
                    return
                logger.warning(
                    "Telegram API returned %d on attempt %d: %s",
                    resp.status_code, attempt + 1, resp.text[:200],
                )
            except Exception as exc:
                logger.warning("Telegram send failed (attempt %d): %s", attempt + 1, exc)

            if attempt == 0:
                time.sleep(1)

        logger.error("Telegram notification dropped after 2 attempts.")
