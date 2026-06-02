"""
Paper Trading Bot — No API keys needed!
Uses Binance public API for real-time candle data and simulates trades locally.
"""
import time
import pandas as pd
from datetime import datetime
from src.strategy import generate_signals
import requests
import json
import os

# ═══════════════════════════════════════════════════════════════════════
# PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class PaperTrader:
    """Simulates a futures trading account with virtual balance."""
    
    def __init__(self, initial_balance=5000.0, leverage=10, margin_type="ISOLATED"):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.margin_type = margin_type
        self.position = None  # Current open position
        self.trade_history = []  # All completed trades
        self.trade_count = 0
        
    def open_position(self, symbol, side, entry_price, size, sl_price, tp_price):
        """Open a new paper position."""
        if self.position is not None:
            return False  # Already in a position
        
        margin_used = (entry_price * size) / self.leverage
        if margin_used > self.balance:
            print(f"  Insufficient margin. Need ${margin_used:,.2f}, have ${self.balance:,.2f}")
            return False
        
        self.position = {
            "symbol": symbol,
            "side": side,  # "LONG" or "SHORT"
            "entry_price": entry_price,
            "size": size,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "margin_used": margin_used,
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_id": self.trade_count + 1
        }
        self.balance -= margin_used  # Lock margin
        self.trade_count += 1
        return True
    
    def check_position(self, current_high, current_low, current_close):
        """Check if SL or TP was hit on the current candle. Returns result or None."""
        if self.position is None:
            return None
        
        pos = self.position
        result = None
        exit_price = None
        exit_reason = None
        
        if pos["side"] == "LONG":
            # Check SL first (worst case)
            if current_low <= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "STOP LOSS"
            # Check TP
            elif current_high >= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "TAKE PROFIT"
                
        elif pos["side"] == "SHORT":
            if current_high >= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "STOP LOSS"
            elif current_low <= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "TAKE PROFIT"
        
        if exit_price is not None:
            result = self._close_position(exit_price, exit_reason)
        
        return result
    
    def _close_position(self, exit_price, reason):
        """Close the current position and record the trade."""
        pos = self.position
        
        if pos["side"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]
        
        # Leveraged PnL
        pnl_pct = (pnl / pos["margin_used"]) * 100
        
        # Return margin + PnL
        self.balance += pos["margin_used"] + pnl
        
        trade_record = {
            "id": pos["trade_id"],
            "symbol": pos["symbol"],
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "size": pos["size"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "result": "WIN" if pnl > 0 else "LOSS",
            "reason": reason,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "balance_after": round(self.balance, 2)
        }
        
        self.trade_history.append(trade_record)
        self.position = None
        return trade_record
    
    def get_unrealized_pnl(self, current_price):
        """Calculate unrealized PnL for open position."""
        if self.position is None:
            return 0.0
        pos = self.position
        if pos["side"] == "LONG":
            return (current_price - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - current_price) * pos["size"]
    
    def get_stats(self):
        """Get overall trading statistics."""
        if not self.trade_history:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "total_pnl": 0, "balance": self.balance,
                "return_pct": 0
            }
        
        wins = sum(1 for t in self.trade_history if t["result"] == "WIN")
        losses = len(self.trade_history) - wins
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        
        return {
            "total_trades": len(self.trade_history),
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / len(self.trade_history)) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "balance": round(self.balance, 2),
            "return_pct": round(((self.balance - self.initial_balance) / self.initial_balance) * 100, 2),
            "best_trade": round(max(t["pnl"] for t in self.trade_history), 2),
            "worst_trade": round(min(t["pnl"] for t in self.trade_history), 2)
        }
    
    def save_history(self, filepath="paper_trades.json"):
        """Save trade history to JSON file."""
        data = {
            "initial_balance": self.initial_balance,
            "current_balance": round(self.balance, 2),
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "stats": self.get_stats(),
            "trades": self.trade_history
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath


# ═══════════════════════════════════════════════════════════════════════
# LIVE DATA FETCHER (Public API — No Auth Required!)
# ═══════════════════════════════════════════════════════════════════════

def fetch_public_candles(symbol: str, interval: str = "15m", limit: int = 300) -> pd.DataFrame:
    """
    Fetch candles from Binance public API (no auth needed).
    Works from any country, no API keys required.
    """
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        klines = resp.json()
        
        data = []
        for k in klines:
            data.append([
                pd.to_datetime(k[0], unit="ms"),
                float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            ])
        
        return pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    except Exception as e:
        print(f"Error fetching candles: {e}")
        return pd.DataFrame()


def get_current_price(symbol: str) -> float:
    """Get the current mark price for a futures symbol."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        resp = requests.get(url, timeout=10)
        return float(resp.json()["price"])
    except:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# MAIN PAPER TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_paper_bot(
    symbol: str = "BTCUSDT",
    sweep_mode: str = "single",
    use_trend_filter: bool = True,
    trade_direction: str = "both",
    sltp_mode: str = "structure",
    rr_ratio: float = 2.0,
    fixed_sl_pct: float = 20.0,
    fixed_tp_pct: float = 2.0,
    trigger_modes: list = None,
    trigger_logic: str = "OR",
    risk_per_trade: float = 10.0,
    leverage: int = 10,
    margin_type: str = "ISOLATED",
    initial_balance: float = 5000.0,
    poll_interval: int = 30,
    log_callback=None
):
    """
    Paper trading bot using real Binance market data.
    No API keys needed — uses public endpoints.
    """
    if trigger_modes is None:
        trigger_modes = ["engulfing"]
    
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if log_callback:
            log_callback(line)
        print(line)
    
    # Initialize paper trader
    trader = PaperTrader(
        initial_balance=initial_balance,
        leverage=leverage,
        margin_type=margin_type
    )
    
    log("=" * 60)
    log(f"  PAPER TRADING BOT — {symbol}")
    log(f"  Balance: ${initial_balance:,.2f} | Leverage: {leverage}x {margin_type}")
    log(f"  Sweep: {sweep_mode} | Triggers: {', '.join(trigger_modes)} ({trigger_logic})")
    log(f"  EMA Filter: {'ON' if use_trend_filter else 'OFF'} | Direction: {trade_direction}")
    log(f"  Risk/Trade: ${risk_per_trade} | Poll: {poll_interval}s")
    log("=" * 60)
    log("")
    
    last_processed_candle = None
    
    while True:
        try:
            # 1. Fetch live candles (public API)
            df = fetch_public_candles(symbol, limit=300)
            if df.empty:
                log("Failed to fetch candle data. Retrying...")
                time.sleep(10)
                continue
            
            # 2. Run strategy
            df_signals = generate_signals(
                df,
                sweep_mode=sweep_mode,
                use_trend_filter=use_trend_filter,
                trade_direction=trade_direction,
                sltp_mode=sltp_mode,
                rr_ratio=rr_ratio,
                fixed_sl_pct=fixed_sl_pct,
                fixed_tp_pct=fixed_tp_pct,
                rolling_window=96,
                trigger_modes=trigger_modes,
                trigger_logic=trigger_logic
            )
            
            # 3. Check last closed candle
            latest = df_signals.iloc[-2]
            candle_time = latest["timestamp"]
            signal = latest["signal"]
            close_price = latest["close"]
            high_price = latest["high"]
            low_price = latest["low"]
            state = latest["state_log"]
            
            # Skip if already processed
            if candle_time == last_processed_candle:
                # Still check open position against current candle
                if trader.position:
                    current = df_signals.iloc[-1]
                    result = trader.check_position(current["high"], current["low"], current["close"])
                    if result:
                        emoji = "🟢" if result["result"] == "WIN" else "🔴"
                        log(f"{emoji} TRADE CLOSED: {result['reason']} | PnL: ${result['pnl']:+,.2f} ({result['pnl_pct']:+.1f}%) | Balance: ${result['balance_after']:,.2f}")
                        trader.save_history()
                time.sleep(poll_interval)
                continue
            
            last_processed_candle = candle_time
            
            # 4. Check existing position against this candle
            if trader.position:
                result = trader.check_position(high_price, low_price, close_price)
                if result:
                    emoji = "🟢" if result["result"] == "WIN" else "🔴"
                    log(f"{emoji} TRADE CLOSED: {result['reason']} | PnL: ${result['pnl']:+,.2f} ({result['pnl_pct']:+.1f}%) | Balance: ${result['balance_after']:,.2f}")
                    stats = trader.get_stats()
                    log(f"   Stats: {stats['total_trades']} trades | Win Rate: {stats['win_rate']}% | Total PnL: ${stats['total_pnl']:+,.2f}")
                    trader.save_history()
            
            # 5. Log current state
            pos_info = ""
            if trader.position:
                upnl = trader.get_unrealized_pnl(close_price)
                pos_info = f" | POS: {trader.position['side']} @ ${trader.position['entry_price']:,.2f} (uPnL: ${upnl:+,.2f})"
            
            log(f"Candle {candle_time} | ${close_price:,.2f} | {state}{pos_info}")
            
            # 6. Open new position if signal fires and we're flat
            if trader.position is None:
                if signal == 1:
                    sl = latest["sl_level"]
                    tp = latest["tp_level"]
                    risk_dist = close_price - sl
                    if risk_dist > 0:
                        position_size = risk_per_trade / risk_dist
                        success = trader.open_position(symbol, "LONG", close_price, position_size, sl, tp)
                        if success:
                            log(f"🟢 LONG OPENED! Entry: ${close_price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | Size: {position_size:.4f} | {leverage}x")
                            trader.save_history()
                
                elif signal == -1:
                    sl = latest["sl_level"]
                    tp = latest["tp_level"]
                    risk_dist = sl - close_price
                    if risk_dist > 0:
                        position_size = risk_per_trade / risk_dist
                        success = trader.open_position(symbol, "SHORT", close_price, position_size, sl, tp)
                        if success:
                            log(f"🔴 SHORT OPENED! Entry: ${close_price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | Size: {position_size:.4f} | {leverage}x")
                            trader.save_history()
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            log("")
            log("=" * 60)
            log("  BOT STOPPED")
            stats = trader.get_stats()
            if stats["total_trades"] > 0:
                log(f"  Total Trades: {stats['total_trades']}")
                log(f"  Win Rate: {stats['win_rate']}%")
                log(f"  Total PnL: ${stats['total_pnl']:+,.2f}")
                log(f"  Final Balance: ${stats['balance']:,.2f} ({stats['return_pct']:+.1f}%)")
                log(f"  Best Trade: ${stats['best_trade']:+,.2f}")
                log(f"  Worst Trade: ${stats['worst_trade']:+,.2f}")
            log(f"  Trade history saved to paper_trades.json")
            log("=" * 60)
            trader.save_history()
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_paper_bot(
        symbol="BTCUSDT",
        sweep_mode="single",
        use_trend_filter=True,
        trade_direction="both",
        trigger_modes=["engulfing"],
        trigger_logic="AND",
        risk_per_trade=10.0,
        leverage=10,
        margin_type="ISOLATED",
        initial_balance=5000.0,
        poll_interval=30
    )
