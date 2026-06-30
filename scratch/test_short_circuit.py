import time
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.data_loader import fetch_extended_history
from src.strategy import generate_signals
from src.engine import run_backtest as current_run_backtest
from src.engine import calculate_metrics

# Short-circuit optimized run_backtest
def short_circuit_run_backtest(
    df_dict: dict, 
    initial_balance: float = 10000.0, 
    risk_per_trade: float = 100.0,
    fee_pct: float = 0.05,        
    slippage_pct: float = 0.02,   
    use_trailing_sl: bool = False,
    trailing_trigger_pct: float = 1.5, 
    trailing_distance_pct: float = 1.0, 
    max_concurrent_trades: int = 2
) -> dict:
    all_rows = []
    for symbol, df in df_dict.items():
        df_copy = df.copy()
        df_copy["symbol"] = symbol
        all_rows.append(df_copy)
        
    master_df = pd.concat(all_rows).sort_values("timestamp").reset_index(drop=True)
    
    balance = initial_balance
    active_trades = {} 
    trade_history = []
    
    # Pre-convert columns to standard Python lists
    timestamp_arr = master_df["timestamp"].tolist()
    symbol_arr = master_df["symbol"].tolist()
    close_arr = master_df["close"].tolist()
    high_arr = master_df["high"].tolist()
    low_arr = master_df["low"].tolist()
    signal_arr = master_df["signal"].astype(int).tolist()
    
    sl_level_arr = master_df["sl_level"].tolist() if "sl_level" in master_df.columns else [0.0] * len(master_df)
    tp_level_arr = master_df["tp_level"].tolist() if "tp_level" in master_df.columns else [0.0] * len(master_df)
    
    equity_curve = [initial_balance]
    equity_timestamps = [timestamp_arr[0]]
    
    trade_peaks = {} 
    latest_close = {}
    
    for i in range(len(master_df)):
        timestamp = timestamp_arr[i]
        symbol = symbol_arr[i]
        close = close_arr[i]
        signal = signal_arr[i]
        
        latest_close[symbol] = close
        
        # --- SHORT CIRCUIT FOR IDLE CANDLES ---
        # If there are no open trades and no new signal, the equity is just the cash balance.
        # We can skip all trade execution logic for this candle.
        if not active_trades and signal == 0:
            equity_curve.append(balance)
            equity_timestamps.append(timestamp)
            continue
            
        high = high_arr[i]
        low = low_arr[i]
        
        # --- 0. PRE-CHECK FOR PORTFOLIO LIQUIDATION ---
        current_equity = balance
        for sym, t in active_trades.items():
            sym_close = latest_close.get(sym, close)
            if t["type"] == "BUY":
                unrealized_pnl = (sym_close - t["entry_price"]) * t["position_size"]
            else:
                unrealized_pnl = (t["entry_price"] - sym_close) * t["position_size"]
            current_equity += unrealized_pnl
            
        if current_equity <= 0:
            if active_trades:
                for sym, t in list(active_trades.items()):
                    sym_close = latest_close.get(sym, close)
                    trade_history.append({
                        "symbol": sym,
                        "entry_time": t["entry_time"],
                        "exit_time": timestamp,
                        "type": "LONG (BUY)" if t["type"] == "BUY" else "SHORT (SELL)",
                        "entry_price": t["entry_price"],
                        "exit_price": sym_close,
                        "sl": t["sl"],
                        "tp": t["tp"],
                        "result": "LIQUIDATED",
                        "pnl": -balance / len(active_trades),
                        "balance": 0.0
                    })
                active_trades.clear()
                trade_peaks.clear()
            balance = 0.0
            current_equity = 0.0
            equity_curve.append(0.0)
            equity_timestamps.append(timestamp)
            continue
            
        # --- 1. CHECK AND EVALUATE ACTIVE TRADE FOR THIS SYMBOL ---
        if symbol in active_trades:
            trade = active_trades[symbol]
            
            if trade["type"] == "BUY":
                if high > trade_peaks[symbol]:
                    trade_peaks[symbol] = high
                    if use_trailing_sl:
                        profit_pct = ((high - trade["entry_price"]) / trade["entry_price"]) * 100.0
                        if profit_pct >= trailing_trigger_pct:
                            new_sl = high * (1 - trailing_distance_pct / 100.0)
                            if new_sl > trade["sl"]:
                                trade["sl"] = new_sl
            else:
                if low < trade_peaks[symbol]:
                    trade_peaks[symbol] = low
                    if use_trailing_sl:
                        profit_pct = ((trade["entry_price"] - low) / trade["entry_price"]) * 100.0
                        if profit_pct >= trailing_trigger_pct:
                            new_sl = low * (1 + trailing_distance_pct / 100.0)
                            if new_sl < trade["sl"]:
                                trade["sl"] = new_sl
                                
            trade_closed = False
            exit_price = None
            result = None
            
            if trade["type"] == "BUY":
                if low <= trade["sl"]:
                    trade_closed = True
                    exit_price = trade["sl"]
                    result = "LOSS"
                elif high >= trade["tp"]:
                    trade_closed = True
                    exit_price = trade["tp"]
                    result = "WIN"
            else:
                if high >= trade["sl"]:
                    trade_closed = True
                    exit_price = trade["sl"]
                    result = "LOSS"
                elif low <= trade["tp"]:
                    trade_closed = True
                    exit_price = trade["tp"]
                    result = "WIN"
                    
            if trade_closed:
                slippage_offset = exit_price * (slippage_pct / 100.0)
                adjusted_exit = (exit_price - slippage_offset) if trade["type"] == "BUY" else (exit_price + slippage_offset)
                
                if trade["type"] == "BUY":
                    gross_pnl = (adjusted_exit - trade["entry_price"]) * t["position_size"]
                else:
                    gross_pnl = (trade["entry_price"] - adjusted_exit) * t["position_size"]
                    
                exit_fee = (adjusted_exit * t["position_size"]) * (fee_pct / 100.0)
                net_pnl = gross_pnl - exit_fee
                
                balance += net_pnl
                
                if result == "LOSS" and net_pnl > 0:
                    result = "TSL WIN"
                    
                if balance < 0:
                    net_pnl -= balance
                    balance = 0.0
                    if result != "WIN" and result != "TSL WIN":
                        result = "LIQUIDATED"
                
                trade_history.append({
                    "symbol": symbol,
                    "entry_time": trade["entry_time"],
                    "exit_time": timestamp,
                    "type": "LONG (BUY)" if trade["type"] == "BUY" else "SHORT (SELL)",
                    "entry_price": trade["entry_price"],
                    "exit_price": adjusted_exit,
                    "sl": trade["sl"],
                    "tp": trade["tp"],
                    "result": result,
                    "pnl": net_pnl,
                    "balance": balance
                })
                
                del active_trades[symbol]
                del trade_peaks[symbol]
                
        # --- 2. SCAN FOR NEW ENTRY SIGNALS ---
        else:
            if signal != 0 and balance > 0 and len(active_trades) < max_concurrent_trades:
                sl = sl_level_arr[i]
                tp = tp_level_arr[i]
                
                risk_distance = (close - sl) if signal == 1 else (sl - close)
                
                if risk_distance > 0:
                    actual_risk = min(risk_per_trade, balance)
                    position_size = actual_risk / risk_distance
                    
                    slippage_offset = close * (slippage_pct / 100.0)
                    adjusted_entry = (close + slippage_offset) if signal == 1 else (close - slippage_offset)
                    
                    entry_fee = (adjusted_entry * position_size) * (fee_pct / 100.0)
                    
                    if entry_fee < balance:
                        balance -= entry_fee
                        
                        active_trades[symbol] = {
                            "type": "BUY" if signal == 1 else "SELL",
                            "entry_time": timestamp,
                            "entry_price": adjusted_entry,
                            "sl": sl,
                            "tp": tp,
                            "position_size": position_size
                        }
                        trade_peaks[symbol] = adjusted_entry
                    
        # --- 3. CALCULATE PORTFOLIO EQUITY CURVE ---
        current_equity = balance
        for sym, t in active_trades.items():
            sym_close = latest_close.get(sym, close)
            if t["type"] == "BUY":
                unrealized_pnl = (sym_close - t["entry_price"]) * t["position_size"]
            else:
                unrealized_pnl = (t["entry_price"] - sym_close) * t["position_size"]
            current_equity += unrealized_pnl
            
        if current_equity <= 0:
            if active_trades:
                for sym, t in list(active_trades.items()):
                    sym_close = latest_close.get(sym, close)
                    trade_history.append({
                        "symbol": sym,
                        "entry_time": t["entry_time"],
                        "exit_time": timestamp,
                        "type": "LONG (BUY)" if t["type"] == "BUY" else "SHORT (SELL)",
                        "entry_price": t["entry_price"],
                        "exit_price": sym_close,
                        "sl": t["sl"],
                        "tp": t["tp"],
                        "result": "LIQUIDATED",
                        "pnl": -balance / len(active_trades),
                        "balance": 0.0
                    })
                active_trades.clear()
                trade_peaks.clear()
            balance = 0.0
            current_equity = 0.0
            
        equity_curve.append(current_equity)
        equity_timestamps.append(timestamp)
        
    trades_df = pd.DataFrame(trade_history)
    equity_df = pd.DataFrame({"timestamp": equity_timestamps, "equity": equity_curve})
    metrics = calculate_metrics(trades_df, equity_df, initial_balance)
    
    return {
        "trades": trades_df,
        "equity_curve": equity_df,
        "metrics": metrics
    }

print("Loading test data...")
# Load a larger dataset to see the difference clearly (e.g. 180 days of data)
df = fetch_extended_history("BTCUSDT", "15m", 180)
df_signals = generate_signals(df, sltp_mode="structure", rr_ratio=2.0)
df_dict = {"BTCUSDT": df_signals}

print("Profiling current run_backtest...")
t0 = time.time()
for _ in range(20):
    res_orig = current_run_backtest(df_dict)
t_orig = time.time() - t0
print(f"Current: 20 iterations took {t_orig:.4f} seconds ({t_orig/20:.4f}s per run)")

print("Profiling short-circuit run_backtest...")
t0 = time.time()
for _ in range(20):
    res_opt = short_circuit_run_backtest(df_dict)
t_opt = time.time() - t0
print(f"Short-circuit: 20 iterations took {t_opt:.4f} seconds ({t_opt/20:.4f}s per run)")

print(f"Speedup: {t_orig / t_opt:.2f}x")

# Assert correctness
assert len(res_orig["trades"]) == len(res_opt["trades"])
if not res_orig["trades"].empty:
    assert np.allclose(res_orig["trades"]["pnl"].values, res_opt["trades"]["pnl"].values)
print("SUCCESS: Backtest results match perfectly!")
