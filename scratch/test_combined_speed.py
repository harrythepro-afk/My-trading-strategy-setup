import time
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.data_loader import fetch_extended_history
from src.strategy import generate_signals, calculate_indicators
from src.engine import run_backtest as original_run_backtest
from src.engine import calculate_metrics
from src.optimizer import run_grid_search as original_run_grid_search

def optimized_run_backtest(
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
    
    # Pre-convert columns to standard Python lists for much faster indexing in the pure Python loop!
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
        high = high_arr[i]
        low = low_arr[i]
        signal = signal_arr[i]
        
        latest_close[symbol] = close
        
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
                    gross_pnl = (adjusted_exit - trade["entry_price"]) * trade["position_size"]
                else:
                    gross_pnl = (trade["entry_price"] - adjusted_exit) * trade["position_size"]
                    
                exit_fee = (adjusted_exit * trade["position_size"]) * (fee_pct / 100.0)
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

def combined_optimized_run_grid_search(
    df_dict: dict,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 100.0,
    fee_pct: float = 0.05,
    slippage_pct: float = 0.02,
    sweep_mode: str = "single",
    use_trend_filter: bool = True,
    trade_direction: str = "both",
    rolling_window: int = 96,
    trigger_modes: list = ["engulfing"],
    trigger_logic: str = "OR"
) -> pd.DataFrame:
    results = []
    
    rr_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    sl_grid = [0.5, 1.0, 1.5, 2.0]
    tp_grid = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    precalc_df_dict = {}
    for sym, df in df_dict.items():
        precalc_df_dict[sym] = calculate_indicators(df.copy(), rolling_window=rolling_window)
        
    base_structure_dict = {}
    for sym, df in precalc_df_dict.items():
        base_structure_dict[sym] = generate_signals(
            df.copy(), 
            sweep_mode=sweep_mode, 
            use_trend_filter=use_trend_filter, 
            trade_direction=trade_direction,
            sltp_mode="structure",
            rr_ratio=1.0,
            rolling_window=rolling_window,
            trigger_modes=trigger_modes,
            trigger_logic=trigger_logic
        )
        
    base_pct_dict = {}
    for sym, df in precalc_df_dict.items():
        base_pct_dict[sym] = generate_signals(
            df.copy(), 
            sweep_mode=sweep_mode, 
            use_trend_filter=use_trend_filter, 
            trade_direction=trade_direction,
            sltp_mode="percentage",
            fixed_sl_pct=1.0,
            fixed_tp_pct=1.0,
            rolling_window=rolling_window,
            trigger_modes=trigger_modes,
            trigger_logic=trigger_logic
        )
        
    for rr in rr_grid:
        temp_dict = {}
        for sym, base_df in base_structure_dict.items():
            df_copy = base_df.copy()
            sig = df_copy["signal"].values
            close = df_copy["close"].values
            sl = df_copy["sl_level"].values
            tp = df_copy["tp_level"].values.copy()
            
            buy_mask = sig == 1
            sell_mask = sig == -1
            
            tp[buy_mask] = close[buy_mask] + rr * (close[buy_mask] - sl[buy_mask])
            tp[sell_mask] = close[sell_mask] - rr * (sl[sell_mask] - close[sell_mask])
            
            df_copy["tp_level"] = tp
            temp_dict[sym] = df_copy
            
        backtest_res = optimized_run_backtest(
            temp_dict, 
            initial_balance=initial_balance, 
            risk_per_trade=risk_per_trade,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct
        )
        
        metrics = backtest_res["metrics"]
        results.append({
            "Mode": "Wick Structure",
            "Details": f"R:R Ratio = {rr}x",
            "Total Trades": metrics["total_trades"],
            "Win Rate (%)": metrics["win_rate"],
            "Net Profit ($)": metrics["net_profit"],
            "Max Drawdown (%)": metrics["max_drawdown"],
            "Sharpe Ratio": metrics["sharpe_ratio"]
        })
        
    for sl in sl_grid:
        for tp in tp_grid:
            if tp < sl:
                continue
                
            temp_dict = {}
            for sym, base_df in base_pct_dict.items():
                df_copy = base_df.copy()
                sig = df_copy["signal"].values
                close = df_copy["close"].values
                sl_arr = df_copy["sl_level"].values.copy()
                tp_arr = df_copy["tp_level"].values.copy()
                
                buy_mask = sig == 1
                sell_mask = sig == -1
                
                sl_factor = sl / 100.0
                tp_factor = tp / 100.0
                
                sl_arr[buy_mask] = close[buy_mask] * (1.0 - sl_factor)
                tp_arr[buy_mask] = close[buy_mask] * (1.0 + tp_factor)
                
                sl_arr[sell_mask] = close[sell_mask] * (1.0 + sl_factor)
                tp_arr[sell_mask] = close[sell_mask] * (1.0 - tp_factor)
                
                df_copy["sl_level"] = sl_arr
                df_copy["tp_level"] = tp_arr
                temp_dict[sym] = df_copy
                
            backtest_res = optimized_run_backtest(
                temp_dict, 
                initial_balance=initial_balance, 
                risk_per_trade=risk_per_trade,
                fee_pct=fee_pct,
                slippage_pct=slippage_pct
            )
            
            metrics = backtest_res["metrics"]
            results.append({
                "Mode": "Fixed %",
                "Details": f"SL={sl}%, TP={tp}%",
                "Total Trades": metrics["total_trades"],
                "Win Rate (%)": metrics["win_rate"],
                "Net Profit ($)": metrics["net_profit"],
                "Max Drawdown (%)": metrics["max_drawdown"],
                "Sharpe Ratio": metrics["sharpe_ratio"]
            })
            
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=["Sharpe Ratio", "Net Profit ($)"], ascending=False).reset_index(drop=True)
    
    return res_df

print("Loading test data...")
df = fetch_extended_history("BTCUSDT", "15m", 30)
df_dict = {"BTCUSDT": df}

print("Running original grid search...")
t0 = time.time()
res_orig = original_run_grid_search(df_dict)
t_orig = time.time() - t0
print(f"Original finished in {t_orig:.4f} seconds.")

print("Running combined optimized grid search...")
t0 = time.time()
res_opt = combined_optimized_run_grid_search(df_dict)
t_opt = time.time() - t0
print(f"Optimized finished in {t_opt:.4f} seconds.")

print(f"Combined Speedup: {t_orig / t_opt:.2f}x")

# Check equivalence
print("Checking mathematical equivalence of outputs:")
if res_orig.equals(res_opt):
    print("SUCCESS: Results are 100% equivalent!")
else:
    print("WARNING: Results differ!")
