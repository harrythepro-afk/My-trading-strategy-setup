import pandas as pd
import numpy as np

def run_backtest(
    df_dict: dict, 
    initial_balance: float = 10000.0, 
    risk_per_trade: float = 100.0,
    fee_pct: float = 0.05,        # Default Binance Taker fee is 0.05%
    slippage_pct: float = 0.02,   # Default execution slippage buffer 0.02%
    use_trailing_sl: bool = False,
    trailing_trigger_pct: float = 1.5, # Starts trailing when price is 1.5% in profit
    trailing_distance_pct: float = 1.0, # Trails exactly 1.0% behind the highest high
    max_concurrent_trades: int = 2
) -> dict:
    """
    Institutional-grade Portfolio Backtester optimized with NumPy arrays for extreme speed.
    """
    # 1. Prepare and tag the data for each coin
    all_rows = []
    for symbol, df in df_dict.items():
        df_copy = df.copy()
        df_copy["symbol"] = symbol
        all_rows.append(df_copy)
        
    # Combine all candles into a single master timeline, sorted chronologically
    master_df = pd.concat(all_rows).sort_values("timestamp").reset_index(drop=True)
    
    balance = initial_balance
    active_trades = {} 
    trade_history = []
    
    equity_curve = [initial_balance]
    equity_timestamps = [master_df.iloc[0]["timestamp"]]
    
    trade_peaks = {} 
    
    # Convert columns to NumPy arrays for extreme speed!
    timestamp_arr = master_df["timestamp"].values
    symbol_arr = master_df["symbol"].values
    close_arr = master_df["close"].values
    high_arr = master_df["high"].values
    low_arr = master_df["low"].values
    signal_arr = master_df["signal"].values
    
    sl_level_arr = master_df["sl_level"].values if "sl_level" in master_df.columns else np.zeros(len(master_df))
    tp_level_arr = master_df["tp_level"].values if "tp_level" in master_df.columns else np.zeros(len(master_df))
    
    latest_close = {}
    
    for i in range(len(master_df)):
        timestamp = pd.Timestamp(timestamp_arr[i])
        symbol = symbol_arr[i]
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        signal = int(signal_arr[i])
        
        # Track the latest close price for each active symbol for O(1) lookups
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
            # Account is force-liquidated! Close all positions immediately.
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
                        "pnl": -balance / len(active_trades), # Loss is capped at remaining cash
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
            
            # Update the peak price reached during this trade to track trailing SL
            if trade["type"] == "BUY":
                if high > trade_peaks[symbol]:
                    trade_peaks[symbol] = high
                    if use_trailing_sl:
                        profit_pct = ((high - trade["entry_price"]) / trade["entry_price"]) * 100.0
                        if profit_pct >= trailing_trigger_pct:
                            new_sl = high * (1 - trailing_distance_pct / 100.0)
                            if new_sl > trade["sl"]:
                                trade["sl"] = new_sl
            else: # SELL (Short)
                if low < trade_peaks[symbol]:
                    trade_peaks[symbol] = low
                    if use_trailing_sl:
                        profit_pct = ((trade["entry_price"] - low) / trade["entry_price"]) * 100.0
                        if profit_pct >= trailing_trigger_pct:
                            new_sl = low * (1 + trailing_distance_pct / 100.0)
                            if new_sl < trade["sl"]:
                                trade["sl"] = new_sl
                                
            # Check Exits
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
            else: # SELL (Short)
                if high >= trade["sl"]:
                    trade_closed = True
                    exit_price = trade["sl"]
                    result = "LOSS"
                elif low <= trade["tp"]:
                    trade_closed = True
                    exit_price = trade["tp"]
                    result = "WIN"
                    
            if trade_closed:
                # Apply slippage and fees
                slippage_offset = exit_price * (slippage_pct / 100.0)
                adjusted_exit = (exit_price - slippage_offset) if trade["type"] == "BUY" else (exit_price + slippage_offset)
                
                if trade["type"] == "BUY":
                    gross_pnl = (adjusted_exit - trade["entry_price"]) * trade["position_size"]
                else:
                    gross_pnl = (trade["entry_price"] - adjusted_exit) * trade["position_size"]
                    
                exit_fee = (adjusted_exit * trade["position_size"]) * (fee_pct / 100.0)
                net_pnl = gross_pnl - exit_fee
                
                balance += net_pnl
                
                # Check for profitable trailing stop loss hit
                if result == "LOSS" and net_pnl > 0:
                    result = "TSL WIN"
                    
                # If balance is drawn below zero, realize absolute loss of whatever capital was left
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
                    # Dynamically scale dollar-risk per trade if account balance is smaller
                    actual_risk = min(risk_per_trade, balance)
                    position_size = actual_risk / risk_distance
                    
                    slippage_offset = close * (slippage_pct / 100.0)
                    adjusted_entry = (close + slippage_offset) if signal == 1 else (close - slippage_offset)
                    
                    entry_fee = (adjusted_entry * position_size) * (fee_pct / 100.0)
                    
                    # Ensure fees do not trigger immediate bankruptcy
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
            # Get latest close price for active symbol instantly from memory lookup
            sym_close = latest_close.get(sym, close)
            if t["type"] == "BUY":
                unrealized_pnl = (sym_close - t["entry_price"]) * t["position_size"]
            else:
                unrealized_pnl = (t["entry_price"] - sym_close) * t["position_size"]
            current_equity += unrealized_pnl
            
        if current_equity <= 0:
            # Wiped out at end of step! Force close
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

def calculate_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_balance: float) -> dict:
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "net_profit_pct": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df["result"].isin(["WIN", "TSL WIN"])])
    win_rate = (wins / total_trades) * 100
    
    net_profit = trades_df["pnl"].sum()
    net_profit_pct = (net_profit / initial_balance) * 100
    
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
    max_drawdown = equity_df["drawdown"].min()
    
    equity_df_temp = equity_df.copy()
    equity_df_temp.set_index("timestamp", inplace=True)
    daily_equity = equity_df_temp["equity"].resample("1D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
    else:
        sharpe = 0.0
        
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "net_profit": round(net_profit, 2),
        "net_profit_pct": round(net_profit_pct, 2),
        "max_drawdown": round(abs(max_drawdown), 2),
        "sharpe_ratio": round(sharpe, 2)
    }
