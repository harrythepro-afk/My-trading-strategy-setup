import time
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.data_loader import fetch_extended_history
from src.strategy import generate_signals, calculate_indicators
from src.engine import run_backtest
from src.optimizer import run_grid_search as original_run_grid_search

def optimized_run_grid_search(
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
    
    # Pre-calculate indicators once
    precalc_df_dict = {}
    for sym, df in df_dict.items():
        precalc_df_dict[sym] = calculate_indicators(df.copy(), rolling_window=rolling_window)
        
    # Pre-generate base structures once
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
        
    # Pre-generate base percentages once
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
        
    # 1. Optimize Wick Structure setups
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
            
        backtest_res = run_backtest(
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
        
    # 2. Optimize Fixed Percentage setups
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
                
            backtest_res = run_backtest(
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

# Load some test data
print("Loading data for test...")
df = fetch_extended_history("BTCUSDT", "15m", 30)
df_dict = {"BTCUSDT": df}

print("Running original grid search...")
t0 = time.time()
res_orig = original_run_grid_search(df_dict)
t_orig = time.time() - t0
print(f"Original finished in {t_orig:.4f} seconds.")

print("Running optimized grid search...")
t0 = time.time()
res_opt = optimized_run_grid_search(df_dict)
t_opt = time.time() - t0
print(f"Optimized finished in {t_opt:.4f} seconds.")

print(f"Speedup: {t_orig / t_opt:.2f}x")

# Check equivalence
print("Checking mathematical equivalence of outputs:")
if res_orig.equals(res_opt):
    print("SUCCESS: Results are 100% equivalent!")
else:
    print("WARNING: Results differ!")
    print("Original top 5:")
    print(res_orig.head())
    print("Optimized top 5:")
    print(res_opt.head())
