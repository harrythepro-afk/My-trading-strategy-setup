import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.data_loader import fetch_extended_history
from src.strategy import generate_signals
from src.engine import run_backtest
from src.optimizer import run_grid_search

# Settings matching user's sidebar
initial_balance = 1000.0
risk_per_trade = 20.0
max_concurrent_trades = 10
fee_pct = 0.05
slippage_pct = 0.02
sweep_mode = "both"
use_trend_filter = False  # Enable 200 EMA Filter is OFF in user's sidebar
trade_direction = "both"
rolling_window = 96
trigger_modes = ["engulfing", "pinbar", "pullback", "immediate"] # All selected in sidebar
trigger_logic = "OR"
sltp_mode = "structure"

# Premium upgrades
use_fixed_session_levels = False
use_atr_penetration = True
atr_penetration_factor = 0.60 # Sweep Depth (ATR Multiple) is 0.60
use_atr_tp = False
atr_tp_factor = 2.5
use_time_filter = False
enable_breakout_reversal = True # Failed Triple Sweep Breakout Flip is ON
enable_quad_breakout = False
breakout_sl_pct = 3.0 # Breakout Stop Loss % is 3.00
breakout_only = False
custom_tp_dict = None

# Fetch data for the basket
symbols = ["ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"]
df_dict = {}
print("Loading data for basket...")
for sym in symbols:
    df_dict[sym] = fetch_extended_history(sym, "15m", 365)

# --- 1. Run Main Backtest for R:R = 2.0 ---
print("\nRunning main backtest with R:R = 2.0...")
df_signals_dict = {}
for sym, df in df_dict.items():
    df_signals_dict[sym] = generate_signals(
        df.copy(),
        sweep_mode=sweep_mode,
        use_trend_filter=use_trend_filter,
        trade_direction=trade_direction,
        sltp_mode=sltp_mode,
        rr_ratio=2.0,
        rolling_window=rolling_window,
        trigger_modes=trigger_modes,
        trigger_logic=trigger_logic,
        use_fixed_session_levels=use_fixed_session_levels,
        use_atr_penetration=use_atr_penetration,
        atr_penetration_factor=atr_penetration_factor,
        use_atr_tp=use_atr_tp,
        atr_tp_factor=atr_tp_factor,
        use_time_filter=use_time_filter,
        enable_breakout_reversal=enable_breakout_reversal,
        breakout_sl_pct=breakout_sl_pct,
        breakout_only=breakout_only,
        enable_quad_breakout=enable_quad_breakout,
        custom_tp_dict=custom_tp_dict
    )

res_backtest = run_backtest(
    df_signals_dict,
    initial_balance=initial_balance,
    risk_per_trade=risk_per_trade,
    fee_pct=fee_pct,
    slippage_pct=slippage_pct,
    use_trailing_sl=False,
    max_concurrent_trades=max_concurrent_trades
)
backtest_metrics = res_backtest["metrics"]
print(f"Main Backtest Results (R:R = 2.0):")
print(f"Total Trades: {backtest_metrics['total_trades']}")
print(f"Win Rate: {backtest_metrics['win_rate']}%")
print(f"Net Profit: ${backtest_metrics['net_profit']}")

# --- 2. Run Grid Search ---
print("\nRunning grid search optimizer...")
res_grid = run_grid_search(
    df_dict=df_dict,
    initial_balance=initial_balance,
    risk_per_trade=risk_per_trade,
    fee_pct=fee_pct,
    slippage_pct=slippage_pct,
    sweep_mode=sweep_mode,
    use_trend_filter=use_trend_filter,
    trade_direction=trade_direction,
    rolling_window=rolling_window,
    trigger_modes=trigger_modes,
    trigger_logic=trigger_logic,
    use_trailing_sl=False,
    max_concurrent_trades=max_concurrent_trades,
    use_fixed_session_levels=use_fixed_session_levels,
    use_atr_penetration=use_atr_penetration,
    atr_penetration_factor=atr_penetration_factor,
    use_atr_tp=use_atr_tp,
    atr_tp_factor=atr_tp_factor,
    use_time_filter=use_time_filter,
    enable_breakout_reversal=enable_breakout_reversal,
    breakout_sl_pct=breakout_sl_pct,
    breakout_only=breakout_only,
    enable_quad_breakout=enable_quad_breakout,
    custom_tp_dict=custom_tp_dict
)

# Extract row for R:R = 2.0x
row_rr2 = res_grid[(res_grid["Mode"] == "Wick Structure") & (res_grid["Details"] == "R:R Ratio = 2.0x")]
if not row_rr2.empty:
    print(f"\nOptimizer Grid Search Results (R:R = 2.0x):")
    print(f"Total Trades: {row_rr2.iloc[0]['Total Trades']}")
    print(f"Win Rate: {row_rr2.iloc[0]['Win Rate (%)']}%")
    print(f"Net Profit: ${row_rr2.iloc[0]['Net Profit ($)']}")
    
    # Check if they match
    trades_match = int(backtest_metrics['total_trades']) == int(row_rr2.iloc[0]['Total Trades'])
    profit_match = np.isclose(float(backtest_metrics['net_profit']), float(row_rr2.iloc[0]['Net Profit ($)']))
    if trades_match and profit_match:
        print("\nSUCCESS: Both results are 100% equivalent!")
    else:
        print("\nWARNING: Results differ!")
else:
    print("Could not find R:R = 2.0x row in optimizer results.")
