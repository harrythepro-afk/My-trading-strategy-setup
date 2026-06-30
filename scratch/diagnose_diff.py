import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.data_loader import fetch_extended_history
from src.strategy import generate_signals
from src.engine import run_backtest

# Load data
symbols = ["ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"]
df_dict = {}
for sym in symbols:
    df_dict[sym] = fetch_extended_history(sym, "15m", 365)

# Main backtest
df_signals_dict = {}
for sym, df in df_dict.items():
    df_signals_dict[sym] = generate_signals(
        df.copy(),
        sweep_mode="both",
        use_trend_filter=False,
        trade_direction="both",
        sltp_mode="structure",
        rr_ratio=2.0,
        rolling_window=96,
        trigger_modes=["engulfing", "pinbar", "pullback", "immediate"],
        trigger_logic="OR",
        use_fixed_session_levels=False,
        use_atr_penetration=True,
        atr_penetration_factor=0.60,
        use_atr_tp=False,
        use_time_filter=False,
        enable_breakout_reversal=True,
        breakout_sl_pct=3.0,
        breakout_only=False
    )

res_backtest = run_backtest(
    df_signals_dict,
    initial_balance=1000.0,
    risk_per_trade=20.0,
    fee_pct=0.05,
    slippage_pct=0.02,
    max_concurrent_trades=10
)
t_backtest = res_backtest["trades"]

# Optimizer signals pre-generated
precalc_df_dict = {}
from src.strategy import calculate_indicators
for sym, df in df_dict.items():
    precalc_df_dict[sym] = calculate_indicators(df.copy(), rolling_window=96)
    
base_df = generate_signals(
    precalc_df_dict["ETHUSDT"].copy(), 
    sweep_mode="both", use_trend_filter=False, trade_direction="both",
    sltp_mode="structure", rr_ratio=1.0, rolling_window=96,
    trigger_modes=["engulfing", "pinbar", "pullback", "immediate"], trigger_logic="OR",
    use_fixed_session_levels=False, use_atr_penetration=True, atr_penetration_factor=0.60,
    use_atr_tp=False, use_time_filter=False, enable_breakout_reversal=True, breakout_sl_pct=3.0, breakout_only=False
)
# Vectorized update
rr = 2.0
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

res_opt_single = run_backtest(
    {"ETHUSDT": df_copy},
    initial_balance=1000.0,
    risk_per_trade=20.0,
    fee_pct=0.05,
    slippage_pct=0.02,
    max_concurrent_trades=10
)
t_opt = res_opt_single["trades"]

# Main backtest for ETHUSDT only
res_backtest_eth = run_backtest(
    {"ETHUSDT": df_signals_dict["ETHUSDT"]},
    initial_balance=1000.0,
    risk_per_trade=20.0,
    fee_pct=0.05,
    slippage_pct=0.02,
    max_concurrent_trades=10
)
t_backtest_eth = res_backtest_eth["trades"]

print(f"ETHUSDT only - Main trades: {len(t_backtest_eth)}, Opt trades: {len(t_opt)}")

# Compare columns of the two dataframes for ETHUSDT
df1 = df_signals_dict["ETHUSDT"]
df2 = df_copy

# Find where signal, sl_level, or tp_level differ
diff_sig = df1[df1["signal"] != df2["signal"]]
diff_sl = df1[~np.isclose(df1["sl_level"], df2["sl_level"], equal_nan=True)]
diff_tp = df1[~np.isclose(df1["tp_level"], df2["tp_level"], equal_nan=True)]

print(f"Signal differences: {len(diff_sig)}")
print(f"SL differences: {len(diff_sl)}")
print(f"TP differences: {len(diff_tp)}")

if len(diff_tp) > 0:
    print("\nFirst 5 TP differences:")
    cols = ["timestamp", "close", "signal", "sl_level", "tp_level"]
    for idx in diff_tp.index[:5]:
        print(f"Row {idx}:")
        print("  Main:", df1.loc[idx, cols].to_dict())
        print("  Opt: ", df2.loc[idx, cols].to_dict())
