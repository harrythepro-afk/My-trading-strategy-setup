import pandas as pd
import numpy as np
from src.strategy import generate_signals
from src.engine import run_backtest

def run_grid_search(
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
    """
    Parametric optimizer updated to pass the trigger_modes list.
    """
    results = []
    
    rr_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    sl_grid = [0.5, 1.0, 1.5, 2.0]
    tp_grid = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # 1. Optimize Wick Structure setups
    for rr in rr_grid:
        temp_dict = {}
        for sym, df in df_dict.items():
            temp_dict[sym] = generate_signals(
                df, 
                sweep_mode=sweep_mode, 
                use_trend_filter=use_trend_filter, 
                trade_direction=trade_direction,
                sltp_mode="structure",
                rr_ratio=rr,
                rolling_window=rolling_window,
                trigger_modes=trigger_modes,
                trigger_logic=trigger_logic
            )
            
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
            for sym, df in df_dict.items():
                temp_dict[sym] = generate_signals(
                    df, 
                    sweep_mode=sweep_mode, 
                    use_trend_filter=use_trend_filter, 
                    trade_direction=trade_direction,
                    sltp_mode="percentage",
                    fixed_sl_pct=sl,
                    fixed_tp_pct=tp,
                    rolling_window=rolling_window,
                    trigger_modes=trigger_modes,
                    trigger_logic=trigger_logic
                )
                
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
