from src.data_loader import fetch_extended_history
from src.strategy import generate_signals
from src.engine import run_backtest

def main():
    print("==================================================")
    print("   Institutional Strategy Upgrades Test Pipeline  ")
    print("==================================================\n")
    
    symbol = "BTCUSDT"
    timeframe = "15m"
    days_to_test = 45
    
    print("Downloading BTCUSDT historical candles...")
    df = fetch_extended_history(symbol=symbol, interval=timeframe, days=days_to_test)
    if df.empty:
        print("Failed to download data.")
        return
        
    print(f"Downloaded {len(df)} candles.")
    
    # 1. Run Standard Retail Setup
    print("\n[Standard Retail Setup] Generating signals...")
    df_retail = generate_signals(
        df.copy(),
        use_fixed_session_levels=False,
        use_atr_penetration=False,
        use_time_filter=False,
        use_atr_tp=False
    )
    retail_results = run_backtest({symbol: df_retail}, initial_balance=10000.0, risk_per_trade=100.0)
    retail_metrics = retail_results["metrics"]
    
    # 2. Run Premium Institutional Setup
    print("\n[Premium Institutional Setup] Generating signals...")
    df_inst = generate_signals(
        df.copy(),
        use_fixed_session_levels=True,       # Sweeps PDH/PDL instead of rolling 24h
        use_atr_penetration=True,           # Volatility-adjusted sweeps
        atr_penetration_factor=0.25,        # min sweep depth 0.25 ATR
        use_time_filter=True,               # London / NY session hours
        use_atr_tp=True,                    # Volatility-based take profits
        atr_tp_factor=2.5                   # TP = Entry + 2.5 * ATR
    )
    inst_results = run_backtest({symbol: df_inst}, initial_balance=10000.0, risk_per_trade=100.0)
    inst_metrics = inst_results["metrics"]
    
    # 3. Compare Results
    print("\n==================================================")
    print("             STRATEGY PROFILE COMPARISON          ")
    print("==================================================")
    print(f"Asset Traded:      {symbol}")
    print(f"History Length:    {days_to_test} Days")
    print("\n--- Standard Retail Setup ---")
    print(f"Total Trades:      {retail_metrics['total_trades']}")
    print(f"Win Rate:          {retail_metrics['win_rate']}%")
    print(f"Net Profit ($):    ${retail_metrics['net_profit']}")
    print(f"ROI:               {retail_metrics['net_profit_pct']}%")
    print(f"Sharpe Ratio:      {retail_metrics['sharpe_ratio']}")
    
    print("\n--- Premium Institutional Setup ---")
    print(f"Total Trades:      {inst_metrics['total_trades']}")
    print(f"Win Rate:          {inst_metrics['win_rate']}%")
    print(f"Net Profit ($):    ${inst_metrics['net_profit']}")
    print(f"ROI:               {inst_metrics['net_profit_pct']}%")
    print(f"Sharpe Ratio:      {inst_metrics['sharpe_ratio']}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
