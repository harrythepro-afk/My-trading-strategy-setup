from src.data_loader import fetch_extended_history
from src.strategy import generate_signals
from src.engine import run_backtest

def main():
    print("==================================================")
    print("      Crypto 24h Low Double-Sweep Test Pipeline   ")
    print("==================================================\n")
    
    # 1. Fetch data: let's download 45 days of BTCUSDT 15m data to have a solid sample size
    symbol = "BTCUSDT"
    timeframe = "15m"
    days_to_test = 45
    
    df = fetch_extended_history(symbol=symbol, interval=timeframe, days=days_to_test)
    
    if df.empty:
        print("Failed to download data. Exiting.")
        return
        
    # 2. Run Strategy (Calculates EMA, 24h Low, sweeps, and triggers)
    print("\nRunning advanced state-machine strategy...")
    df_with_signals = generate_signals(df)
    
    total_signals = df_with_signals["signal"].sum()
    print(f"Signals detected: {total_signals} setups.")
    
    # 3. Run Backtest Engine (Virtual simulation starting with $10,000 risking $100 per trade)
    print("Simulating trades with risk management...")
    results = run_backtest({symbol: df_with_signals}, initial_balance=10000.0, risk_per_trade=100.0)
    
    metrics = results["metrics"]
    trades = results["trades"]
    
    # 4. Print beautiful metrics report
    print("\n==================================================")
    print("              BACKTEST PERFORMANCE REPORT         ")
    print("==================================================")
    print(f"Asset Traded:      {symbol} ({timeframe})")
    print(f"History Tested:    {days_to_test} Days")
    print(f"Total Trades:      {metrics['total_trades']}")
    print(f"Win Rate:          {metrics['win_rate']}%")
    print(f"Net Profit ($):    ${metrics['net_profit']}")
    print(f"Net Profit (%):    {metrics['net_profit_pct']}%")
    print(f"Max Drawdown:      {metrics['max_drawdown']}%")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']}")
    print("==================================================\n")
    
    # Print trade samples
    if not trades.empty:
        print("First 5 Completed Trades:")
        cols_to_print = ["entry_time", "exit_time", "entry_price", "exit_price", "result", "pnl", "balance"]
        # Limit columns and print top rows
        print(trades[cols_to_print].head(5).to_string(index=False))
    else:
        print("No completed trades found. The strategy may have detected setups but they did not hit SL or TP yet.")

if __name__ == "__main__":
    main()
