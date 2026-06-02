from src.data_loader import fetch_extended_history
from src.strategy import generate_signals
from src.engine import run_backtest
import pandas as pd

def evaluate_coin(symbol: str, days: int = 90) -> dict:
    """
    Downloads history and runs strategy + backtest for a specific coin.
    """
    df = fetch_extended_history(symbol=symbol, interval="15m", days=days)
    if df.empty:
        return None
        
    df_signals = generate_signals(df)
    results = run_backtest({symbol: df_signals}, initial_balance=10000.0, risk_per_trade=100.0)
    return results["metrics"]

def main():
    coins = ["XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]
    days_to_test = 90
    
    print("==================================================")
    # Corrected spelling in print statement
    print("     EVALUATING 24h DOUBLE-SWEEP SUCCESS RATE     ")
    print(f"      Data Source: Binance API (Past {days_to_test} Days)      ")
    print("==================================================\n")
    
    records = []
    
    for coin in coins:
        print(f"Analyzing {coin}...")
        metrics = evaluate_coin(coin, days_to_test)
        if metrics:
            records.append({
                "Coin": coin,
                "Total Trades": metrics["total_trades"],
                "Success Rate (Win %)": f"{metrics['win_rate']}%",
                "Net Profit ($)": f"${metrics['net_profit']}",
                "ROI": f"{metrics['net_profit_pct']}%",
                "Max Drawdown": f"{metrics['max_drawdown']}%",
                "Sharpe Ratio": metrics["sharpe_ratio"]
            })
            
    # Print beautiful summary table
    summary_df = pd.DataFrame(records)
    print("\n==========================================================================")
    print("                         STRATEGY SUMMARY REPORT                          ")
    print("==========================================================================")
    print(summary_df.to_string(index=False))
    print("==========================================================================\n")
    
if __name__ == "__main__":
    main()
