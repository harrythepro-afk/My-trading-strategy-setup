import pandas as pd
import numpy as np
from src.strategy import generate_signals

def main():
    print("Testing Double/Triple/Quadruple breakout flips and combined modes...")
    
    # Generate dummy price data showing multiple sweeps/failures
    timestamps = pd.date_range(start="2026-06-01", periods=200, freq="15min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0
    })
    
    # Force a support break setup
    # Initial support is set by rolling window of 96
    df.loc[100:105, "low"] = 95.0
    df.loc[101, "close"] = 96.0
    
    custom_tp_dict = {
        "double_rr": 1.5,
        "double_breakout_rr": 2.2,
        "triple_rr": 2.0,
        "triple_breakout_rr": 2.5,
        "quad_rr": 2.0,
        "quad_breakout_rr": 3.0
    }
    
    # Test combined mode
    df_res = generate_signals(
        df,
        sweep_mode="double_triple_quad",
        enable_double_breakout=True,
        enable_breakout_reversal=True,
        enable_quad_breakout=True,
        custom_tp_dict=custom_tp_dict
    )
    
    print("Signal dataframe generated successfully!")
    print(f"Total columns: {list(df_res.columns)}")
    print(f"Signal value counts:\n{df_res['signal'].value_counts()}")
    print("Test passed successfully!")

if __name__ == "__main__":
    main()
