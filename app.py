import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import our backend files!
from src.data_loader import fetch_extended_history, fetch_all_usdt_symbols, fetch_nse_history
from src.strategy import generate_signals
from src.engine import run_backtest
from src.optimizer import run_grid_search
from src.live_bot import connect_binance, fetch_live_candles, place_order, get_open_positions, get_symbol_precision, cancel_open_orders, configure_futures, get_account_info
from src.paper_trader import PaperTrader, fetch_public_candles, get_current_price
import threading
import time
import json

# Set modern dashboard configurations
st.set_page_config(
    page_title="Multi-Market Sweep & Optimization Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching the crypto symbols list so we only fetch it ONCE from Binance.
def load_all_crypto_symbols():
    return fetch_all_usdt_symbols()

# Custom premium styling using CSS
st.markdown("""
    <style>
        /* Base page styling */
        .reportview-container {
            background: #0f111a;
        }
        /* Custom metric card styling */
        .metric-card {
            background-color: #1a1c24;
            border: 1px solid #2e303c;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: #4f46e5;
        }
        .metric-title {
            color: #9ca3af;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #ffffff;
            font-size: 1.875rem;
            font-weight: 700;
        }
        .text-green { color: #10b981 !important; }
        .text-red { color: #ef4444 !important; }
        .text-indigo { color: #6366f1 !important; }

        /* ===== LANDING PAGE PREMIUM STYLES ===== */
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.15); }
            50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.35); }
        }
        @keyframes float-up {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .hero-banner {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
            border-radius: 16px;
            padding: 45px 35px;
            text-align: center;
            margin-bottom: 30px;
            border: 1px solid rgba(99, 102, 241, 0.25);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 30% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa, #6366f1, #818cf8, #c084fc);
            background-size: 200% auto;
            animation: shimmer 4s linear infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 700px;
            margin: 0 auto 25px auto;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #a5b4fc;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 18px;
        }
        .feature-card {
            background: linear-gradient(145deg, #1a1c2e, #14162a);
            border: 1px solid #2a2d42;
            border-radius: 14px;
            padding: 28px 22px;
            text-align: center;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.6s ease forwards;
            min-height: 210px;
        }
        .feature-card:hover {
            transform: translateY(-8px);
            border-color: #6366f1;
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
        }
        .feature-icon {
            font-size: 2.2rem;
            margin-bottom: 12px;
            display: block;
            animation: float-up 3s ease-in-out infinite;
        }
        .feature-title {
            color: #e2e8f0;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .feature-desc {
            color: #8891a5;
            font-size: 0.85rem;
            line-height: 1.55;
        }
        .flow-step {
            background: linear-gradient(145deg, #1e2035, #181a2e);
            border: 1px solid #2e3150;
            border-radius: 12px;
            padding: 20px 18px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
        }
        .flow-step:hover {
            border-color: #6366f1;
            transform: scale(1.03);
        }
        .flow-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #ffffff;
            font-weight: 800;
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        .flow-label {
            color: #e2e8f0;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .flow-detail {
            color: #8891a5;
            font-size: 0.78rem;
            line-height: 1.5;
        }
        .flow-arrow {
            color: #6366f1;
            font-size: 1.6rem;
            text-align: center;
            padding-top: 30px;
            animation: pulse-glow 2s ease infinite;
        }
        .config-preview {
            background: linear-gradient(145deg, #141627, #1a1d34);
            border: 1px solid #2e3150;
            border-radius: 14px;
            padding: 24px;
            animation: pulse-glow 3s ease-in-out infinite;
        }
        .config-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid rgba(99, 102, 241, 0.08);
        }
        .config-row:last-child { border-bottom: none; }
        .config-label {
            color: #8891a5;
            font-size: 0.85rem;
        }
        .config-value {
            color: #a5b4fc;
            font-size: 0.85rem;
            font-weight: 700;
            background: rgba(99, 102, 241, 0.12);
            padding: 3px 10px;
            border-radius: 6px;
        }
        .config-value-green {
            color: #34d399;
            font-size: 0.85rem;
            font-weight: 700;
            background: rgba(16, 185, 129, 0.12);
            padding: 3px 10px;
            border-radius: 6px;
        }
        .section-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #2e3150, transparent);
            margin: 35px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.header("🛠️ Market Configuration")

# Market Category selector
market_choice = st.sidebar.radio(
    "Select Market Category",
    options=["Crypto (Binance API)", "Indian Stocks (NSE Yahoo API)"],
    index=0,
    help="Select whether you want to backtest global cryptocurrencies or Indian equities."
)

symbols_selected = []
rolling_window = 96 # Default

if "Crypto" in market_choice:
    with st.spinner("Fetching active crypto symbols..."):
        crypto_symbols = load_all_crypto_symbols()
        
    symbols_selected = st.sidebar.multiselect(
        "Crypto Symbol Basket",
        options=crypto_symbols,
        default=["BTCUSDT"],
        help="Select one or multiple crypto pairs to build a diversified portfolio!"
    )
    rolling_window = 96
    
else: # Indian Stocks
    nse_options = [
        "^NSEI (Nifty 50 Index)",
        "RELIANCE.NS (Reliance)",
        "TCS.NS (TCS)",
        "HDFCBANK.NS (HDFC Bank)",
        "INFY.NS (Infosys)",
        "ICICIBANK.NS (ICICI Bank)",
        "SBIN.NS (State Bank of India)",
        "BHARTIARTL.NS (Airtel)",
        "LTIM.NS (LTIMindtree)"
    ]
    
    symbols_selected = st.sidebar.multiselect(
        "Indian Stock Basket",
        options=nse_options,
        default=["RELIANCE.NS (Reliance)"],
        help="Select top liquid NSE stocks. You can also type custom NSE symbols ending with '.NS'!"
    )
    rolling_window = 25

# Clean symbol strings to pull from APIs
symbols_clean = [s.split(" ")[0] for s in symbols_selected]

# Select history length
days = st.sidebar.slider(
    "Test History Length (Days)", 
    min_value=10, 
    max_value=365 if "Crypto" in market_choice else 59, 
    value=90 if "Crypto" in market_choice else 45, 
    step=5,
    help="Note: Crypto supports up to 365 days (1 Year) of history. Indian Stocks (NSE Yahoo API) are restricted by Yahoo Finance to a hard limit of the last 59 days for 15m intraday candles."
)
interval = "15m"

st.sidebar.subheader("💰 Capital & Risk")
initial_balance = st.sidebar.number_input("Starting Cash ($)", min_value=10.0, max_value=1000000.0, value=1000.0, step=50.0)
risk_per_trade = st.sidebar.number_input("Risk Per Trade ($)", min_value=1.0, max_value=5000.0, value=10.0, step=1.0)
max_concurrent_trades = st.sidebar.number_input("Max Concurrent Trades", min_value=1, max_value=10, value=2, step=1, help="Limits the maximum number of open trades at the same time across all coins in the backtester.")

# Exchange Fees & Execution Slippage
st.sidebar.subheader("🔌 Exchange Fees & Slippage")
fee_pct = st.sidebar.slider(
    "Exchange Taker Fee %",
    min_value=0.0,
    max_value=0.2,
    value=0.05 if "Crypto" in market_choice else 0.03,
    step=0.01
)
slippage_pct = st.sidebar.slider("Execution Slippage %", min_value=0.0, max_value=0.1, value=0.02, step=0.01)

st.sidebar.subheader("🛡️ Setup & Filters")
sweep_mode_label = st.sidebar.selectbox(
    "Sweep Mode",
    options=["Single Sweep & Reclaim", "Double Sweep & Accumulation", "Triple Sweep & Accumulation", "Double & Triple Sweep"],
    index=0
)

if "Single" in sweep_mode_label:
    sweep_mode = "single"
elif "Double" in sweep_mode_label and "Triple" in sweep_mode_label:
    sweep_mode = "both"
elif "Double" in sweep_mode_label:
    sweep_mode = "double"
else:
    sweep_mode = "triple"

# MULTI-SELECT TRIGGER CONFIRMATION
triggers_selected = st.sidebar.multiselect(
    "Entry Trigger Confirmation",
    options=[
        "Bullish/Bearish Engulfing Pattern",
        "3-Candle Trend Pullback",
        "Pinbar / Hammer Price Rejection",
        "Immediate Limit Reclaim",
        "Volume Spike Confirmation"
    ],
    default=["Bullish/Bearish Engulfing Pattern"],
    help="Select one or multiple triggers. Use the logic selector below to choose AND or OR mode."
)

# AND / OR LOGIC SELECTOR
trigger_logic_label = st.sidebar.radio(
    "Trigger Combination Logic",
    options=["OR — Any trigger fires the trade", "AND — All triggers must fire together"],
    index=0,
    help="OR mode: A trade fires if ANY selected trigger is true (more trades, faster entries). AND mode: ALL selected triggers must be true simultaneously (fewer but higher-confidence trades)."
)
trigger_logic = "OR" if "OR" in trigger_logic_label else "AND"

# Map selected strings to backend code keys
trigger_modes = []
for t in triggers_selected:
    if "Engulfing" in t: trigger_modes.append("engulfing")
    elif "Pullback" in t: trigger_modes.append("pullback")
    elif "Pinbar" in t: trigger_modes.append("pinbar")
    elif "Immediate" in t: trigger_modes.append("immediate")
    elif "Volume" in t: trigger_modes.append("volume")

if not trigger_modes:
    # Safe fallback default
    trigger_modes = ["engulfing"]

trade_direction_label = st.sidebar.selectbox(
    "Trade Direction",
    options=["Both (Long & Short)", "Long Only (Sweeps Only)", "Short Only (Sweeps Only)"],
    index=0
)
trade_direction = "both" if "Both" in trade_direction_label else ("long" if "Long" in trade_direction_label else "short")

use_trend_filter = st.sidebar.toggle("Enable 200 EMA Filter", value=True)

st.sidebar.subheader("🎯 Risk Management (SL & TP)")
sltp_mode_label = st.sidebar.selectbox(
    "SL/TP Calculation Mode",
    options=["Wick Structure + RR Ratio", "Fixed Percentage Offset"],
    index=0
)
sltp_mode = "structure" if "Structure" in sltp_mode_label else "percentage"

# Initialize defaults
rr_ratio = 2.0
fixed_sl_pct = 20.0
fixed_tp_pct = 2.0

if sltp_mode == "structure":
    rr_ratio = st.sidebar.slider("Take Profit Multiplier (Risk:Reward)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
else:
    fixed_sl_pct = st.sidebar.slider("Stop Loss %", min_value=0.1, max_value=20.0, value=20.0, step=0.1)
    fixed_tp_pct = st.sidebar.slider("Take Profit %", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

st.sidebar.subheader("🏃 Trailing Stop Loss")
use_trailing_sl = st.sidebar.toggle("Enable Trailing Stop", value=False)
trailing_trigger_pct = 1.5
trailing_distance_pct = 1.0

if use_trailing_sl:
    trailing_trigger_pct = st.sidebar.slider("Trailing Trigger %", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    trailing_distance_pct = st.sidebar.slider("Trailing Distance %", min_value=0.2, max_value=3.0, value=1.0, step=0.1)

# --- PREMIUM INSTITUTIONAL UPGRADES ---
st.sidebar.subheader("🏆 Premium Institutional Upgrades")
enable_institutional = st.sidebar.toggle("Enable Premium Upgrades", value=False, help="Unlocks prop-desk level volatility, time, and session extremes filters.")

use_fixed_session_levels = False
use_atr_penetration = False
atr_penetration_factor = 0.25
use_atr_tp = False
atr_tp_factor = 2.5
use_time_filter = False
enable_breakout_reversal = False
breakout_sl_pct = 20.0
breakout_only = False

if enable_institutional:
    enable_breakout_reversal = st.sidebar.checkbox(
        "Failed Triple Sweep Breakout Flip",
        value=True,
        help="If a Triple Sweep fails (breaks support/resistance), immediately flips and enters a breakout trade in the opposite direction."
    )
    if enable_breakout_reversal:
        breakout_sl_pct = st.sidebar.slider(
            "Breakout Stop Loss %",
            min_value=0.1, max_value=20.0, value=20.0, step=0.1,
            help="Set the fixed percentage Stop Loss for breakout flip trades."
        )
        breakout_only = st.sidebar.checkbox(
            "Breakout Flips Only (Suppress Normal Sweeps)",
            value=False,
            help="If checked, completely bypasses standard reverse-sweep entries and ONLY trades the breakout when a sweep fails."
        )
    use_fixed_session_levels = st.sidebar.checkbox(
        "Previous Day Extremes (PDH/PDL)", 
        value=True,
        help="Sweeps fixed session levels (PDH/PDL) instead of rolling windows."
    )
    use_atr_penetration = st.sidebar.checkbox(
        "Require Volatility Wick Sweep", 
        value=True,
        help="Filters out quiet noise. Require sweeps to go deep by ATR factor."
    )
    if use_atr_penetration:
        atr_penetration_factor = st.sidebar.slider(
            "Sweep Depth (ATR Multiple)", 
            min_value=0.05, max_value=1.5, value=0.25, step=0.05
        )
    use_time_filter = st.sidebar.checkbox(
        "Time-of-Day Session Filter", 
        value=True,
        help="Enters trades only during high-volume London & NY open hours (UTC)."
    )
    use_atr_tp = st.sidebar.checkbox(
        "Volatility-Based Take Profits (ATR)", 
        value=False,
        help="Uses dynamic Average True Range profit targets instead of static R:R."
    )
    if use_atr_tp:
        atr_tp_factor = st.sidebar.slider(
            "Take Profit Size (ATR Multiple)", 
            min_value=1.0, max_value=5.0, value=2.5, step=0.1
        )

# Caching the historical candle downloads to make updates INSTANT
def get_cached_history(symbol, interval, days, market_choice):
    if "Crypto" in market_choice:
        return fetch_extended_history(symbol=symbol, interval=interval, days=days)
    else:
        return fetch_nse_history(ticker=symbol, days=days)

# Run Backtest Trigger
if st.sidebar.button("⚡ Run Portfolio Backtest", width='stretch'):
    if not symbols_selected:
        st.error("Error: Please select at least one symbol in the basket.")
    else:
        with st.spinner("Connecting to markets and running portfolio event loop..."):
            # 1. Fetch data (loads instantly from memory cache in 0.0s, or downloads all coins in parallel!)
            from concurrent.futures import ThreadPoolExecutor
            
            loaded_dfs = {}
            
            def load_sym_data(sym):
                try:
                    df = get_cached_history(sym, interval, days, market_choice)
                    return sym, df
                except Exception as e:
                    print(f"Error loading {sym}: {e}")
                    return sym, None
                    
            with ThreadPoolExecutor(max_workers=min(10, len(symbols_clean) or 1)) as executor:
                results = list(executor.map(load_sym_data, symbols_clean))
                
            for sym, df in results:
                if df is not None and not df.empty:
                    loaded_dfs[sym] = df
                    
            if not loaded_dfs:
                st.error("Error: Could not retrieve data for any selected symbols.")
            else:
                # 2. Run Strategy calculations passing trigger_modes list
                df_signals_dict = {}
                for sym, df in loaded_dfs.items():
                    df_signals_dict[sym] = generate_signals(
                        df, 
                        sweep_mode=sweep_mode, 
                        use_trend_filter=use_trend_filter, 
                        trade_direction=trade_direction,
                        sltp_mode=sltp_mode,
                        rr_ratio=rr_ratio,
                        fixed_sl_pct=fixed_sl_pct,
                        fixed_tp_pct=fixed_tp_pct,
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
                        breakout_only=breakout_only
                    )
                    
                # 3. Simulate multi-asset chronological trading
                results = run_backtest(
                    df_dict=df_signals_dict, 
                    initial_balance=initial_balance, 
                    risk_per_trade=risk_per_trade,
                    fee_pct=fee_pct,
                    slippage_pct=slippage_pct,
                    use_trailing_sl=use_trailing_sl,
                    trailing_trigger_pct=trailing_trigger_pct,
                    trailing_distance_pct=trailing_distance_pct,
                    max_concurrent_trades=max_concurrent_trades
                )
                
                # Store results and parameters in session_state so they persist across re-runs
                st.session_state.backtest_results = results
                st.session_state.backtest_df_signals_dict = df_signals_dict
                st.session_state.backtest_loaded_dfs = loaded_dfs
                st.session_state.backtest_symbols = symbols_clean
                st.session_state.backtest_market_choice = market_choice
                st.session_state.backtest_use_trend_filter = use_trend_filter
                st.session_state.backtest_trigger_modes = trigger_modes
                st.session_state.backtest_trigger_logic = trigger_logic
                st.session_state.backtest_sweep_mode = sweep_mode
                st.session_state.backtest_trade_direction = trade_direction
                st.session_state.backtest_sltp_mode = sltp_mode
                st.session_state.backtest_rr_ratio = rr_ratio
                st.session_state.backtest_risk_per_trade = risk_per_trade
                st.session_state.backtest_enable_institutional = enable_institutional
                st.session_state.backtest_use_fixed_session_levels = use_fixed_session_levels
                st.session_state.backtest_use_atr_penetration = use_atr_penetration
                st.session_state.backtest_atr_penetration_factor = atr_penetration_factor
                st.session_state.backtest_use_atr_tp = use_atr_tp
                st.session_state.backtest_atr_tp_factor = atr_tp_factor
                st.session_state.backtest_use_time_filter = use_time_filter
                st.session_state.backtest_enable_breakout_reversal = enable_breakout_reversal
                st.session_state.backtest_breakout_sl_pct = breakout_sl_pct
                st.session_state.backtest_breakout_only = breakout_only
                st.session_state.backtest_max_concurrent_trades = max_concurrent_trades

# Display results from session_state (persists when other buttons are clicked)
if "backtest_results" in st.session_state:
    results = st.session_state.backtest_results
    df_signals_dict = st.session_state.backtest_df_signals_dict
    
    # Retrieve parameters used for this specific backtest run
    backtest_symbols = st.session_state.get("backtest_symbols", symbols_clean)
    backtest_market_choice = st.session_state.get("backtest_market_choice", market_choice)
    backtest_use_trend_filter = st.session_state.get("backtest_use_trend_filter", use_trend_filter)
    backtest_trigger_modes = st.session_state.get("backtest_trigger_modes", trigger_modes)
    backtest_trigger_logic = st.session_state.get("backtest_trigger_logic", trigger_logic)
    backtest_sweep_mode = st.session_state.get("backtest_sweep_mode", sweep_mode)
    backtest_trade_direction = st.session_state.get("backtest_trade_direction", trade_direction)
    backtest_sltp_mode = st.session_state.get("backtest_sltp_mode", sltp_mode)
    backtest_rr_ratio = st.session_state.get("backtest_rr_ratio", rr_ratio)
    backtest_risk_per_trade = st.session_state.get("backtest_risk_per_trade", risk_per_trade)
    backtest_enable_institutional = st.session_state.get("backtest_enable_institutional", False)
    backtest_use_fixed_session_levels = st.session_state.get("backtest_use_fixed_session_levels", False)
    backtest_use_atr_penetration = st.session_state.get("backtest_use_atr_penetration", False)
    backtest_atr_penetration_factor = st.session_state.get("backtest_atr_penetration_factor", 0.25)
    backtest_use_atr_tp = st.session_state.get("backtest_use_atr_tp", False)
    backtest_atr_tp_factor = st.session_state.get("backtest_atr_tp_factor", 2.5)
    backtest_use_time_filter = st.session_state.get("backtest_use_time_filter", False)
    backtest_enable_breakout_reversal = st.session_state.get("backtest_enable_breakout_reversal", False)
    backtest_breakout_sl_pct = st.session_state.get("backtest_breakout_sl_pct", 1.0)
    backtest_breakout_only = st.session_state.get("backtest_breakout_only", False)
    backtest_max_concurrent_trades = st.session_state.get("backtest_max_concurrent_trades", 2)
    
    metrics = results["metrics"]
    trades = results["trades"]
    equity_curve = results["equity_curve"]
                
    # --- STRATEGY RATING BANNER ENGINE ---
    sharpe = metrics["sharpe_ratio"]
    total_trades = metrics["total_trades"]
    
    if total_trades == 0:
        rating_badge = """
            <div style="background-color: #1a202c; border: 1px solid #4a5568; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #cbd5e0;">🔍 Strategy Rating: NO TRADES DETECTED</span>
                <br><p style="color: #a0aec0; margin: 8px 0 0 0;">No trading signals occurred. Try extending history length, disabling trend filter, or choosing different assets.</p>
            </div>
        """
    elif sharpe >= 3.0:
        rating_badge = """
            <div style="background-color: #1e3a2f; border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #10b981;">⚡ Strategy Rating: GOD-TIER / OUTSTANDING</span>
                <br><p style="color: #a3b899; margin: 8px 0 0 0;">Annualized Sharpe of 3.0+. Elite prop-desk grade equity curve with massive returns per unit of volatility!</p>
            </div>
        """
    elif sharpe >= 2.0:
        rating_badge = """
            <div style="background-color: #1a2e3b; border: 1px solid #3b82f6; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #3b82f6;">🔥 Strategy Rating: EXCELLENT / PROFESSIONAL</span>
                <br><p style="color: #93c5fd; margin: 8px 0 0 0;">Annualized Sharpe of 2.0 - 2.9. High-performance, extremely smooth return profile matching institutional standard quality.</p>
            </div>
        """
    elif sharpe >= 1.0:
        rating_badge = """
            <div style="background-color: #3b2a1a; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #f59e0b;">🟢 Strategy Rating: GOOD / PROFITABLE</span>
                <br><p style="color: #fde047; margin: 8px 0 0 0;">Annualized Sharpe of 1.0 - 1.9. Solid, viable retail performance with stable risk-managed capital growth.</p>
            </div>
        """
    else:
        rating_badge = """
            <div style="background-color: #3b1e1e; border: 1px solid #ef4444; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #ef4444;">❌ Strategy Rating: SUB-OPTIMAL / RISKY</span>
                <br><p style="color: #fca5a5; margin: 8px 0 0 0;">Annualized Sharpe below 1.0. High risk for very little reward. Try toggling EMA filters, adjusting SL/TP settings, or testing other symbols.</p>
            </div>
        """
    
    st.markdown(rating_badge, unsafe_allow_html=True)
    
    # Performance Banner Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Trades</div>
                <div class="metric-value text-indigo">{metrics['total_trades']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        color_class = "text-green" if metrics['net_profit'] >= 0 else "text-red"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Net Profit ($)</div>
                <div class="metric-value {color_class}">${metrics['net_profit']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        color_class = "text-green" if metrics['net_profit_pct'] >= 0 else "text-red"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROI</div>
                <div class="metric-value {color_class}">{metrics['net_profit_pct']}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value text-green">{metrics['win_rate']}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value text-red">{metrics['max_drawdown']}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    # INDIVIDUAL ASSET BREAKDOWN SECTION
    st.subheader("📊 Individual Asset Performance Breakdown")
    breakdown_records = []
    
    for sym in backtest_symbols:
        sym_trades = trades[trades["symbol"] == sym] if not trades.empty else pd.DataFrame()
        if sym_trades.empty:
            breakdown_records.append({
                "Asset": sym,
                "Total Trades": 0,
                "Win Rate (%)": "0.00%",
                "Net Profit ($)": "$0.00",
                "Performance Result": "No Trades Triggers"
            })
        else:
            sym_wins = len(sym_trades[sym_trades["result"].isin(["WIN", "TSL WIN"])])
            sym_win_rate = (sym_wins / len(sym_trades)) * 100
            sym_profit = sym_trades["pnl"].sum()
            breakdown_records.append({
                "Asset": sym,
                "Total Trades": len(sym_trades),
                "Win Rate (%)": f"{sym_win_rate:.2f}%",
                "Net Profit ($)": f"${sym_profit:,.2f}",
                "Performance Result": "🟢 PROFIT" if sym_profit >= 0 else "🔴 LOSS"
            })
    st.dataframe(pd.DataFrame(breakdown_records), width='stretch')
    st.markdown("---")
    
    # Layout Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Interactive Chart", "📜 Trade History", "📈 Equity Curve", "⚡ Optimization Lab", "🤖 Live Testnet Bot", "💰 Paper Trading"])
    
    with tab1:
        st.subheader(f"🔍 Dynamic Price & Liquidity Levels Chart ({', '.join(backtest_symbols)})")
        
        # Dynamic downsampling control to eliminate browser SVG rendering lag!
        col_c1, col_c2 = st.columns([1.5, 2.5])
        with col_c1:
            chart_range_label = st.selectbox(
                "Select Candle Render Range (Improves interaction speed)",
                options=[
                    "Last 500 candles (~5 days of 15m data)", 
                    "Last 1,000 candles (~10 days of 15m data)", 
                    "Last 2,500 candles (~26 days of 15m data)", 
                    "Full Historical Dataset (May lag browser)"
                ],
                index=1,
                help="Plotly candlesticks slow down when rendering > 2,000 candles. Slicing limits DOM complexity and keeps interactions butter-smooth!"
            )
            
        fig = go.Figure()
        primary_sym = backtest_symbols[0]
        primary_df = df_signals_dict[primary_sym]
        
        # Apply downsampling range slicing
        if "500" in chart_range_label:
            slice_df = primary_df.tail(500)
        elif "1,000" in chart_range_label:
            slice_df = primary_df.tail(1000)
        elif "2,500" in chart_range_label:
            slice_df = primary_df.tail(2500)
        else:
            slice_df = primary_df
            
        # Candlesticks using sliced data
        fig.add_trace(go.Candlestick(
            x=slice_df['timestamp'],
            open=slice_df['open'],
            high=slice_df['high'],
            low=slice_df['low'],
            close=slice_df['close'],
            name=f'{primary_sym} Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))
        
        # 200 EMA using sliced data
        fig.add_trace(go.Scatter(
            x=slice_df['timestamp'],
            y=slice_df['ema_200'],
            name='200 EMA (Trend Filter)',
            line=dict(color='#8b5cf6', width=1.5, dash='solid' if backtest_use_trend_filter else 'dot')
        ))
        
        # 24h Low / PDL Levels
        pdl_pds = slice_df['pdl'] if ('pdl' in slice_df.columns and backtest_use_fixed_session_levels) else slice_df['24h_low']
        fig.add_trace(go.Scatter(
            x=slice_df['timestamp'],
            y=pdl_pds,
            name='Previous Day Low (PDL)' if backtest_use_fixed_session_levels else ('Previous Session Low' if "Indian" in backtest_market_choice else '24h Low'),
            line=dict(color='#f59e0b', width=1, dash='dash')
        ))
        
        # 24h High / PDH Levels
        pdh_pds = slice_df['pdh'] if ('pdh' in slice_df.columns and backtest_use_fixed_session_levels) else slice_df['24h_high']
        fig.add_trace(go.Scatter(
            x=slice_df['timestamp'],
            y=pdh_pds,
            name='Previous Day High (PDH)' if backtest_use_fixed_session_levels else ('Previous Session High' if "Indian" in backtest_market_choice else '24h High'),
            line=dict(color='#3b82f6', width=1, dash='dash')
        ))
        
        # Add markers for LONG entries (within our sliced window!)
        long_signals = slice_df[slice_df['signal'] == 1]
        if not long_signals.empty:
            fig.add_trace(go.Scatter(
                x=long_signals['timestamp'],
                y=long_signals['close'],
                mode='markers',
                name='LONG Entry',
                marker=dict(
                    symbol='triangle-up', size=14, color='#10b981', line=dict(width=2, color='#ffffff')
                )
            ))
            
        # Add markers for SHORT entries (within our sliced window!)
        short_signals = slice_df[slice_df['signal'] == -1]
        if not short_signals.empty:
            fig.add_trace(go.Scatter(
                x=short_signals['timestamp'],
                y=short_signals['close'],
                mode='markers',
                name='SHORT Entry',
                marker=dict(
                    symbol='triangle-down', size=14, color='#ef4444', line=dict(width=2, color='#ffffff')
                )
            ))
            
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False, # Heavy rangeslider doubles DOM node rendering and is the #1 cause of Plotly scroll lag!
            height=650,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
        )
        st.plotly_chart(fig, width='stretch')
        
    with tab2:
        st.subheader("📜 Completed Trades Log")
        if not trades.empty:
            display_df = trades.copy()
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            display_df['balance'] = display_df['balance'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, width='stretch')
        else:
            st.info("No trades completed. Adjust settings or choose other assets.")
            
    with tab3:
        st.subheader("📈 Capital Growth (Equity Curve)")
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=equity_curve['timestamp'],
            y=equity_curve['equity'],
            name='Portfolio Value',
            line=dict(color='#10b981', width=2),
            fill='tozeroy',
            fillcolor='rgba(16,185,129,0.1)'
        ))
        eq_fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(eq_fig, width='stretch')
        
    with tab4:
        st.subheader("⚡ Parametric Grid Search Optimizer")
        st.markdown("Run a parallel brute-force search over **20+ combinations of Stop Loss and Take Profit levels** using the current data. The laboratory will automatically score them and rank them by their **Sharpe Ratio** to find the mathematical sweet spot!")
        
        if "backtest_loaded_dfs" not in st.session_state:
            st.warning("⚠️ Please click '⚡ Run Portfolio Backtest' in the sidebar first to load historical data before running the optimizer!")
        else:
            if st.button("🚀 Start Grid Search Optimization", width='stretch'):
                with st.spinner("Brute-forcing parameters in the optimization lab..."):
                    opt_results = run_grid_search(
                        df_dict=st.session_state.backtest_loaded_dfs,
                        initial_balance=initial_balance,
                        risk_per_trade=risk_per_trade,
                        fee_pct=fee_pct,
                        slippage_pct=slippage_pct,
                        sweep_mode=sweep_mode,
                        use_trend_filter=use_trend_filter,
                        trade_direction=trade_direction,
                        rolling_window=rolling_window,
                        trigger_modes=trigger_modes,
                        trigger_logic=trigger_logic
                    )
                    st.success("Optimization Complete! Ranked Parameters by Sharpe Ratio:")
                    st.dataframe(opt_results, width='stretch')
    
    with tab5:
        st.subheader("🤖 Binance Futures Testnet — Live Mock Trading")
        st.markdown("""
        Connect to the **Binance Futures Testnet** and let the bot monitor live price data using your exact strategy configuration.
        Trades will execute on the testnet using **virtual funds** — no real money at risk!
        """)
        
        st.markdown("""
        > **Setup Instructions:**
        > 1. Go to [Binance Futures Testnet](https://testnet.binancefuture.com/) and create an account
        > 2. Generate API keys from the testnet dashboard
        > 3. Create a `.env` file in your project root (copy from `.env.example`)
        > 4. Paste your testnet API Key and Secret into the `.env` file
        """)
        
        st.markdown("---")
        
        # Only allow crypto symbols for live trading
        if "Crypto" not in market_choice:
            st.warning("⚠️ Live Testnet Bot is only available for Crypto (Binance) markets. Switch your market in the sidebar.")
        else:
            live_symbol = st.selectbox(
                "Select Symbol for Live Monitoring",
                options=symbols_clean if symbols_clean else ["BTCUSDT"],
                index=0,
                help="Choose which symbol the bot will monitor and trade on."
            )
            
            live_risk = st.number_input(
                "Risk Per Trade ($) — Testnet",
                min_value=1.0,
                max_value=1000.0,
                value=min(risk_per_trade, 10.0),
                step=1.0,
                help="Dollar amount risked per trade on the testnet account."
            )
            
            col_lev, col_margin, col_poll = st.columns(3)
            with col_lev:
                futures_leverage = st.select_slider(
                    "Leverage",
                    options=[1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100, 125],
                    value=10,
                    help="Futures leverage multiplier. Higher leverage = higher risk & reward."
                )
            with col_margin:
                margin_type_label = st.radio(
                    "Margin Mode",
                    options=["Isolated", "Cross"],
                    index=0,
                    help="Isolated: only the margin for this position is at risk. Cross: your entire account balance is shared as margin."
                )
                futures_margin = "ISOLATED" if margin_type_label == "Isolated" else "CROSSED"
            with col_poll:
                poll_interval = st.slider(
                    "Poll (sec)",
                    min_value=10,
                    max_value=120,
                    value=30,
                    step=5,
                    help="How often the bot checks for new candle closes."
                )
            
            # Config summary card
            inst_html = ""
            if enable_institutional:
                inst_html = f"""
                <div class="config-row"><span class="config-label" style="font-weight:bold;color:#a5b4fc;">🏆 Institutional</span><span class="config-value-green">ENABLED</span></div>
                <div class="config-row"><span class="config-label">Session Levels</span><span class="config-value">{'PDH/PDL' if use_fixed_session_levels else 'Rolling 24h'}</span></div>
                <div class="config-row"><span class="config-label">Volatility Sweep</span><span class="config-value">{'ON ('+str(atr_penetration_factor)+'x ATR)' if use_atr_penetration else 'OFF'}</span></div>
                <div class="config-row"><span class="config-label">Time Filter</span><span class="config-value">{'London & NY' if use_time_filter else '24/7'}</span></div>
                <div class="config-row"><span class="config-label">Volatility TP</span><span class="config-value">{'ON ('+str(atr_tp_factor)+'x ATR)' if use_atr_tp else 'OFF'}</span></div>
                """
            st.markdown(f"""
            <div class="config-preview" style="margin: 15px 0;">
                <div class="config-row"><span class="config-label">Symbol</span><span class="config-value">{live_symbol}</span></div>
                <div class="config-row"><span class="config-label">Leverage</span><span class="config-value" style="color: {'#ef4444' if futures_leverage >= 50 else '#f59e0b' if futures_leverage >= 20 else '#10b981'};">{futures_leverage}x {margin_type_label}</span></div>
                <div class="config-row"><span class="config-label">Sweep Mode</span><span class="config-value">{sweep_mode.title()}</span></div>
                <div class="config-row"><span class="config-label">Triggers</span><span class="config-value">{", ".join(trigger_modes)} ({trigger_logic})</span></div>
                <div class="config-row"><span class="config-label">EMA Filter</span><span class="config-value">{'ON' if use_trend_filter else 'OFF'}</span></div>
                <div class="config-row"><span class="config-label">Direction</span><span class="config-value">{trade_direction.title()}</span></div>
                <div class="config-row"><span class="config-label">SL/TP</span><span class="config-value">{sltp_mode}</span></div>
                <div class="config-row"><span class="config-label">Risk/Trade</span><span class="config-value-green">${live_risk:,.0f}</span></div>
                <div class="config-row"><span class="config-label">Poll</span><span class="config-value">Every {poll_interval}s</span></div>
                {inst_html}
            </div>
            """, unsafe_allow_html=True)
            
            if futures_leverage >= 50:
                st.warning(f"⚠️ **{futures_leverage}x leverage is extremely high!** A {round(100/futures_leverage, 1)}% move against you = liquidation. This is testnet money, but build good habits.")
            
            col_test, col_run = st.columns(2)
            
            with col_test:
                if st.button("🔌 Test Connection", width='stretch'):
                    with st.spinner("Testing Binance Testnet connection..."):
                        test_client = connect_binance()
                        if test_client:
                            st.success("✅ Connected to Binance Futures Testnet!")
                            try:
                                balances = test_client.futures_account_balance()
                                for b in balances:
                                    if b["asset"] == "USDT":
                                        st.metric("Testnet USDT Balance", f"${float(b['balance']):,.2f}")
                                pos = get_open_positions(test_client, live_symbol)
                                if pos:
                                    st.info(f"📌 Active {pos['side']} position: {pos['size']} units @ ${pos['entry_price']:,.2f} (PnL: ${pos['unrealized_pnl']:,.2f})")
                                else:
                                    st.info(f"No open position on {live_symbol}. Ready to trade!")
                            except Exception as e:
                                st.warning(f"Connected but could not fetch details: {e}")
                        else:
                            st.warning("⚠️ Running in DUMMY mode — no API keys found in `.env`. The bot will simulate signals but won't place real testnet orders.")
            
            with col_run:
                if st.button("🚀 Start Live Scan (Single Pass)", width='stretch'):
                    with st.spinner(f"Fetching live {live_symbol} candles and scanning for signals..."):
                        scan_client = connect_binance()
                        df_live = fetch_live_candles(scan_client, live_symbol, limit=300)
                        
                        if df_live.empty:
                            st.error("Failed to fetch live candle data.")
                        else:
                            df_live_signals = generate_signals(
                                df_live,
                                sweep_mode=sweep_mode,
                                use_trend_filter=use_trend_filter,
                                trade_direction=trade_direction,
                                sltp_mode=sltp_mode,
                                rr_ratio=rr_ratio,
                                fixed_sl_pct=fixed_sl_pct,
                                fixed_tp_pct=fixed_tp_pct,
                                rolling_window=96,
                                trigger_modes=trigger_modes,
                                trigger_logic=trigger_logic
                            )
                            st.session_state.live_scan_data = {
                                "df": df_live_signals,
                                "symbol": live_symbol,
                                "risk": live_risk,
                                "leverage": futures_leverage,
                                "margin": futures_margin
                            }
                            st.success(f"✅ Loaded {len(df_live)} live candles for {live_symbol}")
            
            # Stateful display and action execution (not nested)
            if "live_scan_data" in st.session_state and st.session_state.live_scan_data["symbol"] == live_symbol:
                scan_data = st.session_state.live_scan_data
                df_live_signals = scan_data["df"]
                scan_client = connect_binance()
                
                # Show last 10 candle states
                st.markdown("#### 📊 Recent Candle States")
                recent = df_live_signals.tail(10)[["timestamp", "close", "signal", "state_log"]].copy()
                recent["close"] = recent["close"].apply(lambda x: f"${x:,.2f}")
                recent["signal"] = recent["signal"].map({1: "🟢 LONG", -1: "🔴 SHORT", 0: "—"})
                st.dataframe(recent, width='stretch')
                
                # Check last closed candle for signal
                latest = df_live_signals.iloc[-2]
                signal = latest["signal"]
                close_price = latest["close"]
                
                if signal == 1:
                    sl = latest["sl_level"]
                    tp = latest["tp_level"]
                    risk_dist = close_price - sl
                    precision = get_symbol_precision(scan_client, live_symbol)
                    pos_size = scan_data["risk"] / risk_dist if risk_dist > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: #1e3a2f; border: 1px solid #10b981; border-radius: 12px; padding: 20px; margin: 15px 0; text-align: center;">
                        <span style="font-size: 1.5rem; font-weight: 700; color: #10b981;">🟢 LONG SIGNAL DETECTED!</span>
                        <br><br>
                        <span style="color: #9ca3af;">Entry: <strong style="color: #fff;">${close_price:,.2f}</strong> | SL: <strong style="color: #ef4444;">${sl:,.2f}</strong> | TP: <strong style="color: #10b981;">${tp:,.2f}</strong> | Size: <strong style="color: #a5b4fc;">{round(pos_size, precision['qty_precision'])}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("⚡ Execute LONG on Testnet", width='stretch', key="exec_long"):
                        with st.spinner("Placing order on Binance Testnet..."):
                            success = place_order(scan_client, live_symbol, "BUY", pos_size, sl, tp, precision, leverage=scan_data["leverage"], margin_type=scan_data["margin"])
                            if success:
                                st.success("🎉 LONG order placed with SL & TP brackets!")
                                st.session_state.live_scan_data = None
                            else:
                                st.error("Failed to place order. Check your API keys and testnet balance.")
                                
                elif signal == -1:
                    sl = latest["sl_level"]
                    tp = latest["tp_level"]
                    risk_dist = sl - close_price
                    precision = get_symbol_precision(scan_client, live_symbol)
                    pos_size = scan_data["risk"] / risk_dist if risk_dist > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: #3b1e1e; border: 1px solid #ef4444; border-radius: 12px; padding: 20px; margin: 15px 0; text-align: center;">
                        <span style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">🔴 SHORT SIGNAL DETECTED!</span>
                        <br><br>
                        <span style="color: #9ca3af;">Entry: <strong style="color: #fff;">${close_price:,.2f}</strong> | SL: <strong style="color: #ef4444;">${sl:,.2f}</strong> | TP: <strong style="color: #10b981;">${tp:,.2f}</strong> | Size: <strong style="color: #a5b4fc;">{round(pos_size, precision['qty_precision'])}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("⚡ Execute SHORT on Testnet", width='stretch', key="exec_short"):
                        with st.spinner("Placing order on Binance Testnet..."):
                            success = place_order(scan_client, live_symbol, "SELL", pos_size, sl, tp, precision, leverage=scan_data["leverage"], margin_type=scan_data["margin"])
                            if success:
                                st.success("🎉 SHORT order placed with SL & TP brackets!")
                                st.session_state.live_scan_data = None
                            else:
                                st.error("Failed to place order. Check your API keys and testnet balance.")
                else:
                    st.info(f"⏳ No signal on the latest closed candle ({latest['timestamp']}). The state machine is at: `{latest['state_log']}`")
                
                # Show the live chart
                st.markdown("#### 📈 Live Price Chart")
                live_fig = go.Figure()
                live_fig.add_trace(go.Candlestick(
                    x=df_live_signals['timestamp'].tail(100),
                    open=df_live_signals['open'].tail(100),
                    high=df_live_signals['high'].tail(100),
                    low=df_live_signals['low'].tail(100),
                    close=df_live_signals['close'].tail(100),
                    name=f'{live_symbol}',
                    increasing_line_color='#10b981',
                    decreasing_line_color='#ef4444'
                ))
                if 'ema_200' in df_live_signals.columns:
                    live_fig.add_trace(go.Scatter(
                        x=df_live_signals['timestamp'].tail(100),
                        y=df_live_signals['ema_200'].tail(100),
                        name='200 EMA',
                        line=dict(color='#8b5cf6', width=1.5)
                    ))
                if '24h_low' in df_live_signals.columns:
                    live_fig.add_trace(go.Scatter(
                        x=df_live_signals['timestamp'].tail(100),
                        y=df_live_signals['24h_low'].tail(100),
                        name='24h Low',
                        line=dict(color='#f59e0b', width=1, dash='dash')
                    ))
                if '24h_high' in df_live_signals.columns:
                    live_fig.add_trace(go.Scatter(
                        x=df_live_signals['timestamp'].tail(100),
                        y=df_live_signals['24h_high'].tail(100),
                        name='24h High',
                        line=dict(color='#3b82f6', width=1, dash='dash')
                    ))
                # Mark signals
                live_longs = df_live_signals[df_live_signals['signal'] == 1].tail(20)
                if not live_longs.empty:
                    live_fig.add_trace(go.Scatter(
                        x=live_longs['timestamp'], y=live_longs['close'],
                        mode='markers', name='LONG',
                        marker=dict(symbol='triangle-up', size=14, color='#10b981', line=dict(width=2, color='#fff'))
                    ))
                live_shorts = df_live_signals[df_live_signals['signal'] == -1].tail(20)
                if not live_shorts.empty:
                    live_fig.add_trace(go.Scatter(
                        x=live_shorts['timestamp'], y=live_shorts['close'],
                        mode='markers', name='SHORT',
                        marker=dict(symbol='triangle-down', size=14, color='#ef4444', line=dict(width=2, color='#fff'))
                    ))
                live_fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_rangeslider_visible=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
                )
                st.plotly_chart(live_fig, width='stretch')
            
            st.markdown("---")
            st.markdown("#### 🖥️ Run Continuous Bot (Terminal)")
            st.markdown(f"""
            To run the bot continuously in your terminal (recommended for overnight monitoring), use:
            
            ```bash
            python -c "from src.live_bot import run_live_bot; run_live_bot(symbol='{live_symbol}', sweep_mode='{sweep_mode}', use_trend_filter={use_trend_filter}, trade_direction='{trade_direction}', sltp_mode='{sltp_mode}', rr_ratio={rr_ratio}, trigger_modes={trigger_modes}, trigger_logic='{trigger_logic}', risk_per_trade={live_risk}, leverage={futures_leverage}, margin_type='{futures_margin}', poll_interval={poll_interval})"
            ```
            
            The terminal bot will run indefinitely, checking for signals every {poll_interval} seconds and auto-executing **{futures_leverage}x {margin_type_label}** futures trades on the testnet.
            """)
    
    with tab6:
        st.subheader("💰 Paper Trading — Live Prices, Zero Risk")
        st.markdown("""
        Uses **real Binance market data** (public API, no keys needed!) and simulates futures trades locally.
        Your strategy runs against live prices with a virtual balance — perfect for testing before going real.
        """)
        
        if "Crypto" not in market_choice:
            st.warning("⚠️ Paper Trading uses Binance Futures data. Switch to Crypto in the sidebar.")
        else:
            pt_col1, pt_col2, pt_col3 = st.columns(3)
            with pt_col1:
                pt_symbol = st.selectbox(
                    "Symbol",
                    options=symbols_clean if symbols_clean else ["BTCUSDT"],
                    index=0,
                    key="pt_symbol"
                )
            with pt_col2:
                pt_balance = st.number_input(
                    "Starting Balance ($)",
                    min_value=100.0, max_value=100000.0, value=5000.0, step=500.0,
                    key="pt_balance"
                )
            with pt_col3:
                pt_leverage = st.select_slider(
                    "Leverage",
                    options=[1, 2, 3, 5, 10, 15, 20, 25, 50],
                    value=10,
                    key="pt_leverage"
                )
            
            # Initialize paper trader in session state
            if "paper_trader" not in st.session_state:
                st.session_state.paper_trader = PaperTrader(initial_balance=pt_balance, leverage=pt_leverage)
                st.session_state.paper_logs = []
                st.session_state.last_paper_candle = None
            
            trader = st.session_state.paper_trader
            
            # Config card
            st.markdown(f"""
            <div class="config-preview" style="margin: 15px 0;">
                <div class="config-row"><span class="config-label">Symbol</span><span class="config-value">{pt_symbol}</span></div>
                <div class="config-row"><span class="config-label">Balance</span><span class="config-value-green">${trader.balance:,.2f}</span></div>
                <div class="config-row"><span class="config-label">Leverage</span><span class="config-value">{pt_leverage}x</span></div>
                <div class="config-row"><span class="config-label">Triggers</span><span class="config-value">{", ".join(trigger_modes)} ({trigger_logic})</span></div>
                <div class="config-row"><span class="config-label">Position</span><span class="config-value">{'FLAT' if trader.position is None else f"{trader.position['side']} @ ${trader.position['entry_price']:,.2f}"}</span></div>
                <div class="config-row"><span class="config-label">Trades</span><span class="config-value">{trader.trade_count}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                scan_clicked = st.button("🔍 Scan Now", width='stretch', help="Fetch live candles and check for signals")
            with btn_col2:
                reset_clicked = st.button("🔄 Reset Account", width='stretch')
            with btn_col3:
                close_clicked = st.button("❌ Close Position", width='stretch', disabled=(trader.position is None))
            
            if reset_clicked:
                st.session_state.paper_trader = PaperTrader(initial_balance=pt_balance, leverage=pt_leverage)
                st.session_state.paper_logs = []
                st.session_state.last_paper_candle = None
                st.session_state.paper_scan_data = None
                st.rerun()
            
            if close_clicked and trader.position is not None:
                current_price = get_current_price(pt_symbol)
                if current_price > 0:
                    result = trader._close_position(current_price, "MANUAL CLOSE")
                    emoji = "🟢" if result["result"] == "WIN" else "🔴"
                    st.session_state.paper_logs.append(f"{emoji} CLOSED: PnL ${result['pnl']:+,.2f} ({result['pnl_pct']:+.1f}%)")
                    st.session_state.paper_scan_data = None
                    st.rerun()
            
            if scan_clicked:
                with st.spinner(f"Fetching live {pt_symbol} data from Binance..."):
                    df_paper = fetch_public_candles(pt_symbol, limit=300)
                
                if df_paper.empty:
                    st.error("Failed to fetch candle data from Binance.")
                else:
                    df_paper_signals = generate_signals(
                        df_paper,
                        sweep_mode=sweep_mode,
                        use_trend_filter=use_trend_filter,
                        trade_direction=trade_direction,
                        sltp_mode=sltp_mode,
                        rr_ratio=rr_ratio,
                        fixed_sl_pct=fixed_sl_pct,
                        fixed_tp_pct=fixed_tp_pct,
                        rolling_window=96,
                        trigger_modes=trigger_modes,
                        trigger_logic=trigger_logic
                    )
                    st.session_state.paper_scan_data = {
                        "df": df_paper_signals,
                        "symbol": pt_symbol
                    }
            
            if "paper_scan_data" in st.session_state and st.session_state.paper_scan_data["symbol"] == pt_symbol:
                scan_data = st.session_state.paper_scan_data
                df_paper_signals = scan_data["df"]
                
                latest = df_paper_signals.iloc[-2]
                current = df_paper_signals.iloc[-1]
                signal = latest["signal"]
                close_price = latest["close"]
                live_price = current["close"]
                
                # Show live price
                price_col1, price_col2, price_col3 = st.columns(3)
                with price_col1:
                    st.metric(f"{pt_symbol} Live Price", f"${live_price:,.2f}")
                with price_col2:
                    st.metric("Strategy State", latest["state_log"])
                with price_col3:
                    if trader.position:
                        upnl = trader.get_unrealized_pnl(live_price)
                        st.metric("Unrealized PnL", f"${upnl:+,.2f}", delta=f"{(upnl/trader.position['margin_used'])*100:+.1f}%")
                    else:
                        st.metric("Position", "FLAT")
                
                # Check open position
                if trader.position:
                    result = trader.check_position(latest["high"], latest["low"], close_price)
                    if result:
                        emoji = "🟢" if result["result"] == "WIN" else "🔴"
                        st.session_state.paper_logs.append(f"{emoji} {result['reason']}: PnL ${result['pnl']:+,.2f} ({result['pnl_pct']:+.1f}%) | Balance: ${result['balance_after']:,.2f}")
                        st.session_state.paper_scan_data = None
                        st.rerun()
                
                # Open new position if signal and flat
                if trader.position is None and signal != 0:
                    sl = latest["sl_level"]
                    tp = latest["tp_level"]
                    side = "LONG" if signal == 1 else "SHORT"
                    risk_dist = (close_price - sl) if signal == 1 else (sl - close_price)
                    
                    if risk_dist > 0:
                        pos_size = risk_per_trade / risk_dist
                        emoji = "🟢" if signal == 1 else "🔴"
                        
                        st.markdown(f"""
                        <div style="background: {'#1e3a2f' if signal == 1 else '#3b1e1e'}; border: 1px solid {'#10b981' if signal == 1 else '#ef4444'}; border-radius: 12px; padding: 20px; margin: 15px 0; text-align: center;">
                            <span style="font-size: 1.5rem; font-weight: 700; color: {'#10b981' if signal == 1 else '#ef4444'};">{emoji} {side} SIGNAL DETECTED!</span>
                            <br><br>
                            <span style="color: #9ca3af;">Entry: <strong style="color: #fff;">${close_price:,.2f}</strong> | SL: <strong style="color: #ef4444;">${sl:,.2f}</strong> | TP: <strong style="color: #10b981;">${tp:,.2f}</strong></span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"⚡ Paper Trade {side}", width='stretch', key="pt_exec"):
                            success = trader.open_position(pt_symbol, side, close_price, pos_size, sl, tp)
                            if success:
                                st.session_state.paper_logs.append(f"{emoji} {side} OPENED @ ${close_price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}")
                                st.session_state.paper_scan_data = None
                                st.rerun()
                else:
                    if trader.position is None:
                        st.info(f"⏳ No signal. State: `{latest['state_log']}` — Click Scan again to check next candle.")
                
                # Recent candle states table
                st.markdown("#### 📊 Recent Candle States")
                recent = df_paper_signals.tail(10)[["timestamp", "close", "signal", "state_log"]].copy()
                recent["close"] = recent["close"].apply(lambda x: f"${x:,.2f}")
                recent["signal"] = recent["signal"].map({1: "🟢 LONG", -1: "🔴 SHORT", 0: "—"})
                st.dataframe(recent, width='stretch')
                
                # Live chart
                st.markdown("#### 📈 Live Price Chart")
                paper_fig = go.Figure()
                paper_fig.add_trace(go.Candlestick(
                    x=df_paper_signals['timestamp'].tail(100),
                    open=df_paper_signals['open'].tail(100),
                    high=df_paper_signals['high'].tail(100),
                    low=df_paper_signals['low'].tail(100),
                    close=df_paper_signals['close'].tail(100),
                    name=pt_symbol,
                    increasing_line_color='#10b981',
                    decreasing_line_color='#ef4444'
                ))
                if 'ema_200' in df_paper_signals.columns:
                    paper_fig.add_trace(go.Scatter(
                        x=df_paper_signals['timestamp'].tail(100),
                        y=df_paper_signals['ema_200'].tail(100),
                        name='200 EMA', line=dict(color='#8b5cf6', width=1.5)
                    ))
                if '24h_low' in df_paper_signals.columns:
                    paper_fig.add_trace(go.Scatter(
                        x=df_paper_signals['timestamp'].tail(100),
                        y=df_paper_signals['24h_low'].tail(100),
                        name='24h Low', line=dict(color='#f59e0b', width=1, dash='dash')
                    ))
                if '24h_high' in df_paper_signals.columns:
                    paper_fig.add_trace(go.Scatter(
                        x=df_paper_signals['timestamp'].tail(100),
                        y=df_paper_signals['24h_high'].tail(100),
                        name='24h High', line=dict(color='#3b82f6', width=1, dash='dash')
                    ))
                # Mark position entry
                if trader.position:
                    paper_fig.add_hline(
                        y=trader.position['entry_price'],
                        line_dash='dot', line_color='#a5b4fc',
                        annotation_text=f"{trader.position['side']} Entry"
                    )
                    paper_fig.add_hline(
                        y=trader.position['sl_price'],
                        line_dash='dash', line_color='#ef4444',
                        annotation_text='SL'
                    )
                    paper_fig.add_hline(
                        y=trader.position['tp_price'],
                        line_dash='dash', line_color='#10b981',
                        annotation_text='TP'
                    )
                paper_fig.update_layout(
                    template="plotly_dark", height=500,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_rangeslider_visible=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
                )
                st.plotly_chart(paper_fig, width='stretch')
            
            # Trade history and logs
            if trader.trade_history:
                st.markdown("#### 📝 Paper Trade History")
                stats = trader.get_stats()
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Total Trades", stats["total_trades"])
                with stat_col2:
                    st.metric("Win Rate", f"{stats['win_rate']}%")
                with stat_col3:
                    st.metric("Total PnL", f"${stats['total_pnl']:+,.2f}")
                with stat_col4:
                    st.metric("Balance", f"${stats['balance']:,.2f}", delta=f"{stats['return_pct']:+.1f}%")
                
                trade_df = pd.DataFrame(trader.trade_history)
                st.dataframe(trade_df, width='stretch')
            
            if st.session_state.paper_logs:
                st.markdown("#### 💬 Activity Log")
                for log_msg in reversed(st.session_state.paper_logs[-20:]):
                    st.text(log_msg)
            
            # Terminal command
            st.markdown("---")
            st.markdown("#### 🖥️ Run Continuous Paper Bot (Terminal)")
            st.markdown(f"""
            ```bash
            python -c "from src.paper_trader import run_paper_bot; run_paper_bot(symbol='{pt_symbol}', sweep_mode='{sweep_mode}', use_trend_filter={use_trend_filter}, trade_direction='{trade_direction}', trigger_modes={trigger_modes}, trigger_logic='{trigger_logic}', risk_per_trade={risk_per_trade}, leverage={pt_leverage}, initial_balance={pt_balance}, poll_interval=30)"
            ```
            Runs 24/7 in your terminal. Press `Ctrl+C` to stop — it saves all trades to `paper_trades.json`.
            """)
else:
    # ============================================================
    # PREMIUM INTERACTIVE LANDING PAGE
    # ============================================================
    
    # --- HERO BANNER ---
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">⚡ Institutional-Grade Quantitative Research</div>
            <div class="hero-title">Multi-Market Sweep & Optimization Lab</div>
            <div class="hero-subtitle">
                A premium backtesting engine that detects <strong style="color:#a5b4fc;">liquidity sweeps</strong> on intraday price data,
                confirms entries with advanced candlestick triggers, and simulates risk-managed portfolios across
                <strong style="color:#34d399;">Crypto (Binance)</strong> and <strong style="color:#f59e0b;">Indian Stocks (NSE)</strong>.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # --- FEATURE CARDS ---
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">🔍</span>
                <div class="feature-title">Sweep Detection</div>
                <div class="feature-desc">Bi-directional state machine scanning both 24h Highs and Lows for institutional stop-hunts in real-time.</div>
            </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
            <div class="feature-card" style="animation-delay: 0.15s;">
                <span class="feature-icon">🎯</span>
                <div class="feature-title">5 Entry Triggers</div>
                <div class="feature-desc">Engulfing, Pullback, Pinbar, Immediate Reclaim, and Volume Spike — select one or combine all.</div>
            </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown("""
            <div class="feature-card" style="animation-delay: 0.3s;">
                <span class="feature-icon">🛡️</span>
                <div class="feature-title">Dynamic Risk Engine</div>
                <div class="feature-desc">Wick-structure stop losses, R:R multipliers, trailing stops, and fixed-dollar risk sizing.</div>
            </div>
        """, unsafe_allow_html=True)
    with fc4:
        st.markdown("""
            <div class="feature-card" style="animation-delay: 0.45s;">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">Grid Optimizer</div>
                <div class="feature-desc">Brute-force parameter search across 20+ SL/TP combinations ranked by Sharpe Ratio.</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # --- TRADE FLOW VISUALIZATION ---
    st.markdown("### 🔄 How a Trade Flows Through the Engine")
    st.caption("Every trade follows this precise 5-step lifecycle — from sweep detection to exit execution.")
    
    f1, a1, f2, a2, f3, a3, f4, a4, f5 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])
    with f1:
        st.markdown("""
            <div class="flow-step">
                <div class="flow-number">1</div>
                <div class="flow-label">Detect Sweep</div>
                <div class="flow-detail">Price wicks beyond 24h High or Low, triggering hidden liquidity.</div>
            </div>
        """, unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="flow-step">
                <div class="flow-number">2</div>
                <div class="flow-label">Confirm Reclaim</div>
                <div class="flow-detail">Price closes back inside the level — the sweep is validated.</div>
            </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with f3:
        ema_label = "EMA ✅" if use_trend_filter else "EMA OFF"
        st.markdown(f"""
            <div class="flow-step">
                <div class="flow-number">3</div>
                <div class="flow-label">Trend + Trigger</div>
                <div class="flow-detail">200 EMA check ({ema_label}), then entry trigger must fire.</div>
            </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with f4:
        st.markdown("""
            <div class="flow-step">
                <div class="flow-number">4</div>
                <div class="flow-label">Size & Enter</div>
                <div class="flow-detail">Position sized by $ risk ÷ SL distance. Fees & slippage applied.</div>
            </div>
        """, unsafe_allow_html=True)
    with a4:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with f5:
        st.markdown("""
            <div class="flow-step">
                <div class="flow-number">5</div>
                <div class="flow-label">SL / TP Exit</div>
                <div class="flow-detail">Trade exits at Stop Loss or Take Profit. Trailing SL locks profits.</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # --- LIVE CONFIGURATION PREVIEW + STRATEGY EXPLANATION ---
    col_config, col_explain = st.columns([1, 1.4])
    
    with col_config:
        st.markdown("### ⚙️ Your Current Configuration")
        st.caption("These values update live as you change the sidebar settings.")
        
        market_label = "Crypto (Binance)" if "Crypto" in market_choice else "Indian Stocks (NSE)"
        symbols_display = ", ".join(symbols_clean) if symbols_clean else "None selected"
        trigger_display = ", ".join(triggers_selected) if triggers_selected else "Engulfing (default)"
        ema_status = "Enabled ✅" if use_trend_filter else "Disabled ❌"
        sltp_display = f"Wick Structure (R:R {rr_ratio}x)" if sltp_mode == "structure" else f"Fixed (SL {fixed_sl_pct}%, TP {fixed_tp_pct}%)"
        trailing_display = f"On (Trigger: {trailing_trigger_pct}%, Trail: {trailing_distance_pct}%)" if use_trailing_sl else "Off"
        
        st.markdown(f"""
            <div class="config-preview">
                <div class="config-row"><span class="config-label">Market</span><span class="config-value">{market_label}</span></div>
                <div class="config-row"><span class="config-label">Symbols</span><span class="config-value">{symbols_display}</span></div>
                <div class="config-row"><span class="config-label">History</span><span class="config-value">{days} days @ 15m</span></div>
                <div class="config-row"><span class="config-label">Starting Cash</span><span class="config-value-green">${initial_balance:,.0f}</span></div>
                <div class="config-row"><span class="config-label">Risk Per Trade</span><span class="config-value-green">${risk_per_trade:,.0f}</span></div>
                <div class="config-row"><span class="config-label">Sweep Mode</span><span class="config-value">{sweep_mode.title()} Sweep</span></div>
                <div class="config-row"><span class="config-label">200 EMA Filter</span><span class="config-value">{ema_status}</span></div>
                <div class="config-row"><span class="config-label">SL/TP Mode</span><span class="config-value">{sltp_display}</span></div>
                <div class="config-row"><span class="config-label">Trigger Logic</span><span class="config-value">{trigger_logic} Mode</span></div>
                <div class="config-row"><span class="config-label">Trailing Stop</span><span class="config-value">{trailing_display}</span></div>
                <div class="config-row"><span class="config-label">Fees</span><span class="config-value">{fee_pct}% + {slippage_pct}% slip</span></div>
                {f'<div class="config-row" style="border-top:1px solid rgba(255,255,255,0.15);padding-top:10px;"><span class="config-label" style="font-weight:bold;color:#a5b4fc;">🏆 Institutional Upgrades</span><span class="config-value-green">ENABLED</span></div>' if enable_institutional else ''}
                {f'<div class="config-row"><span class="config-label">Session Levels</span><span class="config-value">{"PDH/PDL" if use_fixed_session_levels else "Rolling 24h"}</span></div>' if enable_institutional and use_fixed_session_levels else ''}
                {f'<div class="config-row"><span class="config-label">Volatility Sweep</span><span class="config-value">{"ON ("+str(atr_penetration_factor)+"x ATR)" if use_atr_penetration else "OFF"}</span></div>' if enable_institutional and use_atr_penetration else ''}
                {f'<div class="config-row"><span class="config-label">Time Filter</span><span class="config-value">{"ON (London & NY)" if use_time_filter else "OFF"}</span></div>' if enable_institutional and use_time_filter else ''}
                {f'<div class="config-row"><span class="config-label">Volatility TP</span><span class="config-value">{"ON ("+str(atr_tp_factor)+"x ATR)" if use_atr_tp else "OFF"}</span></div>' if enable_institutional and use_atr_tp else ''}
                {f'<div class="config-row"><span class="config-label">Breakout Flip</span><span class="config-value">{"ON (SL "+str(breakout_sl_pct)+"%)" if enable_breakout_reversal else "OFF"}</span></div>' if enable_institutional and enable_breakout_reversal else ''}
            </div>
        """, unsafe_allow_html=True)
    
    with col_explain:
        st.markdown("### 📖 Strategy Quick Reference")
        st.caption("Expand any section below to learn how that part of the strategy works.")
        
        with st.expander("🔍 What is a Liquidity Sweep?", expanded=False):
            st.markdown("""
            Price is pushed past a key level (24h High/Low) to **trigger retail stop-loss orders**. 
            Big players use this forced selling/buying to fill their own large positions at better prices. 
            When price then reverses back inside the level (the **reclaim**), it signals the sweep is complete 
            and a reversal is likely.
            """)
        
        if sweep_mode == "single":
            with st.expander("1️⃣ Single Sweep Mode (Currently Active)", expanded=True):
                st.markdown("""
                **LONG**: Price wicks *below* 24h Low → records the wick low as SL → closes back *above* the low (reclaim) → trigger fires → **Enter Long** 🟢  
                **SHORT**: Price wicks *above* 24h High → records the wick high as SL → closes back *below* the high → trigger fires → **Enter Short** 🔴
                """)
        elif sweep_mode == "double":
            with st.expander("2️⃣ Double Sweep Mode (Currently Active)", expanded=True):
                st.markdown("""
                **LONG**: Price sweeps 24h Low **twice** (L1 → reclaim → L3 sweep lower) → reclaims above L2 → trigger fires → **Enter Long** 🟢  
                **SHORT**: Price sweeps 24h High **twice** (H1 → reclaim → H3 sweep higher) → reclaims below H2 → trigger fires → **Enter Short** 🔴  
                
                *Double sweep provides stronger confirmation but generates fewer signals.*
                """)
        elif sweep_mode == "both":
            with st.expander("🔗 Double & Triple Sweep Mode (Currently Active)", expanded=True):
                st.markdown("""
                **Active Setup**: Scans for **either** Double Sweep or Triple Sweep setups concurrently!  
                
                *This combination captures the high frequency of double sweeps along with the maximum-tier confirmation of triple sweeps, giving you the best of both worlds!*
                """)
        else:
            with st.expander("3️⃣ Triple Sweep Mode (Currently Active)", expanded=True):
                st.markdown("""
                **LONG**: Price sweeps 24h Low **three times** (L1 → reclaim → L3 sweep → reclaim → L5 sweep lower) → reclaims above L3 → trigger fires → **Enter Long** 🟢  
                **SHORT**: Price sweeps 24h High **three times** (H1 → reclaim → H3 sweep → reclaim → H5 sweep higher) → reclaims below H3 → trigger fires → **Enter Short** 🔴  
                
                *Triple sweep represents maximum-tier institutional stop-hunts. Outstanding accuracy but extremely rare.*
                """)
        
        with st.expander("🎯 Entry Triggers Explained", expanded=False):
            trigger_names = ", ".join(triggers_selected) if triggers_selected else "Engulfing (default)"
            logic_desc = "**ALL** must fire simultaneously (AND)" if trigger_logic == "AND" else "**ANY** single trigger fires the trade (OR)"
            st.markdown(f"""
            **Your active triggers**: `{trigger_names}`  
            **Combination Logic**: {logic_desc}
            
            | Trigger | What It Detects |
            |---|---|
            | **Engulfing** | Current candle body completely covers previous candle |
            | **Pullback** | 3 trend candles + 1 pullback candle (3G-1R or 3R-1G) |
            | **Pinbar** | Tiny body with a wick ≥ 2x the body (price rejection) |
            | **Immediate** | Enter instantly on the reclaim — no pattern needed |
            | **Volume** | Candle volume ≥ 1.5x the 10-bar average volume |
            
            *{'In AND mode, every selected trigger must be true on the same candle for a trade to fire. This gives fewer but much higher-confidence entries.' if trigger_logic == 'AND' else 'In OR mode, a trade fires as soon as any single trigger is true. This gives more trades and faster entries.'}*
            """)
        
        with st.expander("📐 200 EMA Trend Filter", expanded=False):
            st.markdown(f"""
            **Currently**: {'**Enabled** ✅ — Longs only above EMA, Shorts only below' if use_trend_filter else '**Disabled** ❌ — Trades in any direction regardless of trend'}  
            
            The 200-period Exponential Moving Average acts as a trend compass. 
            When enabled, it prevents counter-trend trades that statistically have lower win rates.
            """)
        
        with st.expander("🛡️ Risk Management (SL, TP, Sizing)", expanded=False):
            st.markdown(f"""
            **Mode**: `{sltp_display}`  
            
            - **Wick Structure**: SL at the sweep wick tip, TP projected by R:R multiplier  
            - **Fixed %**: SL and TP at fixed percentage offsets from entry  
            - **Position Size** = Risk Budget ($) ÷ Risk Distance → you always lose exactly your risk budget if SL hits  
            - **Trailing Stop**: {'Enabled — locks profits when trade is ' + str(trailing_trigger_pct) + '% in profit' if use_trailing_sl else 'Disabled'}
            """)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # --- CALL TO ACTION ---
    st.markdown("""
        <div style="text-align: center; padding: 25px 0 10px 0;">
            <p style="color: #6b7280; font-size: 1.05rem;">👈 Configure your parameters in the sidebar, then click <strong style="color: #a5b4fc;">⚡ Run Portfolio Backtest</strong> to begin.</p>
        </div>
    """, unsafe_allow_html=True)
