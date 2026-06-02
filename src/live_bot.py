import time
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from src.strategy import generate_signals

# 1. Securely load API Keys from the hidden .env file
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# By default, we connect to the Binance Demo Trading platform (safe virtual funds!)
# Demo Trading uses different endpoints from the old testnet:
#   Futures: https://demo-fapi.binance.com
#   Spot:    https://demo-api.binance.com
USE_DEMO = True

def connect_binance() -> Client:
    """
    Safely connects to the Binance Demo Trading API.
    Uses demo-fapi.binance.com for futures.
    Returns None if no keys are configured (DUMMY mode).
    """
    if not API_KEY or not API_SECRET or API_KEY == "your_binance_api_key_here":
        print("\n⚠️  Warning: No API keys found in your .env file!")
        print("We are running in 'DUMMY mode' (simulated prints only).")
        print("To trade on Demo, create a '.env' file and add your keys.\n")
        return None
    
    if USE_DEMO:
        # Binance Demo Trading uses custom endpoints
        client = Client(api_key=API_KEY, api_secret=API_SECRET)
        # Override the futures URL to point to Demo Trading
        client.FUTURES_URL = 'https://demo-fapi.binance.com/fapi'
        client.API_URL = 'https://demo-api.binance.com/api'
        print("🤖 [BOT] Connected to Binance Demo Trading (demo-fapi.binance.com)!")
    else:
        client = Client(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        print("🤖 [BOT] Connected to Binance Testnet!")
    
    # Print account balance for confirmation
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b["asset"] == "USDT":
                print(f"💰 [BOT] Demo USDT Balance: ${float(b['balance']):,.2f}")
    except Exception as e:
        print(f"⚠️  Could not fetch balance: {e}")
    
    return client

def get_open_positions(client, symbol: str) -> dict:
    """
    Returns current open position for the symbol, or None if flat.
    """
    if client is None:
        return None
    try:
        positions = client.futures_position_information(symbol=symbol.upper())
        for pos in positions:
            qty = float(pos["positionAmt"])
            if qty != 0.0:
                return {
                    "symbol": pos["symbol"],
                    "side": "LONG" if qty > 0 else "SHORT",
                    "size": abs(qty),
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"])
                }
    except Exception as e:
        print(f"⚠️  Could not fetch positions: {e}")
    return None

def cancel_open_orders(client, symbol: str):
    """Cancel all open orders for a symbol (cleanup before new entry)."""
    if client is None:
        return
    try:
        client.futures_cancel_all_open_orders(symbol=symbol.upper())
        print(f"🧹 [BOT] Cleared existing open orders on {symbol}.")
    except Exception as e:
        print(f"⚠️  Could not cancel orders: {e}")

def configure_futures(client, symbol: str, leverage: int = 10, margin_type: str = "ISOLATED"):
    """
    Configure the futures contract before trading.
    Sets leverage and margin type (ISOLATED or CROSSED).
    Must be called before placing any orders.
    """
    if client is None:
        print(f"🤖 [DUMMY] Would set {symbol} leverage={leverage}x, margin={margin_type}")
        return True
    
    success = True
    
    # 1. Set leverage
    try:
        client.futures_change_leverage(symbol=symbol.upper(), leverage=leverage)
        print(f"⚙️  [BOT] Leverage set to {leverage}x on {symbol}")
    except Exception as e:
        err_msg = str(e)
        if "No need to change" in err_msg:
            print(f"⚙️  [BOT] Leverage already at {leverage}x on {symbol}")
        else:
            print(f"⚠️  Could not set leverage: {e}")
            success = False
    
    # 2. Set margin type
    try:
        client.futures_change_margin_type(symbol=symbol.upper(), marginType=margin_type)
        print(f"⚙️  [BOT] Margin type set to {margin_type} on {symbol}")
    except Exception as e:
        err_msg = str(e)
        if "No need to change" in err_msg:
            print(f"⚙️  [BOT] Margin type already {margin_type} on {symbol}")
        else:
            print(f"⚠️  Could not set margin type: {e}")
    
    return success

def get_account_info(client) -> dict:
    """
    Get comprehensive futures account info.
    Returns balance, available margin, total unrealized PnL.
    """
    if client is None:
        return None
    try:
        account = client.futures_account()
        return {
            "total_balance": float(account["totalWalletBalance"]),
            "available_balance": float(account["availableBalance"]),
            "total_unrealized_pnl": float(account["totalUnrealizedProfit"]),
            "total_margin_balance": float(account["totalMarginBalance"]),
            "total_open_orders": int(account.get("totalOpenOrderInitialMargin", 0) > 0),
            "positions": [
                {
                    "symbol": p["symbol"],
                    "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                    "size": abs(float(p["positionAmt"])),
                    "entry_price": float(p["entryPrice"]),
                    "unrealized_pnl": float(p["unrealizedProfit"]),
                    "leverage": int(p["leverage"]),
                    "margin_type": p.get("marginType", "unknown")
                }
                for p in account.get("positions", [])
                if float(p["positionAmt"]) != 0
            ]
        }
    except Exception as e:
        print(f"⚠️  Could not fetch account info: {e}")
        return None

def get_symbol_precision(client, symbol: str) -> dict:
    """
    Get the quantity precision and price precision for a symbol.
    Required to round orders correctly on Binance.
    """
    defaults = {"qty_precision": 3, "price_precision": 2, "min_qty": 0.001}
    if client is None:
        return defaults
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol.upper():
                qty_precision = s["quantityPrecision"]
                price_precision = s["pricePrecision"]
                min_qty = 0.001
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        min_qty = float(f["minQty"])
                return {"qty_precision": qty_precision, "price_precision": price_precision, "min_qty": min_qty}
    except Exception as e:
        print(f"⚠️  Could not fetch symbol info: {e}")
    return defaults

def place_order(client, symbol: str, side: str, position_size: float, sl_price: float, tp_price: float, precision: dict, leverage: int = 10, margin_type: str = "ISOLATED"):
    """
    Places a FUTURES market entry order with SL and TP brackets.
    Auto-configures leverage and margin type before entry.
    """
    qty = round(position_size, precision["qty_precision"])
    sl_rounded = round(sl_price, precision["price_precision"])
    tp_rounded = round(tp_price, precision["price_precision"])
    
    if qty < precision["min_qty"]:
        print(f"⚠️  [BOT] Calculated qty {qty} is below minimum {precision['min_qty']}. Skipping order.")
        return False
    
    if client is None:
        print(f"\n🤖 [DUMMY BOT] ══════════════════════════════════════")
        print(f"   Signal: {'🟢 LONG' if side == 'BUY' else '🔴 SHORT'}")
        print(f"   Symbol: {symbol} (Futures)")
        print(f"   Leverage: {leverage}x ({margin_type})")
        print(f"   Quantity: {qty}")
        print(f"   Stop Loss: ${sl_rounded:,.2f}")
        print(f"   Take Profit: ${tp_rounded:,.2f}")
        print(f"   ══════════════════════════════════════════════════")
        return True
        
    try:
        # Step 0: Configure futures contract
        configure_futures(client, symbol, leverage=leverage, margin_type=margin_type)
        
        # Step 1: Cancel any existing orders on this symbol
        cancel_open_orders(client, symbol)
        
        # Step 2: Open the position at current market price
        print(f"\n🤖 [BOT] Opening {side} market order for {symbol} (qty: {qty})...")
        entry_order = client.futures_create_order(
            symbol=symbol.upper(),
            side=side,
            type="MARKET",
            quantity=qty
        )
        print(f"✅ Entry Order Filled! OrderID: {entry_order.get('orderId', 'N/A')}")
        
        # Step 3: Set the Stop Loss bracket
        sl_side = "SELL" if side == "BUY" else "BUY"
        print(f"🤖 [BOT] Placing Stop Loss at ${sl_rounded:,.2f}...")
        client.futures_create_order(
            symbol=symbol.upper(),
            side=sl_side,
            type="STOP_MARKET",
            stopPrice=sl_rounded,
            closePosition=True
        )
        
        # Step 4: Set the Take Profit bracket
        print(f"🤖 [BOT] Placing Take Profit at ${tp_rounded:,.2f}...")
        client.futures_create_order(
            symbol=symbol.upper(),
            side=sl_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=tp_rounded,
            closePosition=True
        )
        print("🎉 Bracket Orders (SL & TP) successfully attached! Position is fully protected.")
        return True
        
    except Exception as e:
        print(f"❌ Error executing live orders: {e}")
        return False

def fetch_live_candles(client, symbol: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch latest historical candles from Binance Futures.
    Uses authenticated client if available, falls back to public API.
    """
    try:
        if client:
            klines = client.futures_klines(symbol=symbol.upper(), interval="15m", limit=limit)
        else:
            import requests
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol.upper()}&interval=15m&limit={limit}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
            
        formatted_data = []
        for candle in klines:
            formatted_data.append([
                pd.to_datetime(candle[0], unit="ms"),
                float(candle[1]),
                float(candle[2]),
                float(candle[3]),
                float(candle[4]),
                float(candle[5])
            ])
            
        df = pd.DataFrame(formatted_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df
        
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return pd.DataFrame()

def run_live_bot(
    symbol: str = "BTCUSDT",
    sweep_mode: str = "single",
    use_trend_filter: bool = True,
    trade_direction: str = "both",
    sltp_mode: str = "structure",
    rr_ratio: float = 2.0,
    fixed_sl_pct: float = 20.0,
    fixed_tp_pct: float = 2.0,
    trigger_modes: list = None,
    trigger_logic: str = "OR",
    risk_per_trade: float = 10.0,
    leverage: int = 10,
    margin_type: str = "ISOLATED",
    poll_interval: int = 30,
    log_callback=None
):
    """
    Main live trading loop.
    Checks Binance every poll_interval seconds for new candle closes.
    
    Parameters:
    - log_callback: Optional callable for Streamlit logging (e.g., st.write).
                    If None, prints to stdout.
    """
    if trigger_modes is None:
        trigger_modes = ["engulfing"]
    
    def log(msg):
        timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        if log_callback:
            log_callback(timestamped)
        print(timestamped)
    
    log(f"🚀 Bot started! Monitoring {symbol} on Binance Futures Testnet")
    log(f"   Sweep: {sweep_mode} | Triggers: {', '.join(trigger_modes)} ({trigger_logic})")
    log(f"   EMA Filter: {'ON' if use_trend_filter else 'OFF'} | Direction: {trade_direction}")
    log(f"   SL/TP: {sltp_mode} | Risk/Trade: ${risk_per_trade}")
    log(f"   Leverage: {leverage}x | Margin: {margin_type}")
    log(f"   Poll Interval: {poll_interval}s")
    log("")
    
    client = connect_binance()
    precision = get_symbol_precision(client, symbol)
    
    # Track the last candle timestamp to avoid duplicate signals
    last_processed_candle = None
    
    # We need enough candles for the 200 EMA + 96 rolling window
    candle_limit = 300
    
    while True:
        try:
            # 1. Fetch live candle data
            df = fetch_live_candles(client, symbol, limit=candle_limit)
            
            if df.empty:
                log("⚠️  Empty candle data. Retrying...")
                time.sleep(10)
                continue
            
            # 2. Run the full strategy state machine
            df_signals = generate_signals(
                df,
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
            
            # 3. Check the last CLOSED candle (index -2, since -1 is the live open candle)
            latest = df_signals.iloc[-2]
            candle_time = latest["timestamp"]
            signal = latest["signal"]
            close_price = latest["close"]
            state = latest["state_log"]
            
            # Skip if we already processed this candle
            if candle_time == last_processed_candle:
                time.sleep(poll_interval)
                continue
            
            last_processed_candle = candle_time
            
            # 4. Log the current state
            log(f"📊 Candle: {candle_time} | Close: ${close_price:,.2f} | State: {state}")
            
            # 5. Check for existing position
            existing_pos = get_open_positions(client, symbol)
            if existing_pos:
                log(f"📌 Active {existing_pos['side']} position: {existing_pos['size']} units @ ${existing_pos['entry_price']:,.2f} (PnL: ${existing_pos['unrealized_pnl']:,.2f})")
                # Don't open new positions if we already have one
                if signal != 0:
                    log(f"⏭️  Signal detected but skipping — already in a {existing_pos['side']} position.")
                time.sleep(poll_interval)
                continue
            
            # 6. Execute if signal fires
            if signal == 1:
                sl_level = latest["sl_level"]
                tp_level = latest["tp_level"]
                risk_distance = close_price - sl_level
                
                if risk_distance > 0:
                    position_size = risk_per_trade / risk_distance
                    log(f"🟢 LONG SIGNAL! Entry: ${close_price:,.2f} | SL: ${sl_level:,.2f} | TP: ${tp_level:,.2f} | Size: {round(position_size, precision['qty_precision'])} | {leverage}x {margin_type}")
                    place_order(client, symbol, "BUY", position_size, sl_level, tp_level, precision, leverage=leverage, margin_type=margin_type)
                else:
                    log("⚠️  Invalid risk distance for LONG. Skipping.")
                    
            elif signal == -1:
                sl_level = latest["sl_level"]
                tp_level = latest["tp_level"]
                risk_distance = sl_level - close_price
                
                if risk_distance > 0:
                    position_size = risk_per_trade / risk_distance
                    log(f"🔴 SHORT SIGNAL! Entry: ${close_price:,.2f} | SL: ${sl_level:,.2f} | TP: ${tp_level:,.2f} | Size: {round(position_size, precision['qty_precision'])} | {leverage}x {margin_type}")
                    place_order(client, symbol, "SELL", position_size, sl_level, tp_level, precision, leverage=leverage, margin_type=margin_type)
                else:
                    log("⚠️  Invalid risk distance for SHORT. Skipping.")
            else:
                log("⏳ No signal on this candle. Scanning continues...")
                
            # 7. Wait before next poll
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            log("🤖 [BOT] Shutting down gracefully. Bye!")
            break
        except Exception as e:
            log(f"❌ Error in live loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_live_bot(
        symbol="BTCUSDT",
        sweep_mode="single",
        use_trend_filter=True,
        trade_direction="both",
        trigger_modes=["engulfing"],
        trigger_logic="OR",
        risk_per_trade=10.0,
        leverage=10,
        margin_type="ISOLATED"
    )
