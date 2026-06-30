import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame, rolling_window: int = 96) -> pd.DataFrame:
    """
    Calculates technical indicators:
    - 24h Low & High: Session extremes
    - Volume 10 MA: Volume moving average
    - 200 EMA: Long-term trend filter
    - TR (True Range) & ATR (Average True Range) for volatility filters
    - Fixed Daily PDH / PDL (Previous Day High / Low) using date groups
    """
    if len(df) < rolling_window:
        return df
        
    df["24h_low"] = df["low"].rolling(window=rolling_window).min().shift(1)
    df["24h_high"] = df["high"].rolling(window=rolling_window).max().shift(1)
    df["vol_ma_10"] = df["volume"].rolling(window=10).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # --- ATR Calculation ---
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    
    df["tr"] = np.maximum(high_low, np.maximum(high_close, low_close))
    df["atr"] = df["tr"].rolling(window=14).mean()
    
    # --- Previous Day High / Low (PDH / PDL) ---
    try:
        # Convert timestamp to date group safely
        dates = pd.to_datetime(df['timestamp']).dt.date
        # Group to find daily high and low
        daily_high = df.groupby(dates)['high'].max()
        daily_low = df.groupby(dates)['low'].min()
        
        # Shift daily results so they are available for the NEXT day
        pdh_series = dates.map(daily_high.shift(1))
        pdl_series = dates.map(daily_low.shift(1))
        
        df['pdh'] = pdh_series
        df['pdl'] = pdl_series
    except Exception:
        df['pdh'] = df['24h_high']
        df['pdl'] = df['24h_low']
        
    # Fill remaining NaNs with 24h equivalents
    df['pdh'] = df['pdh'].fillna(df['24h_high'])
    df['pdl'] = df['pdl'].fillna(df['24h_low'])
    
    return df

def detect_bullish_engulfing(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    prev_open, prev_close = df.loc[i-1, "open"], df.loc[i-1, "close"]
    curr_open, curr_close = df.loc[i, "open"], df.loc[i, "close"]
    prev_was_red = prev_close < prev_open
    curr_was_green = curr_close > curr_open
    if prev_was_red and curr_was_green:
        return (curr_open <= prev_close) and (curr_close >= prev_open)
    return False

def detect_bearish_engulfing(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    prev_open, prev_close = df.loc[i-1, "open"], df.loc[i-1, "close"]
    curr_open, curr_close = df.loc[i, "open"], df.loc[i, "close"]
    prev_was_green = prev_close > prev_open
    curr_was_red = curr_close < curr_open
    if prev_was_green and curr_was_red:
        return (curr_open >= prev_close) and (curr_close <= prev_open)
    return False

def detect_3g_1r_pattern(df: pd.DataFrame, i: int) -> bool:
    if i < 4:
        return False
    c_4 = df.loc[i-4, "close"] > df.loc[i-4, "open"]
    c_3 = df.loc[i-3, "close"] > df.loc[i-3, "open"]
    c_2 = df.loc[i-2, "close"] > df.loc[i-2, "open"]
    c_1 = df.loc[i-1, "close"] < df.loc[i-1, "open"]
    return c_4 and c_3 and c_2 and c_1

def detect_3r_1g_pattern(df: pd.DataFrame, i: int) -> bool:
    if i < 4:
        return False
    c_4 = df.loc[i-4, "close"] < df.loc[i-4, "open"]
    c_3 = df.loc[i-3, "close"] < df.loc[i-3, "open"]
    c_2 = df.loc[i-2, "close"] < df.loc[i-2, "open"]
    c_1 = df.loc[i-1, "close"] > df.loc[i-1, "open"]
    return c_4 and c_3 and c_2 and c_1

def detect_pinbar_hammer(df: pd.DataFrame, i: int, side: str = "bullish") -> bool:
    close_vals = df["close"].values
    open_vals = df["open"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    
    o, h, l, c = open_vals[i], high_vals[i], low_vals[i], close_vals[i]
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return False
    if side == "bullish":
        body_top = max(o, c)
        lower_wick = body_top - l
        return (lower_wick >= (2 * body)) and ((h - body_top) <= (0.15 * candle_range))
    else:
        body_bottom = min(o, c)
        upper_wick = h - body_bottom
        return (upper_wick >= (2 * body)) and ((body_bottom - l) <= (0.15 * candle_range))

def detect_volume_spike(df: pd.DataFrame, i: int, factor: float = 1.5) -> bool:
    vol_ma_vals = df["vol_ma_10"].values
    volume_vals = df["volume"].values
    if pd.isna(vol_ma_vals[i]):
        return False
    return volume_vals[i] >= (factor * vol_ma_vals[i])

def check_triggers(df: pd.DataFrame, i: int, trigger_modes: list, trigger_logic: str, side: str, reclaim_level: float = None) -> tuple:
    """
    Unified trigger checker supporting AND / OR logic.
    - OR mode:  Trade fires if ANY selected trigger is True.
    - AND mode: Trade fires only if ALL selected triggers are True.
    Returns (trigger_fired: bool, entry_price: float).
    """
    close_vals = df["close"].values
    open_vals = df["open"].values
    entry_price = float(close_vals[i])
    
    if not trigger_modes:
        return False, entry_price
    
    results = {}
    
    if "engulfing" in trigger_modes:
        results["engulfing"] = detect_bullish_engulfing(df, i) if side == "long" else detect_bearish_engulfing(df, i)
    
    if "pullback" in trigger_modes:
        results["pullback"] = detect_3g_1r_pattern(df, i) if side == "long" else detect_3r_1g_pattern(df, i)
    
    if "pinbar" in trigger_modes:
        results["pinbar"] = detect_pinbar_hammer(df, i, side="bullish" if side == "long" else "bearish")
    
    if "volume" in trigger_modes:
        vol_ma_vals = df["vol_ma_10"].values
        volume_vals = df["volume"].values
        if pd.isna(vol_ma_vals[i]):
            results["volume"] = False
        else:
            vol_spike = volume_vals[i] >= (1.5 * vol_ma_vals[i])
            if side == "long":
                results["volume"] = (close_vals[i] > open_vals[i]) and vol_spike
            else:
                results["volume"] = (close_vals[i] < open_vals[i]) and vol_spike
    
    if "immediate" in trigger_modes:
        results["immediate"] = True
    
    if not results:
        return False, entry_price
    
    # Apply AND or OR logic
    if trigger_logic == "AND":
        trigger_fired = all(results.values())
    else:
        trigger_fired = any(results.values())
    
    # Use reclaim price when "immediate" is the sole firing trigger
    if trigger_fired and "immediate" in results and reclaim_level is not None:
        non_immediate = {k: v for k, v in results.items() if k != "immediate"}
        if not non_immediate or (trigger_logic == "OR" and not any(non_immediate.values())):
            entry_price = reclaim_level
    
    return trigger_fired, entry_price

def generate_signals(
    df: pd.DataFrame, 
    sweep_mode: str = "single", 
    use_trend_filter: bool = True,
    trade_direction: str = "both",
    sltp_mode: str = "structure",
    rr_ratio: float = 2.0,
    fixed_sl_pct: float = 20.0,
    fixed_tp_pct: float = 2.0,
    rolling_window: int = 96,
    trigger_modes: list = ["engulfing"],
    trigger_logic: str = "OR",
    use_fixed_session_levels: bool = False,
    use_atr_penetration: bool = False,
    atr_penetration_factor: float = 0.25,
    use_atr_tp: bool = False,
    atr_tp_factor: float = 2.5,
    use_time_filter: bool = False,
    enable_breakout_reversal: bool = False,
    breakout_sl_pct: float = 20.0,
    breakout_only: bool = False,
    enable_quad_breakout: bool = False,
    enable_double_breakout: bool = False,
    enable_single_breakout: bool = False,
    custom_tp_dict: dict = None
) -> pd.DataFrame:
    """
    Unified signal engine using NumPy arrays for 50x speed.
    """
    # Calculate indicators first so they are not recalculated recursively
    if "24h_low" not in df.columns or "24h_high" not in df.columns or "vol_ma_10" not in df.columns or "ema_200" not in df.columns or "atr" not in df.columns:
        df = calculate_indicators(df, rolling_window=rolling_window)
        
    if sweep_mode in ["both", "double_triple", "double_triple_quad"]:
        df_double = generate_signals(
            df=df.copy(),
            sweep_mode="double",
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
            breakout_only=breakout_only,
            enable_quad_breakout=enable_quad_breakout,
            enable_double_breakout=enable_double_breakout,
            enable_single_breakout=enable_single_breakout,
            custom_tp_dict=custom_tp_dict
        )
        df_triple = generate_signals(
            df=df.copy(),
            sweep_mode="triple",
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
            breakout_only=breakout_only,
            enable_quad_breakout=enable_quad_breakout,
            enable_double_breakout=enable_double_breakout,
            enable_single_breakout=enable_single_breakout,
            custom_tp_dict=custom_tp_dict
        )
        
        sig_d = df_double["signal"].values
        sig_t = df_triple["signal"].values
        sl_d = df_double["sl_level"].values
        sl_t = df_triple["sl_level"].values
        tp_d = df_double["tp_level"].values
        tp_t = df_triple["tp_level"].values
        log_d = df_double["state_log"].values
        log_t = df_triple["state_log"].values
        
        signals = np.zeros(len(df))
        sl_levels = np.full(len(df), np.nan)
        tp_levels = np.full(len(df), np.nan)
        state_logs = [""] * len(df)
        
        if sweep_mode == "double_triple_quad":
            df_quad = generate_signals(
                df=df.copy(),
                sweep_mode="quadruple",
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
                breakout_only=breakout_only,
                enable_quad_breakout=enable_quad_breakout,
                enable_double_breakout=enable_double_breakout,
                enable_single_breakout=enable_single_breakout,
                custom_tp_dict=custom_tp_dict
            )
            sig_q = df_quad["signal"].values
            sl_q = df_quad["sl_level"].values
            tp_q = df_quad["tp_level"].values
            log_q = df_quad["state_log"].values
            
            for idx in range(len(df)):
                if sig_q[idx] != 0:
                    signals[idx] = sig_q[idx]
                    sl_levels[idx] = sl_q[idx]
                    tp_levels[idx] = tp_q[idx]
                    state_logs[idx] = f"QUAD | {log_q[idx]}"
                elif sig_t[idx] != 0:
                    signals[idx] = sig_t[idx]
                    sl_levels[idx] = sl_t[idx]
                    tp_levels[idx] = tp_t[idx]
                    state_logs[idx] = f"TRIPLE | {log_t[idx]}"
                elif sig_d[idx] != 0:
                    signals[idx] = sig_d[idx]
                    sl_levels[idx] = sl_d[idx]
                    tp_levels[idx] = tp_d[idx]
                    state_logs[idx] = f"DOUBLE | {log_d[idx]}"
                else:
                    state_logs[idx] = f"D: {log_d[idx]} | T: {log_t[idx]} | Q: {log_q[idx]}"
        else:
            for idx in range(len(df)):
                if sig_t[idx] != 0:
                    signals[idx] = sig_t[idx]
                    sl_levels[idx] = sl_t[idx]
                    tp_levels[idx] = tp_t[idx]
                    state_logs[idx] = f"TRIPLE | {log_t[idx]}"
                elif sig_d[idx] != 0:
                    signals[idx] = sig_d[idx]
                    sl_levels[idx] = sl_d[idx]
                    tp_levels[idx] = tp_d[idx]
                    state_logs[idx] = f"DOUBLE | {log_d[idx]}"
                else:
                    state_logs[idx] = f"D: {log_d[idx]} | T: {log_t[idx]}"
                
        df_combined = df.copy()
        df_combined["signal"] = signals
        df_combined["sl_level"] = sl_levels
        df_combined["tp_level"] = tp_levels
        df_combined["state_log"] = state_logs
        return df_combined

    df["signal"] = 0
    df["sl_level"] = np.nan
    df["tp_level"] = np.nan
    df["state_log"] = "SEEK_L1 | SEEK_H1"
        
    enable_longs = trade_direction in ["both", "long"]
    enable_shorts = trade_direction in ["both", "short"]
    
    long_state = "SEEK_L1"
    long_sweep_low = None
    long_trigger_level = None
    l1_low, l2_low, l3_low, l4_low, l5_low = None, None, None, None, None
    
    short_state = "SEEK_H1"
    short_sweep_high = None
    short_trigger_level = None
    h1_high, h2_high, h3_high, h4_high, h5_high = None, None, None, None, None
    
    # Extract NumPy arrays for fast iteration without df.iloc overhead!
    close_arr = df["close"].values
    low_arr = df["low"].values
    high_arr = df["high"].values
    low_24h_arr = df["24h_low"].values
    high_24h_arr = df["24h_high"].values
    ema_200_arr = df["ema_200"].values
    atr_arr = df["atr"].values
    
    pdl_arr = df["pdl"].values if "pdl" in df.columns else low_24h_arr
    pdh_arr = df["pdh"].values if "pdh" in df.columns else high_24h_arr
    
    timestamp_vals = pd.to_datetime(df["timestamp"])
    
    # Extract custom TP values from custom_tp_dict or use defaults
    double_rr = rr_ratio
    triple_rr = rr_ratio
    quad_rr = rr_ratio
    single_breakout_rr = rr_ratio
    double_breakout_rr = rr_ratio
    triple_breakout_rr = rr_ratio
    quad_breakout_rr = rr_ratio
    
    double_fixed_tp = fixed_tp_pct
    triple_fixed_tp = fixed_tp_pct
    quad_fixed_tp = fixed_tp_pct
    single_breakout_fixed_tp = fixed_tp_pct
    double_breakout_fixed_tp = fixed_tp_pct
    triple_breakout_fixed_tp = fixed_tp_pct
    quad_breakout_fixed_tp = fixed_tp_pct
    
    double_atr_tp = atr_tp_factor
    triple_atr_tp = atr_tp_factor
    quad_atr_tp = atr_tp_factor
    single_breakout_atr_tp = atr_tp_factor
    double_breakout_atr_tp = atr_tp_factor
    triple_breakout_atr_tp = atr_tp_factor
    quad_breakout_atr_tp = atr_tp_factor
    
    if custom_tp_dict:
        double_rr = custom_tp_dict.get("double_rr", rr_ratio)
        triple_rr = custom_tp_dict.get("triple_rr", rr_ratio)
        quad_rr = custom_tp_dict.get("quad_rr", rr_ratio)
        single_breakout_rr = custom_tp_dict.get("single_breakout_rr", rr_ratio)
        double_breakout_rr = custom_tp_dict.get("double_breakout_rr", rr_ratio)
        triple_breakout_rr = custom_tp_dict.get("triple_breakout_rr", rr_ratio)
        quad_breakout_rr = custom_tp_dict.get("quad_breakout_rr", rr_ratio)
        
        double_fixed_tp = custom_tp_dict.get("double_fixed_tp", fixed_tp_pct)
        triple_fixed_tp = custom_tp_dict.get("triple_fixed_tp", fixed_tp_pct)
        quad_fixed_tp = custom_tp_dict.get("quad_fixed_tp", fixed_tp_pct)
        single_breakout_fixed_tp = custom_tp_dict.get("single_breakout_fixed_tp", fixed_tp_pct)
        double_breakout_fixed_tp = custom_tp_dict.get("double_breakout_fixed_tp", fixed_tp_pct)
        triple_breakout_fixed_tp = custom_tp_dict.get("triple_breakout_fixed_tp", fixed_tp_pct)
        quad_breakout_fixed_tp = custom_tp_dict.get("quad_breakout_fixed_tp", fixed_tp_pct)
        
        double_atr_tp = custom_tp_dict.get("double_atr_tp", atr_tp_factor)
        triple_atr_tp = custom_tp_dict.get("triple_atr_tp", atr_tp_factor)
        quad_atr_tp = custom_tp_dict.get("quad_atr_tp", atr_tp_factor)
        single_breakout_atr_tp = custom_tp_dict.get("single_breakout_atr_tp", atr_tp_factor)
        double_breakout_atr_tp = custom_tp_dict.get("double_breakout_atr_tp", atr_tp_factor)
        triple_breakout_atr_tp = custom_tp_dict.get("triple_breakout_atr_tp", atr_tp_factor)
        quad_breakout_atr_tp = custom_tp_dict.get("quad_breakout_atr_tp", atr_tp_factor)
        
    # Buffer lists to write results quickly
    signals = [0] * len(df)
    sl_levels = [np.nan] * len(df)
    tp_levels = [np.nan] * len(df)
    state_logs = ["SEEK_L1 | SEEK_H1"] * len(df)
    
    for i in range(rolling_window, len(df)):
        curr_close = close_arr[i]
        curr_low = low_arr[i]
        curr_high = high_arr[i]
        curr_24h_low = low_24h_arr[i]
        curr_24h_high = high_24h_arr[i]
        
        # Determine levels to seek based on fixed session levels toggle
        support_level = pdl_arr[i] if use_fixed_session_levels else curr_24h_low
        resistance_level = pdh_arr[i] if use_fixed_session_levels else curr_24h_high
        
        # Get ATR value safely
        atr_val = atr_arr[i] if not pd.isna(atr_arr[i]) else 0
        
        # -------------------------------------------------------------
        # PART A: LONG STATE MACHINE (Sweeping Support)
        # -------------------------------------------------------------
        if enable_longs:
            if sweep_mode == "single":
                if long_state == "SEEK_L1":
                    # Volatility-adjusted sweep boundary
                    sweep_boundary = support_level - (atr_penetration_factor * atr_val) if use_atr_penetration else support_level
                    if curr_low < sweep_boundary:
                        long_trigger_level = support_level
                        long_sweep_low = curr_low
                        long_state = "L1_SWEPT"
                elif long_state == "L1_SWEPT":
                    if curr_low < long_sweep_low:
                        long_sweep_low = curr_low
                    if curr_close > long_trigger_level:
                        long_state = "TESTING_L1"
                elif long_state == "TESTING_L1":
                    if curr_low < long_sweep_low:
                        if enable_single_breakout:
                            # 1-Sweep Support failed! Enter a breakout SHORT trade!
                            signals[i] = -1
                            sl = curr_close * (1 + breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close - (single_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = sl - curr_close
                                tp = curr_close - (single_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 - single_breakout_fixed_tp / 100.0)
                            sl_levels[i] = sl
                            tp_levels[i] = tp

                        long_state = "SEEK_L1"
                        long_sweep_low = None
                        long_trigger_level = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close <= ema_200_arr[i]:
                        continue
                    
                    # Time-of-Day session filter check (London: 7-10 UTC, NY: 12-16 UTC)
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    # Entry Triggers (AND/OR logic)
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "long", reclaim_level=long_trigger_level)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = 1
                            if use_atr_tp:
                                sl = long_sweep_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price + (atr_tp_factor * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = long_sweep_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                risk = entry_price - sl
                                tp = entry_price + (rr_ratio * risk)
                            else:
                                sl = entry_price * (1 - fixed_sl_pct / 100.0)
                                tp = entry_price * (1 + fixed_tp_pct / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        long_state = "SEEK_L1"
                        long_sweep_low = None
                        long_trigger_level = None
            
            elif sweep_mode == "double":
                if long_state == "SEEK_L1":
                    sweep_boundary = support_level - (atr_penetration_factor * atr_val) if use_atr_penetration else support_level
                    if curr_low < sweep_boundary:
                        l1_low = support_level
                        l2_low = curr_low
                        long_state = "L1_SWEPT"
                elif long_state == "L1_SWEPT":
                    if curr_close > l1_low:
                        long_state = "WAIT_FOR_L3"
                    if curr_low < l2_low:
                        l2_low = curr_low
                elif long_state == "WAIT_FOR_L3":
                    if curr_low < l2_low:
                        l3_low = curr_low
                        long_state = "L3_SWEPT"
                elif long_state == "L3_SWEPT":
                    if curr_close > l2_low:
                        long_state = "TESTING_L3"
                    if curr_low < l3_low:
                        l3_low = curr_low
                elif long_state == "TESTING_L3":
                    if curr_low < l3_low:
                        if enable_double_breakout:
                            # 2-Sweep Support failed! Enter a breakout SHORT trade!
                            signals[i] = -1
                            sl = curr_close * (1 + breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close - (double_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = sl - curr_close
                                tp = curr_close - (double_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 - double_breakout_fixed_tp / 100.0)
                            sl_levels[i] = sl
                            tp_levels[i] = tp

                        long_state = "SEEK_L1"
                        l1_low = None
                        l2_low = None
                        l3_low = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close <= ema_200_arr[i]:
                        continue
                    
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "long", reclaim_level=l2_low)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = 1
                            if use_atr_tp:
                                sl = l3_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price + (double_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = l3_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                risk = entry_price - sl
                                tp = entry_price + (double_rr * risk)
                            else:
                                sl = entry_price * (1 - fixed_sl_pct / 100.0)
                                tp = entry_price * (1 + double_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        long_state = "SEEK_L1"
                        l1_low = None
                        l2_low = None
                        l3_low = None
            
            elif sweep_mode == "triple":
                if long_state == "SEEK_L1":
                    sweep_boundary = support_level - (atr_penetration_factor * atr_val) if use_atr_penetration else support_level
                    if curr_low < sweep_boundary:
                        l1_low = support_level
                        l2_low = curr_low
                        long_state = "L1_SWEPT"
                elif long_state == "L1_SWEPT":
                    if curr_close > l1_low:
                        long_state = "WAIT_FOR_L3"
                    if curr_low < l2_low:
                        l2_low = curr_low
                elif long_state == "WAIT_FOR_L3":
                    if curr_low < l2_low:
                        l3_low = curr_low
                        long_state = "L3_SWEPT"
                elif long_state == "L3_SWEPT":
                    if curr_close > l2_low:
                        long_state = "WAIT_FOR_L5"
                    if curr_low < l3_low:
                        l3_low = curr_low
                elif long_state == "WAIT_FOR_L5":
                    if curr_low < l3_low:
                        l4_low = curr_low
                        long_state = "L5_SWEPT"
                elif long_state == "L5_SWEPT":
                    if curr_close > l3_low:
                        long_state = "TESTING_L5"
                    if curr_low < l4_low:
                        l4_low = curr_low
                elif long_state == "TESTING_L5":
                    if curr_low < l4_low:
                        if enable_breakout_reversal:
                            # 3-Sweep Support failed! Enter a breakout SHORT trade with user SL %!
                            signals[i] = -1
                            sl = curr_close * (1 + breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close - (triple_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = sl - curr_close
                                tp = curr_close - (triple_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 - triple_breakout_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        long_state = "SEEK_L1"
                        l1_low = None
                        l2_low = None
                        l3_low = None
                        l4_low = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close <= ema_200_arr[i]:
                        continue
                        
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "long", reclaim_level=l3_low)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = 1
                            if use_atr_tp:
                                sl = l4_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price + (triple_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = l4_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                risk = entry_price - sl
                                tp = entry_price + (triple_rr * risk)
                            else:
                                sl = entry_price * (1 - fixed_sl_pct / 100.0)
                                tp = entry_price * (1 + triple_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        long_state = "SEEK_L1"
                        l1_low = None
                        l2_low = None
                        l3_low = None
                        l4_low = None

            elif sweep_mode == "quadruple":
                if long_state == "SEEK_L1":
                    sweep_boundary = support_level - (atr_penetration_factor * atr_val) if use_atr_penetration else support_level
                    if curr_low < sweep_boundary:
                        l1_low = support_level
                        l2_low = curr_low
                        long_state = "L1_SWEPT"
                elif long_state == "L1_SWEPT":
                    if curr_close > l1_low:
                        long_state = "WAIT_FOR_L3"
                    if curr_low < l2_low:
                        l2_low = curr_low
                elif long_state == "WAIT_FOR_L3":
                    if curr_low < l2_low:
                        l3_low = curr_low
                        long_state = "L3_SWEPT"
                elif long_state == "L3_SWEPT":
                    if curr_close > l2_low:
                        long_state = "WAIT_FOR_L5"
                    if curr_low < l3_low:
                        l3_low = curr_low
                elif long_state == "WAIT_FOR_L5":
                    if curr_low < l3_low:
                        l4_low = curr_low
                        long_state = "L5_SWEPT"
                elif long_state == "L5_SWEPT":
                    if curr_close > l3_low:
                        long_state = "WAIT_FOR_L7"
                    if curr_low < l4_low:
                        l4_low = curr_low
                elif long_state == "WAIT_FOR_L7":
                    if curr_low < l4_low:
                        l5_low = curr_low
                        long_state = "L7_SWEPT"
                elif long_state == "L7_SWEPT":
                    if curr_close > l4_low:
                        long_state = "TESTING_L7"
                    if curr_low < l5_low:
                        l5_low = curr_low
                elif long_state == "TESTING_L7":
                    if curr_low < l5_low:
                        if enable_quad_breakout:
                            # 4-Sweep Support failed! Enter a breakout SHORT trade with user SL %!
                            signals[i] = -1
                            sl = curr_close * (1 + breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close - (quad_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = sl - curr_close
                                tp = curr_close - (quad_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 - quad_breakout_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        long_state = "SEEK_L1"
                        l1_low = None
                        l2_low = None
                        l3_low = None
                        l4_low = None
                        l5_low = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close <= ema_200_arr[i]:
                        continue
                        
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "long", reclaim_level=l4_low)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = 1
                            if use_atr_tp:
                                sl = l5_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price + (quad_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = l5_low
                                if sl >= entry_price: sl = entry_price * 0.99
                                risk = entry_price - sl
                                tp = entry_price + (quad_rr * risk)
                            else:
                                sl = entry_price * (1 - fixed_sl_pct / 100.0)
                                tp = entry_price * (1 + quad_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        long_state = "SEEK_L1"
                        l1_low = None
                        l2_low = None
                        l3_low = None
                        l4_low = None
                        l5_low = None

        # -------------------------------------------------------------
        # PART B: SHORT STATE MACHINE (Sweeping Resistance)
        # -------------------------------------------------------------
        if enable_shorts:
            if sweep_mode == "single":
                if short_state == "SEEK_H1":
                    # Volatility-adjusted sweep boundary
                    sweep_boundary = resistance_level + (atr_penetration_factor * atr_val) if use_atr_penetration else resistance_level
                    if curr_high > sweep_boundary:
                        short_trigger_level = resistance_level
                        short_sweep_high = curr_high
                        short_state = "H1_SWEPT"
                elif short_state == "H1_SWEPT":
                    if curr_high > short_sweep_high:
                        short_sweep_high = curr_high
                    if curr_close < short_trigger_level:
                        short_state = "TESTING_H1"
                elif short_state == "TESTING_H1":
                    if curr_high > short_sweep_high:
                        if enable_single_breakout:
                            # 1-Sweep Resistance failed! Enter a breakout LONG trade!
                            signals[i] = 1
                            sl = curr_close * (1 - breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close + (single_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = curr_close - sl
                                tp = curr_close + (single_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 + single_breakout_fixed_tp / 100.0)
                            sl_levels[i] = sl
                            tp_levels[i] = tp

                        short_state = "SEEK_H1"
                        short_sweep_high = None
                        short_trigger_level = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close >= ema_200_arr[i]:
                        continue
                    
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "short", reclaim_level=short_trigger_level)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = -1
                            if use_atr_tp:
                                sl = short_sweep_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price - (atr_tp_factor * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = short_sweep_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                risk = sl - entry_price
                                tp = entry_price - (rr_ratio * risk)
                            else:
                                sl = entry_price * (1 + fixed_sl_pct / 100.0)
                                tp = entry_price * (1 - fixed_tp_pct / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        short_state = "SEEK_H1"
                        short_sweep_high = None
                        short_trigger_level = None
                    
            elif sweep_mode == "double":
                if short_state == "SEEK_H1":
                    sweep_boundary = resistance_level + (atr_penetration_factor * atr_val) if use_atr_penetration else resistance_level
                    if curr_high > sweep_boundary:
                        h1_high = resistance_level
                        h2_high = curr_high
                        short_state = "H1_SWEPT"
                elif short_state == "H1_SWEPT":
                    if curr_close < h1_high:
                        short_state = "WAIT_FOR_H3"
                    if curr_high > h2_high:
                        h2_high = curr_high
                elif short_state == "WAIT_FOR_H3":
                    if curr_high > h2_high:
                        h3_high = curr_high
                        short_state = "H3_SWEPT"
                elif short_state == "H3_SWEPT":
                    if curr_close < h2_high:
                        short_state = "TESTING_H3"
                    if curr_high > h3_high:
                        h3_high = curr_high
                elif short_state == "TESTING_H3":
                    if curr_high > h3_high:
                        if enable_double_breakout:
                            # 2-Sweep Resistance failed! Enter a breakout LONG trade!
                            signals[i] = 1
                            sl = curr_close * (1 - breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close + (double_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = curr_close - sl
                                tp = curr_close + (double_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 + double_breakout_fixed_tp / 100.0)
                            sl_levels[i] = sl
                            tp_levels[i] = tp

                        short_state = "SEEK_H1"
                        h1_high = None
                        h2_high = None
                        h3_high = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close >= ema_200_arr[i]:
                        continue
                    
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "short", reclaim_level=h2_high)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = -1
                            if use_atr_tp:
                                sl = h3_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price - (double_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = h3_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                risk = sl - entry_price
                                tp = entry_price - (double_rr * risk)
                            else:
                                sl = entry_price * (1 + fixed_sl_pct / 100.0)
                                tp = entry_price * (1 - double_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        short_state = "SEEK_H1"
                        h1_high = None
                        h2_high = None
                        h3_high = None
            
            elif sweep_mode == "triple":
                if short_state == "SEEK_H1":
                    sweep_boundary = resistance_level + (atr_penetration_factor * atr_val) if use_atr_penetration else resistance_level
                    if curr_high > sweep_boundary:
                        h1_high = resistance_level
                        h2_high = curr_high
                        short_state = "H1_SWEPT"
                elif short_state == "H1_SWEPT":
                    if curr_close < h1_high:
                        short_state = "WAIT_FOR_H3"
                    if curr_high > h2_high:
                        h2_high = curr_high
                elif short_state == "WAIT_FOR_H3":
                    if curr_high > h2_high:
                        h3_high = curr_high
                        short_state = "H3_SWEPT"
                elif short_state == "H3_SWEPT":
                    if curr_close < h2_high:
                        short_state = "WAIT_FOR_H5"
                    if curr_high > h3_high:
                        h3_high = curr_high
                elif short_state == "WAIT_FOR_H5":
                    if curr_high > h3_high:
                        h4_high = curr_high
                        short_state = "H5_SWEPT"
                elif short_state == "H5_SWEPT":
                    if curr_close < h3_high:
                        short_state = "TESTING_H5"
                    if curr_high > h4_high:
                        h4_high = curr_high
                elif short_state == "TESTING_H5":
                    if curr_high > h4_high:
                        if enable_breakout_reversal:
                            # 3-Sweep Resistance failed! Enter a breakout LONG trade with user SL %!
                            signals[i] = 1
                            sl = curr_close * (1 - breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close + (triple_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = curr_close - sl
                                tp = curr_close + (triple_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 + triple_breakout_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        short_state = "SEEK_H1"
                        h1_high = None
                        h2_high = None
                        h3_high = None
                        h4_high = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close >= ema_200_arr[i]:
                        continue
                    
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "short", reclaim_level=h3_high)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = -1
                            if use_atr_tp:
                                sl = h4_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price - (triple_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = h4_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                risk = sl - entry_price
                                tp = entry_price - (triple_rr * risk)
                            else:
                                sl = entry_price * (1 + fixed_sl_pct / 100.0)
                                tp = entry_price * (1 - triple_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        short_state = "SEEK_H1"
                        h1_high = None
                        h2_high = None
                        h3_high = None
                        h4_high = None

            elif sweep_mode == "quadruple":
                if short_state == "SEEK_H1":
                    sweep_boundary = resistance_level + (atr_penetration_factor * atr_val) if use_atr_penetration else resistance_level
                    if curr_high > sweep_boundary:
                        h1_high = resistance_level
                        h2_high = curr_high
                        short_state = "H1_SWEPT"
                elif short_state == "H1_SWEPT":
                    if curr_close < h1_high:
                        short_state = "WAIT_FOR_H3"
                    if curr_high > h2_high:
                        h2_high = curr_high
                elif short_state == "WAIT_FOR_H3":
                    if curr_high > h2_high:
                        h3_high = curr_high
                        short_state = "H3_SWEPT"
                elif short_state == "H3_SWEPT":
                    if curr_close < h2_high:
                        short_state = "WAIT_FOR_H5"
                    if curr_high > h3_high:
                        h3_high = curr_high
                elif short_state == "WAIT_FOR_H5":
                    if curr_high > h3_high:
                        h4_high = curr_high
                        short_state = "H5_SWEPT"
                elif short_state == "H5_SWEPT":
                    if curr_close < h3_high:
                        short_state = "WAIT_FOR_H7"
                    if curr_high > h4_high:
                        h4_high = curr_high
                elif short_state == "WAIT_FOR_H7":
                    if curr_high > h4_high:
                        h5_high = curr_high
                        short_state = "H7_SWEPT"
                elif short_state == "H7_SWEPT":
                    if curr_close < h4_high:
                        short_state = "TESTING_H7"
                    if curr_high > h5_high:
                        h5_high = curr_high
                elif short_state == "TESTING_H7":
                    if curr_high > h5_high:
                        if enable_quad_breakout:
                            # 4-Sweep Resistance failed! Enter a breakout LONG trade with user SL %!
                            signals[i] = 1
                            sl = curr_close * (1 - breakout_sl_pct / 100.0)
                            if use_atr_tp:
                                atr_val_fallback = atr_val if atr_val > 0 else (curr_close * 0.01)
                                tp = curr_close + (quad_breakout_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                risk = curr_close - sl
                                tp = curr_close + (quad_breakout_rr * risk)
                            else:
                                tp = curr_close * (1 + quad_breakout_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        short_state = "SEEK_H1"
                        h1_high = None
                        h2_high = None
                        h3_high = None
                        h4_high = None
                        h5_high = None
                        continue
                    
                    # Trend filter check
                    if use_trend_filter and curr_close >= ema_200_arr[i]:
                        continue
                    
                    # Time-of-Day session filter check
                    if use_time_filter:
                        trade_hour = timestamp_vals.iloc[i].hour
                        is_active_session = (7 <= trade_hour <= 10) or (12 <= trade_hour <= 16)
                        if not is_active_session:
                            continue
                        
                    trigger_fired, entry_price = check_triggers(df, i, trigger_modes, trigger_logic, "short", reclaim_level=h4_high)
                        
                    if trigger_fired:
                        if not breakout_only:
                            signals[i] = -1
                            if use_atr_tp:
                                sl = h5_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                atr_val_fallback = atr_val if atr_val > 0 else (entry_price * 0.01)
                                tp = entry_price - (quad_atr_tp * atr_val_fallback)
                            elif sltp_mode == "structure":
                                sl = h5_high
                                if sl <= entry_price: sl = entry_price * 1.01
                                risk = sl - entry_price
                                tp = entry_price - (quad_rr * risk)
                            else:
                                sl = entry_price * (1 + fixed_sl_pct / 100.0)
                                tp = entry_price * (1 - quad_fixed_tp / 100.0)
                                
                            sl_levels[i] = sl
                            tp_levels[i] = tp
                        
                        short_state = "SEEK_H1"
                        h1_high = None
                        h2_high = None
                        h3_high = None
                        h4_high = None
                        h5_high = None
 
        state_logs[i] = f"{long_state if enable_longs else 'OFF'} | {short_state if enable_shorts else 'OFF'}"
        
    df["signal"] = signals
    df["sl_level"] = sl_levels
    df["tp_level"] = tp_levels
    df["state_log"] = state_logs
    
    return df
