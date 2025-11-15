# backtest.py (patched)
import os
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import time

# =========================
# SHARED CONFIG (ENV-driven) - match trading_bot.py
# =========================
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kucoinfutures")
SYMBOL = os.getenv("SYMBOL", "DOGE/USDT:USDT")
TIMEFRAME_ENTRY = os.getenv("ENTRY_TF", "1h")
TIMEFRAME_FILTER = os.getenv("HTF", "4h")
DAYS_BACK = int(os.getenv("DAYS_BACK", "180"))

TOTAL_PORTFOLIO_CAPITAL = float(os.getenv("TOTAL_PORTFOLIO_CAPITAL", "10000"))
PER_COIN_ALLOCATION = float(os.getenv("PER_COIN_ALLOCATION", "0.20"))
INITIAL_CAPITAL = TOTAL_PORTFOLIO_CAPITAL * PER_COIN_ALLOCATION

RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.02"))
RR_FIXED = float(os.getenv("RR_FIXED", "5.0"))
DYNAMIC_RR = os.getenv("DYNAMIC_RR", "true").lower() == "true"
MIN_RR = float(os.getenv("MIN_RR", "4.0"))
MAX_RR = float(os.getenv("MAX_RR", "6.0"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.5"))
USE_ATR_STOPS = os.getenv("USE_ATR_STOPS", "true").lower() == "true"
USE_H1_FILTER = os.getenv("USE_H1_FILTER", "true").lower() == "true"

USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "false").lower() == "true"
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MIN_RATIO = float(os.getenv("VOL_MIN_RATIO", "0.5"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "25"))
BIAS_CONFIRM_BEAR = int(os.getenv("BIAS_CONFIRM_BEAR", "2"))
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "0.0"))

MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.20"))
MAX_TRADE_SIZE = float(os.getenv("MAX_TRADE_SIZE", "100000"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.0006"))
INCLUDE_FUNDING = os.getenv("INCLUDE_FUNDING", "true").lower() == "true"

FUNDING_INTERVAL_HOURS = int(os.getenv("FUNDING_INTERVAL_HOURS", "8"))
LIQUIDATION_PENALTY_RATE = float(os.getenv("LIQUIDATION_PENALTY_RATE", "0.005"))
LEVERAGE = float(os.getenv("LEVERAGE", "1.0"))

TRADE_CSV_FILENAME = f"{SYMBOL.replace('/', '_').replace(':','_')}_FUTURES_backtest_trades.csv"
DEBUG_COMPARE = os.getenv("DEBUG_COMPARE", "false").lower() == "true"
DEBUG_CSV = os.getenv("DEBUG_CSV", "debug_signals_bt.csv")

# =========================
# HELPERS (same as deploy)
# =========================
def get_exchange():
    return getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

def timeframe_to_ms(tf: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    num = int(''.join([c for c in tf if c.isdigit()]))
    unit = ''.join([c for c in tf if c.isalpha()])
    return num * units[unit]

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1500, pause=0.12):
    tf_ms = timeframe_to_ms(timeframe)
    out = []
    cursor = since_ms
    last_ts = None
    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except ccxt.RateLimitExceeded:
            time.sleep(1.0); continue
        except Exception as e:
            print(f"fetch_ohlcv error @ {datetime.utcfromtimestamp(cursor/1000)}: {e}")
            break
        if not batch:
            break
        out.extend(batch)
        newest = batch[-1][0]
        if last_ts is not None and newest <= last_ts:
            cursor += tf_ms
        else:
            cursor = newest + tf_ms
        last_ts = newest
        if newest >= until_ms - tf_ms:
            break
        if pause: time.sleep(pause)

    if not out:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    seen = {row[0]: row for row in out}
    rows = [seen[k] for k in sorted(seen.keys()) if since_ms <= k <= until_ms]
    df = pd.DataFrame(rows, columns=["timestamp","Open","High","Low","Close","Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()[["Open","High","Low","Close","Volume"]]

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_futures_position_size(price, sl, capital, risk_percent, max_trade_size):
    risk_per_trade = capital * risk_percent
    risk_per_contract = abs(price - sl)
    max_by_risk = (risk_per_trade / risk_per_contract) if risk_per_contract > 0 else 0
    max_by_capital = capital / price
    size_base = min(max_by_risk, max_by_capital, max_trade_size / price)
    return max(size_base, 0)

# amount_to_precision (match deploy)
def amount_to_precision(exchange, symbol, amt):
    try:
        return float(exchange.amount_to_precision(symbol, amt))
    except Exception:
        return float(round(amt, 8))

# funding helpers (match deploy)
def fetch_funding_history(exchange, symbol, since_ms, until_ms):
    rates = []
    cursor = since_ms
    while cursor < until_ms:
        try:
            page = exchange.fetchFundingRateHistory(symbol, since=cursor, limit=1000)
        except Exception as e:
            print(f"Funding history fetch error (fallback to none): {e}")
            break
        if not page:
            break
        rates.extend(page)
        newest = page[-1].get('timestamp') or page[-1].get('ts')
        if newest is None or newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(0.1)

    if not rates:
        return pd.DataFrame(columns=["timestamp","rate"])

    df = pd.DataFrame([{"timestamp": r['timestamp'], "rate": float(r.get('fundingRate', r.get('info', {}).get('fundingRate', 0.0)))} for r in rates])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df.sort_index()

def align_funding_schedule(h1_index, funding_df):
    if funding_df is None or funding_df.empty:
        return pd.Series(0.0, index=h1_index)
    funding_series = pd.Series(0.0, index=h1_index)
    for ts, row in funding_df.iterrows():
        idx = funding_series.index.searchsorted(ts)
        if idx < len(funding_series):
            funding_series.iloc[idx] = row["rate"]
    return funding_series

def fetch_leverage_tiers(exchange, symbol):
    try:
        tiers = exchange.fetchLeverageTiers([symbol])[symbol]
        out = []
        for t in tiers:
            out.append({
                "floor": float(t.get("minNotional", t.get("tierFloor", 0.0))),
                "cap": float(t.get("maxNotional", t.get("tierCap", float('inf')))),
                "mmr": float(t.get("maintenanceMarginRate", t.get("maintenanceRate", 0.005)))
            })
        out.sort(key=lambda x: x["floor"])
        return out
    except Exception as e:
        print(f"Leverage tiers fetch failed (using default mmr 0.005): {e}")
        return [{"floor": 0.0, "cap": float('inf'), "mmr": 0.005}]

def get_mmr_for_notional(tiers, notional):
    for t in tiers:
        if t["floor"] <= notional <= t["cap"]:
            return t["mmr"]
    return tiers[-1]["mmr"]

def estimate_liquidation_bounds(entry_price, notional, position_side, mmr, leverage=1.0):
    imr = 1.0 / leverage
    frac = max(imr - mmr, 0.0)
    if position_side == 1:
        liq_price = entry_price * (1.0 - frac)
        return liq_price, None
    else:
        liq_price = entry_price * (1.0 + frac)
        return None, liq_price

# =========================
# FETCH DATA
# =========================
exchange = get_exchange()

now_utc = datetime.now(timezone.utc)
since_dt = now_utc - timedelta(days=DAYS_BACK)
until_dt = now_utc
since_ms = int(since_dt.timestamp() * 1000)
until_ms = int(until_dt.timestamp() * 1000)

print(f"⏱ Fetching {TIMEFRAME_ENTRY} and {TIMEFRAME_FILTER} for {SYMBOL}: {since_dt} → {until_dt} (UTC)")
h1 = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME_ENTRY, since_ms, until_ms, limit=1500, pause=0.12)
h4 = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME_FILTER, since_ms, until_ms, limit=1500, pause=0.12)
if h1.empty or h4.empty:
    raise ValueError(f"No futures data from exchange for {SYMBOL}. h1 empty: {h1.empty}, h4 empty: {h4.empty}")
print(f"✅ Loaded {len(h1)} (1h) and {len(h4)} (4h) bars")

funding_df = None
if INCLUDE_FUNDING:
    print("⏱ Fetching funding rate history…")
    funding_df = fetch_funding_history(exchange, SYMBOL, since_ms, until_ms)
    if funding_df is None or funding_df.empty:
        print("⚠️ No funding history returned; funding will be 0.")
funding_series = align_funding_schedule(h1.index, funding_df)

tiers = fetch_leverage_tiers(exchange, SYMBOL)

# =========================
# INDICATORS
# =========================
h1['Bias'] = 0
h1.loc[h1['Close'] > h1['Close'].shift(1), 'Bias'] = 1
h1.loc[h1['Close'] < h1['Close'].shift(1), 'Bias'] = -1

h4['Trend'] = 0
h4.loc[h4['Close'] > h4['Close'].shift(1), 'Trend'] = 1
h4.loc[h4['Close'] < h4['Close'].shift(1), 'Trend'] = -1

h1['H4_Trend'] = h4['Trend'].reindex(h1.index, method='ffill').fillna(0).astype(int)

if USE_ATR_STOPS:
    h1['ATR'] = calculate_atr(h1, ATR_PERIOD)

if USE_VOLUME_FILTER:
    h1['Avg_Volume'] = h1['Volume'].rolling(window=VOL_LOOKBACK).mean()

h1['RSI'] = calculate_rsi(h1['Close'], RSI_PERIOD)

# Drop last open candle - same as deploy
if len(h1) >= 2:
    h1_closed = h1.iloc[:-1].copy()
else:
    h1_closed = h1.copy()
if len(h4) >= 2:
    h4_closed = h4.iloc[:-1].copy()
else:
    h4_closed = h4.copy()

# =========================
# Entry Signals (1 = long, -1 = short)
# =========================
h1_closed['Entry_Signal'] = 0
for i in range(1, len(h1_closed)):
    price = h1_closed['Close'].iat[i]
    open_price = h1_closed['Open'].iat[i]
    prev_close = h1_closed['Close'].iat[i-1]
    h4_trend = h1_closed['H4_Trend'].iat[i]

    bullish_sweep = (price > open_price) and (price > prev_close)
    vol_ok_long = True
    if USE_VOLUME_FILTER and not pd.isna(h1_closed['Avg_Volume'].iat[i]):
        vol_ok_long = h1_closed['Volume'].iat[i] >= VOL_MIN_RATIO * h1_closed['Avg_Volume'].iat[i]
    rsi_ok_long = True if pd.isna(h1_closed['RSI'].iat[i]) else (h1_closed['RSI'].iat[i] > RSI_OVERSOLD)
    h4_ok_long = (not USE_H1_FILTER) or (h4_trend == 1)

    bearish_sweep = (price < open_price) and (price < prev_close)
    vol_ok_short = True
    if USE_VOLUME_FILTER and not pd.isna(h1_closed['Avg_Volume'].iat[i]):
        vol_ok_short = h1_closed['Volume'].iat[i] >= VOL_MIN_RATIO * h1_closed['Avg_Volume'].iat[i]
    rsi_ok_short = True if pd.isna(h1_closed['RSI'].iat[i]) else (h1_closed['RSI'].iat[i] < (100 - RSI_OVERSOLD))
    h4_ok_short = (not USE_H1_FILTER) or (h4_trend == -1)

    if bullish_sweep and vol_ok_long and rsi_ok_long and h4_ok_long:
        h1_closed.iat[i, h1_closed.columns.get_loc('Entry_Signal')] = 1
    elif bearish_sweep and vol_ok_short and rsi_ok_short and h4_ok_short:
        h1_closed.iat[i, h1_closed.columns.get_loc('Entry_Signal')] = -1

# =========================
# Backtest engine (close-based SL/TP + funding + liquidation)
# =========================
capital = INITIAL_CAPITAL
position = 0
entry_price = entry_sl = entry_tp = 0.0
entry_time = None
entry_size = 0.0
bias_flip_count = 0
permanently_stopped = False

liq_lower = None
liq_upper = None
entry_notional = 0.0

trades = []
equity_curve = []
peak_equity = INITIAL_CAPITAL

print("\n" + "="*70)
print("⚙️  FUTURES BACKTEST — Backtest with same rules as deploy")
print("="*70)

# iterate over closed bars
for i in range(len(h1_closed)):
    ts = h1_closed.index[i]
    price = float(h1_closed['Close'].iat[i])
    signal = int(h1_closed['Entry_Signal'].iat[i])
    bias = int(h1_closed['Bias'].iat[i])
    h4_trend = int(h1_closed['H4_Trend'].iat[i])

    # funding application
    if INCLUDE_FUNDING and position != 0:
        rate = float(funding_series.loc[ts]) if ts in funding_series.index and not pd.isna(funding_series.loc[ts]) else 0.0
        if rate != 0.0:
            fee = entry_notional * rate * (1 if position == 1 else -1) * (-1)
            capital += fee

    # drawdown check
    if equity_curve:
        peak_equity = max(peak_equity, capital)
        curr_dd = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0.0
        if curr_dd >= MAX_DRAWDOWN and not permanently_stopped:
            permanently_stopped = True
            print(f"\n🛑 PERMANENT STOP at {ts} | DD: {curr_dd*100:.2f}%")
            if position != 0:
                exit_price = price
                gross_pnl = entry_size * (exit_price - entry_price) * (1 if position == 1 else -1)
                position_value = abs(exit_price * entry_size)
                exit_slippage = position_value * SLIPPAGE_RATE
                exit_fee = position_value * FEE_RATE
                net_pnl = gross_pnl - exit_slippage - exit_fee
                capital += net_pnl
                trades.append({
                    'Trade_ID': len(trades) + 1,
                    'Entry_DateTime': entry_time,
                    'Exit_DateTime': ts,
                    'Position': 'Long' if position==1 else 'Short',
                    'Entry_Price': round(entry_price, 6),
                    'Exit_Price': round(exit_price, 6),
                    'Take_Profit': round(entry_tp, 6),
                    'Stop_Loss': round(entry_sl, 6),
                    'Position_Size_Base': round(entry_size, 8),
                    'PnL_$': round(net_pnl, 2),
                    'Win': 1 if net_pnl > 0 else 0,
                    'Exit_Reason': 'MAX DRAWDOWN',
                    'Capital_After': round(capital, 2)
                })
                position = 0
                liq_lower = liq_upper = None
                entry_notional = 0.0

    # liquidation check (very conservative for 1x)
    if position != 0 and not permanently_stopped:
        if position == 1 and liq_lower is not None and price <= liq_lower:
            position_value = abs(price * entry_size)
            penalty = position_value * LIQUIDATION_PENALTY_RATE
            gross_pnl = entry_size * (price - entry_price)
            net_pnl = gross_pnl - penalty
            capital += net_pnl
            trades.append({
                'Trade_ID': len(trades) + 1,
                'Entry_DateTime': entry_time,
                'Exit_DateTime': ts,
                'Position': 'Long',
                'Entry_Price': round(entry_price, 6),
                'Exit_Price': round(price, 6),
                'Take_Profit': round(entry_tp, 6),
                'Stop_Loss': round(entry_sl, 6),
                'Position_Size_Base': round(entry_size, 8),
                'PnL_$': round(net_pnl, 2),
                'Win': 1 if net_pnl > 0 else 0,
                'Exit_Reason': 'LIQUIDATION',
                'Capital_After': round(capital, 2)
            })
            position = 0
            liq_lower = liq_upper = None
            entry_notional = 0.0

        elif position == -1 and liq_upper is not None and price >= liq_upper:
            position_value = abs(price * entry_size)
            penalty = position_value * LIQUIDATION_PENALTY_RATE
            gross_pnl = entry_size * (entry_price - price)
            net_pnl = gross_pnl - penalty
            capital += net_pnl
            trades.append({
                'Trade_ID': len(trades) + 1,
                'Entry_DateTime': entry_time,
                'Exit_DateTime': ts,
                'Position': 'Short',
                'Entry_Price': round(entry_price, 6),
                'Exit_Price': round(price, 6),
                'Take_Profit': round(entry_tp, 6),
                'Stop_Loss': round(entry_sl, 6),
                'Position_Size_Base': round(entry_size, 8),
                'PnL_$': round(net_pnl, 2),
                'Win': 1 if net_pnl > 0 else 0,
                'Exit_Reason': 'LIQUIDATION',
                'Capital_After': round(capital, 2)
            })
            position = 0
            liq_lower = liq_upper = None
            entry_notional = 0.0

    # normal exit logic
    if position != 0 and not permanently_stopped:
        exit_flag = False
        exit_price = price
        exit_reason = ""

        if position == 1:
            if price >= entry_tp:
                exit_flag, exit_price, exit_reason = True, entry_tp, "Take Profit"
                bias_flip_count = 0
            elif price <= entry_sl:
                exit_flag, exit_price, exit_reason = True, entry_sl, "Stop Loss"
                bias_flip_count = 0
            elif USE_H1_FILTER and (h4_trend < 0 and bias < 0):
                exit_flag, exit_price, exit_reason = True, price, "4H Trend Reversal"
                bias_flip_count = 0
            elif bias < 0:
                bias_flip_count += 1
                if bias_flip_count >= BIAS_CONFIRM_BEAR:
                    exit_flag, exit_price, exit_reason = True, price, "Bias Reversal"
                    bias_flip_count = 0
            else:
                bias_flip_count = 0
        else:
            if price <= entry_tp:
                exit_flag, exit_price, exit_reason = True, entry_tp, "Take Profit"
                bias_flip_count = 0
            elif price >= entry_sl:
                exit_flag, exit_price, exit_reason = True, entry_sl, "Stop Loss"
                bias_flip_count = 0
            elif USE_H1_FILTER and (h4_trend > 0 and bias > 0):
                exit_flag, exit_price, exit_reason = True, price, "4H Trend Reversal"
                bias_flip_count = 0
            elif bias > 0:
                bias_flip_count += 1
                if bias_flip_count >= BIAS_CONFIRM_BEAR:
                    exit_flag, exit_price, exit_reason = True, price, "Bias Reversal"
                    bias_flip_count = 0
            else:
                bias_flip_count = 0

        if exit_flag:
            gross_pnl = entry_size * (exit_price - entry_price) * (1 if position == 1 else -1)
            position_value = abs(exit_price * entry_size)
            exit_slippage = position_value * SLIPPAGE_RATE
            exit_fee = position_value * FEE_RATE
            net_pnl = gross_pnl - exit_slippage - exit_fee
            capital += net_pnl

            trades.append({
                'Trade_ID': len(trades) + 1,
                'Entry_DateTime': entry_time,
                'Exit_DateTime': ts,
                'Position': 'Long' if position==1 else 'Short',
                'Entry_Price': round(entry_price, 6),
                'Exit_Price': round(exit_price, 6),
                'Take_Profit': round(entry_tp, 6),
                'Stop_Loss': round(entry_sl, 6),
                'Position_Size_Base': round(entry_size, 8),
                'PnL_$': round(net_pnl, 2),
                'Win': 1 if net_pnl > 0 else 0,
                'Exit_Reason': exit_reason,
                'Capital_After': round(capital, 2)
            })

            position = 0
            entry_price = entry_sl = entry_tp = 0.0
            entry_time = None
            entry_size = 0.0
            liq_lower = liq_upper = None
            entry_notional = 0.0

    # entry logic (on closed bars)
    if position == 0 and not permanently_stopped and signal != 0:
        if USE_ATR_STOPS:
            atr_val = h1_closed['ATR'].iat[i]
            if pd.isna(atr_val) or atr_val <= 0:
                equity_curve.append(capital)
                continue

        if signal == 1:
            sl = price - (ATR_MULT_SL * h1_closed['ATR'].iat[i]) if USE_ATR_STOPS else price * (1 - min(max(price*0.0005,0.0005),0.0015))
            risk = abs(price - sl)
            if risk <= 0:
                equity_curve.append(capital); continue

            rr_ratio = RR_FIXED
            if DYNAMIC_RR and USE_ATR_STOPS and not pd.isna(h1_closed['ATR'].iat[i]) and i >= 6:
                recent_atr = float(h1_closed['ATR'].iloc[i-5:i].mean())
                current_atr = float(h1_closed['ATR'].iat[i])
                if recent_atr > 0:
                    if current_atr > recent_atr * 1.2: rr_ratio = MIN_RR
                    elif current_atr < recent_atr * 0.8: rr_ratio = MAX_RR
            tp = price + rr_ratio * risk

            size_base = calculate_futures_position_size(price, sl, capital, RISK_PERCENT, MAX_TRADE_SIZE)
            # quantize to exchange precision (match deploy)
            size_q = amount_to_precision(exchange, SYMBOL, size_base)
            if size_q <= 0:
                equity_curve.append(capital)
                continue

            position = 1
            entry_price, entry_sl, entry_tp, entry_time, entry_size = price, sl, tp, ts, size_q
            position_value = abs(entry_price * entry_size)
            entry_slippage = position_value * SLIPPAGE_RATE
            entry_fee = position_value * FEE_RATE
            capital -= (entry_slippage + entry_fee)

            entry_notional = position_value
            mmr = get_mmr_for_notional(tiers, entry_notional)
            liq_lower, liq_upper = estimate_liquidation_bounds(entry_price, entry_notional, 1, mmr, leverage=LEVERAGE)

            # debug compare
            if DEBUG_COMPARE:
                line = f"{SYMBOL},{ts},{signal},{price},{sl},{tp},{size_base},{size_q},{rr_ratio},{h1_closed['ATR'].iat[i] if 'ATR' in h1_closed.columns else ''},{h1_closed['RSI'].iat[i]}\n"
                with open(DEBUG_CSV, "a") as f:
                    f.write(line)

        elif signal == -1:
            sl = price + (ATR_MULT_SL * h1_closed['ATR'].iat[i]) if USE_ATR_STOPS else price * (1 + min(max(price*0.0005,0.0005),0.0015))
            risk = abs(sl - price)
            if risk <= 0:
                equity_curve.append(capital); continue

            rr_ratio = RR_FIXED
            if DYNAMIC_RR and USE_ATR_STOPS and not pd.isna(h1_closed['ATR'].iat[i]) and i >= 6:
                recent_atr = float(h1_closed['ATR'].iloc[i-5:i].mean())
                current_atr = float(h1_closed['ATR'].iat[i])
                if recent_atr > 0:
                    if current_atr > recent_atr * 1.2: rr_ratio = MIN_RR
                    elif current_atr < recent_atr * 0.8: rr_ratio = MAX_RR
            tp = price - rr_ratio * risk

            size_base = calculate_futures_position_size(price, sl, capital, RISK_PERCENT, MAX_TRADE_SIZE)
            size_q = amount_to_precision(exchange, SYMBOL, size_base)
            if size_q <= 0:
                equity_curve.append(capital)
                continue

            position = -1
            entry_price, entry_sl, entry_tp, entry_time, entry_size = price, sl, tp, ts, size_q
            position_value = abs(entry_price * entry_size)
            entry_slippage = position_value * SLIPPAGE_RATE
            entry_fee = position_value * FEE_RATE
            capital -= (entry_slippage + entry_fee)

            entry_notional = position_value
            mmr = get_mmr_for_notional(tiers, entry_notional)
            liq_lower, liq_upper = estimate_liquidation_bounds(entry_price, entry_notional, -1, mmr, leverage=LEVERAGE)

            if DEBUG_COMPARE:
                line = f"{SYMBOL},{ts},{signal},{price},{sl},{tp},{size_base},{size_q},{rr_ratio},{h1_closed['ATR'].iat[i] if 'ATR' in h1_closed.columns else ''},{h1_closed['RSI'].iat[i]}\n"
                with open(DEBUG_CSV, "a") as f:
                    f.write(line)

    equity_curve.append(capital)

# =========================
# RESULTS
# =========================
trades_df = pd.DataFrame(trades)

equity_curve_np = np.array(equity_curve)
if len(equity_curve_np):
    peak_equity_curve = np.maximum.accumulate(equity_curve_np)
    drawdown_array = (peak_equity_curve - equity_curve_np) / np.where(peak_equity_curve==0,1,peak_equity_curve) * 100
    max_drawdown = float(np.max(drawdown_array))
else:
    drawdown_array = np.array([0.0])
    max_drawdown = 0.0

if not trades_df.empty:
    wins = int(trades_df['Win'].sum())
    losses = int(len(trades_df) - wins)
    win_rate = (wins / len(trades_df) * 100.0)
    total_pnl = float(trades_df['PnL_$'].sum())
    final_capital = capital

    exit_counts = trades_df['Exit_Reason'].value_counts()

    print("\n" + "="*70)
    print("✅ FUTURES BACKTEST RESULTS")
    print("="*70)
    print(f"Symbol:              {SYMBOL}")
    print(f"Period:              {h1_closed.index[0]} to {h1_closed.index[-1]}")
    print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:       ${final_capital:,.2f}")
    print(f"Total PnL:           ${total_pnl:,.2f}")
    print(f"Return:              {((final_capital/INITIAL_CAPITAL)-1)*100:.2f}%")
    print(f"Total Trades:        {len(trades_df)}")
    print(f"Wins:                {wins} ({win_rate:.1f}%)")
    print(f"Losses:              {losses}")
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    print(f"Permanently Stopped: {'YES 🛑' if permanently_stopped else 'NO ✅'}")

    print("\n📊 EXIT BREAKDOWN:")
    for reason, count in exit_counts.items():
        pct = (count / len(trades_df)) * 100
        print(f"  {reason:20s}: {count:3d} trades ({pct:5.1f}%)")

    trades_df.to_csv(TRADE_CSV_FILENAME, index=False)
    print(f"\n💾 Trades saved to: {TRADE_CSV_FILENAME}")

else:
    print("\n❌ No trades executed.")
    final_capital = INITIAL_CAPITAL

# =========================
# PLOTS
# =========================
plt.figure(figsize=(15,10))

plt.subplot(2,1,1)
plt.plot(h1.index, h1['Close'], label='Price', linewidth=1)

if not trades_df.empty:
    long_entries = trades_df[trades_df['Position'] == 'Long']
    short_entries = trades_df[trades_df['Position'] == 'Short']
    if not long_entries.empty:
        plt.scatter(long_entries['Entry_DateTime'], long_entries['Entry_Price'], marker='^', label='Long Entry', s=80, zorder=5)
    if not short_entries.empty:
        plt.scatter(short_entries['Entry_DateTime'], short_entries['Entry_Price'], marker='v', label='Short Entry', s=80, zorder=5)

    for exit_type, color, marker in [
        ('Take Profit', 'blue', 'o'),
        ('Stop Loss', 'red', 'x'),
        ('4H Trend Reversal', 'orange', 's'),
        ('Bias Reversal', 'purple', 'd'),
        ('LIQUIDATION', 'black', 'X'),
        ('MAX DRAWDOWN', 'black', 'X')
    ]:
        exits = trades_df[trades_df['Exit_Reason'] == exit_type]
        if not exits.empty:
            plt.scatter(exits['Exit_DateTime'], exits['Exit_Price'], marker=marker, label=f'{exit_type}', s=70, zorder=5)

plt.title(f"{SYMBOL} Perps — Futures Backtest (Long+Short, 1x)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.subplot(2,1,2)
equity_dates = h1_closed.index[:len(equity_curve)]
plt.plot(equity_dates, equity_curve, label='Equity', linewidth=2)
plt.axhline(y=INITIAL_CAPITAL, linestyle='--', label='Initial Capital')
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Capital ($)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ FUTURES BACKTEST COMPLETE.")
