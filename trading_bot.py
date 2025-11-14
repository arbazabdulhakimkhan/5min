# bot.py
import os, time, json, traceback, threading
from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import numpy as np
import requests

# =========================
# CONFIG (env-driven)
# =========================
MODE = os.getenv("MODE", "paper").lower()
EXCHANGE_ID = "kucoinfutures"
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "ARB/USDT:USDT,LINK/USDT:USDT,SOL/USDT:USDT,ETH/USDT:USDT,BTC/USDT:USDT").split(",") if s.strip()]

ENTRY_TF = os.getenv("ENTRY_TF", "5m")
HTF = os.getenv("HTF", "15m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "90"))

TOTAL_PORTFOLIO_CAPITAL = float(os.getenv("TOTAL_PORTFOLIO_CAPITAL", "10000"))
PER_COIN_ALLOCATION = float(os.getenv("PER_COIN_ALLOCATION", "0.20"))
PER_COIN_CAP_USD = TOTAL_PORTFOLIO_CAPITAL * PER_COIN_ALLOCATION

# strategy
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.02"))
RR_FIXED = float(os.getenv("RR_FIXED", "5.0"))
DYNAMIC_RR = os.getenv("DYNAMIC_RR", "true").lower() == "true"
MIN_RR = float(os.getenv("MIN_RR", "4.0"))
MAX_RR = float(os.getenv("MAX_RR", "6.0"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.5"))
USE_ATR_STOPS = os.getenv("USE_ATR_STOPS", "true").lower() == "true"
USE_HTF_GATE = os.getenv("USE_HTF_GATE", "true").lower() == "true"

# filters - UPDATED FOR LESS FALSE SIGNALS
USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "true").lower() == "true"
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MIN_RATIO = float(os.getenv("VOL_MIN_RATIO", "1.2"))  # UPDATED: 1.0 -> 1.2
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_THRESHOLD_LONG = float(os.getenv("RSI_THRESHOLD_LONG", "45"))  # UPDATED: 50 -> 45
RSI_THRESHOLD_SHORT = float(os.getenv("RSI_THRESHOLD_SHORT", "55"))  # UPDATED: 50 -> 55
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "0.0"))

# FVG & PA params - UPDATED FOR LESS FALSE SIGNALS
RETEST_BUFFER_PCT = float(os.getenv("RETEST_BUFFER_PCT", "0.0015"))  # UPDATED: 0.003 -> 0.0015
WICK_BODY_RATIO = float(os.getenv("WICK_BODY_RATIO", "1.5"))  # UPDATED: 0.6 -> 1.5
MIN_FVG_GAP_PCT = float(os.getenv("MIN_FVG_GAP_PCT", "0.002"))  # NEW: minimum 0.2% gap
FVG_MAX_AGE_BARS = int(os.getenv("FVG_MAX_AGE_BARS", "20"))  # NEW: FVG expiry
MIN_CONFLUENCE = int(os.getenv("MIN_CONFLUENCE", "4"))  # NEW: require 4/5 factors
SWING_STRENGTH = int(os.getenv("SWING_STRENGTH", "2"))  # NEW: swing detection bars

# risk/fees
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.20"))
MAX_TRADE_SIZE = float(os.getenv("MAX_TRADE_SIZE", "100000"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.0006"))
INCLUDE_FUNDING = os.getenv("INCLUDE_FUNDING", "true").lower() == "true"

# telegram
TELEGRAM_TOKEN_FUT = os.getenv("TELEGRAM_TOKEN_FUT", "")
TELEGRAM_CHAT_ID_FUT = os.getenv("TELEGRAM_CHAT_ID_FUT", "")

# kucoin API (live only)
API_KEY = os.getenv("KUCOIN_API_KEY", "")
API_SECRET = os.getenv("KUCOIN_SECRET", "")
API_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "")

# scheduler
SEND_DAILY_SUMMARY = os.getenv("SEND_DAILY_SUMMARY", "true").lower() == "true"
SUMMARY_HOUR_IST = int(os.getenv("SUMMARY_HOUR", "20"))

SLEEP_CAP = int(os.getenv("SLEEP_CAP", "60"))
LOG_PREFIX = "[FUT-BOT]"

if MODE == "live":
    if not API_KEY or not API_SECRET or not API_PASSPHRASE:
        raise ValueError("Live mode requires KUCOIN_API_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE")

# =========================
# TELEGRAM helpers
# =========================
def send_telegram_fut(msg: str):
    if not TELEGRAM_TOKEN_FUT or not TELEGRAM_CHAT_ID_FUT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN_FUT}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID_FUT, "text": msg},
            timeout=10
        )
    except Exception:
        pass

# =========================
# EXCHANGE & DATA
# =========================
def get_exchange():
    if MODE == "live":
        return ccxt.kucoinfutures({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
    else:
        return ccxt.kucoinfutures({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

def timeframe_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    units = {"m": 60000, "h": 3600000, "d": 86400000}
    n = int(''.join([c for c in tf if c.isdigit()]))
    u = ''.join([c for c in tf if c.isalpha()])
    return n * units[u]

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1500, pause=0.12):
    tf_ms = timeframe_to_ms(timeframe)
    out, cursor, last = [], since_ms, None
    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except ccxt.RateLimitExceeded:
            time.sleep(1); continue
        except Exception as e:
            print("fetch_ohlcv error:", e); break
        if not batch: break
        out.extend(batch)
        newest = batch[-1][0]
        cursor = (newest + tf_ms) if (last is None or newest > last) else cursor + tf_ms
        last = newest
        if newest >= until_ms - tf_ms: break
        time.sleep(pause)
    if not out:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    dedup = {r[0]: r for r in out}
    rows = [dedup[k] for k in sorted(dedup.keys()) if since_ms <= k <= until_ms]
    df = pd.DataFrame(rows, columns=["timestamp","Open","High","Low","Close","Volume"])
    # Use UTC-aware conversion then drop tz (naive) to match your app's conventions
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()[["Open","High","Low","Close","Volume"]]

# funding
def fetch_funding_history(exchange, symbol, since_ms, until_ms):
    rates, cursor = [], since_ms
    while cursor < until_ms:
        try:
            page = exchange.fetchFundingRateHistory(symbol, since=cursor, limit=1000)
        except Exception as e:
            print("Funding history error:", e); break
        if not page: break
        rates += page
        newest = page[-1].get('timestamp') or page[-1].get('ts')
        if newest is None or newest <= cursor: break
        cursor = newest + 1
        time.sleep(0.1)
    if not rates:
        return pd.DataFrame(columns=["rate"])
    df = pd.DataFrame([{"ts": r.get('timestamp') or r.get('ts'), "rate": float(r.get('fundingRate', r.get('info',{}).get('fundingRate', 0.0)))} for r in rates if (r.get('timestamp') or r.get('ts')) is not None])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.drop(columns=["ts"]).set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="last")]

def align_funding_to_index(idx, funding_df):
    s = pd.Series(0.0, index=idx)
    if funding_df is None or funding_df.empty: return s
    for ts, row in funding_df.iterrows():
        j = s.index.searchsorted(ts)
        if j < len(s): s.iloc[j] = row["rate"]
    return s

# =========================
# INDICATORS & HELPERS
# =========================
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def position_size_futures(price, sl, capital, risk_percent, max_trade_size):
    risk_per_trade = capital * risk_percent
    rpc = abs(price - sl)
    max_by_risk = (risk_per_trade / rpc) if rpc > 0 else 0
    max_by_capital = capital / price
    return max(min(max_by_risk, max_by_capital, max_trade_size / price), 0)

# ---------- UPDATED SMC/FVG/PA helpers ----------
def get_htf_structure_level(htf_df, lookback=30, swing_strength=SWING_STRENGTH):
    if len(htf_df) < swing_strength * 2 + 1:
        return {'trend':0, 'last_sh':None, 'last_sl':None}
    
    swing_highs = []
    swing_lows = []
    
    for i in range(swing_strength, len(htf_df) - swing_strength):
        is_swing_high = all(
            htf_df['High'].iloc[i] > htf_df['High'].iloc[i-j] and
            htf_df['High'].iloc[i] > htf_df['High'].iloc[i+j]
            for j in range(1, swing_strength + 1)
        )
        if is_swing_high:
            swing_highs.append((htf_df.index[i], float(htf_df['High'].iloc[i])))
        
        is_swing_low = all(
            htf_df['Low'].iloc[i] < htf_df['Low'].iloc[i-j] and
            htf_df['Low'].iloc[i] < htf_df['Low'].iloc[i+j]
            for j in range(1, swing_strength + 1)
        )
        if is_swing_low:
            swing_lows.append((htf_df.index[i], float(htf_df['Low'].iloc[i])))
    
    last_sh = swing_highs[-1][1] if swing_highs else None
    last_sl = swing_lows[-1][1] if swing_lows else None
    
    trend = 0
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1] > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1] < swing_lows[-2][1]
        if hh and hl:
            trend = 1
        elif lh and ll:
            trend = -1
    
    return {'trend': int(trend), 'last_sh': last_sh, 'last_sl': last_sl}

def find_fvgs(htf_df, min_gap_pct=MIN_FVG_GAP_PCT):
    out = []
    if len(htf_df) < 3:
        return out
    for i in range(1, len(htf_df)):
        prev = htf_df.iloc[i-1]
        curr = htf_df.iloc[i]
        if float(curr['Low']) > float(prev['High']):
            gap_size = (curr['Low'] - prev['High']) / prev['High']
            if gap_size >= min_gap_pct:
                out.append({'type':'bull','low':float(prev['High']),'high':float(curr['Low']),'idx':htf_df.index[i]})
        if float(curr['High']) < float(prev['Low']):
            gap_size = (prev['Low'] - curr['High']) / curr['High']
            if gap_size >= min_gap_pct:
                out.append({'type':'bear','low':float(curr['High']),'high':float(prev['Low']),'idx':htf_df.index[i]})
    return out

def filter_recent_fvgs(fvgs, current_ts, htf_timeframe, max_age_bars=FVG_MAX_AGE_BARS):
    if not fvgs:
        return []

    tf_minutes = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
    }

    tf_key = htf_timeframe.lower()
    minutes_per_bar = tf_minutes.get(tf_key, 15)
    max_age_seconds = max_age_bars * minutes_per_bar * 60

    recent = []
    for z in fvgs:
        age_seconds = (current_ts - z["idx"]).total_seconds()
        if age_seconds <= max_age_seconds:
            recent.append(z)

    return recent

def price_in_zone(price, zone_low, zone_high, buffer_pct=RETEST_BUFFER_PCT):
    if zone_low is None or zone_high is None:
        return False
    tol = max(zone_high * buffer_pct, 0.0005)
    return (zone_low - tol) <= price <= (zone_high + tol)

def is_bullish_engulfing(prev, curr):
    return (prev['Close'] < prev['Open']) and (curr['Close'] > curr['Open']) and (curr['Close'] > prev['Open']) and (curr['Open'] < prev['Close'])

def is_bearish_engulfing(prev, curr):
    return (prev['Close'] > prev['Open']) and (curr['Close'] < curr['Open']) and (curr['Open'] > prev['Close']) and (curr['Close'] < prev['Open'])

def has_rejection_wick(candle, direction='bull', wick_body_ratio=WICK_BODY_RATIO):
    body = abs(candle['Close'] - candle['Open'])
    if body == 0:
        return False
    if direction == 'bull':
        lower_wick = (candle['Open'] - candle['Low']) if candle['Open'] >= candle['Close'] else (candle['Close'] - candle['Low'])
        return (lower_wick / body) >= wick_body_ratio
    else:
        upper_wick = (candle['High'] - candle['Open']) if candle['Open'] <= candle['Close'] else (candle['High'] - candle['Close'])
        return (upper_wick / body) >= wick_body_ratio

# =========================
# STATE & FILES
# =========================
def state_files_for_symbol(symbol: str):
    tag = "fut_" + symbol.replace("/", "_").replace(":", "_")
    return f"state_{tag}.json", f"{tag}_trades.csv"

def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            s = json.load(f)
        for k in ["entry_time", "last_processed_ts", "last_exit_time"]:
            if s.get(k):
                # parse into tz-naive UTC (consistent)
                s[k] = pd.to_datetime(s[k], utc=True).tz_convert(None)
        return s
    return {
        "capital": PER_COIN_CAP_USD,
        "position": 0,
        "entry_price": 0.0,
        "entry_sl": 0.0,
        "entry_tp": 0.0,
        "entry_time": None,
        "entry_size": 0.0,
        "peak_equity": PER_COIN_CAP_USD,
        "last_processed_ts": None,
        "last_exit_time": None
    }

def save_state(state_file, state):
    s = dict(state)
    for k in ["entry_time", "last_processed_ts", "last_exit_time"]:
        if s.get(k) is not None:
            # store as ISO (naive iso produced by pandas-to-datetime then removed tz)
            s[k] = pd.to_datetime(s[k]).tz_localize(None).isoformat()
    with open(state_file, "w") as f:
        json.dump(s, f, indent=2)

def append_trade(csv_file, row):
    write_header = not os.path.exists(csv_file)
    pd.DataFrame([row]).to_csv(csv_file, mode="a", header=write_header, index=False)

# =========================
# LIVE ORDER HELPERS
# =========================
def place_market(exchange, symbol, side, amount, reduce_only=False):
    params = {"reduceOnly": True} if reduce_only else {}
    return exchange.create_order(symbol, type="market", side=side, amount=amount, params=params)

def avg_fill_price(order):
    p = order.get("average") or order.get("price")
    if p: return float(p)
    if "trades" in order and order["trades"]:
        notional = 0.0; qty = 0.0
        for t in order["trades"]:
            pr = float(t["price"]); am = float(t["amount"])
            notional += pr*am; qty += am
        if qty > 0: return notional / qty
    return None

def amount_to_precision(exchange, symbol, amt):
    return float(exchange.amount_to_precision(symbol, amt))

# =========================
# CORE PER-BAR PROCESSOR
# =========================
def process_bar(symbol, etf_df, htf_df, state, exchange=None, funding_series=None):
    etf = etf_df.copy(); htf = htf_df.copy()

    if USE_ATR_STOPS:
        etf['ATR'] = calculate_atr(etf, ATR_PERIOD)
    if USE_VOLUME_FILTER:
        etf['Avg_Volume'] = etf['Volume'].rolling(VOL_LOOKBACK).mean()
    etf['RSI'] = calculate_rsi(etf['Close'], RSI_PERIOD)

    i = len(etf) - 1
    if i < 1:
        return state, None

    curr = etf.iloc[i]
    prev = etf.iloc[i-1]
    price = float(curr['Close'])
    ts = etf.index[i]  # tz-naive (we ensured earlier)

    # funding impact when holding
    if INCLUDE_FUNDING and state["position"] != 0 and funding_series is not None:
        rate = float(funding_series.iloc[i]) if i < len(funding_series) else 0.0
        if rate != 0.0 and state.get("entry_price") and state.get("entry_size"):
            notional = abs(state["entry_price"] * state["entry_size"])
            fee = notional * rate * (1 if state["position"] == 1 else -1) * (-1)
            state["capital"] += fee

    # update peak equity
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    if state["peak_equity"] > 0:
        dd = (state["peak_equity"] - state["capital"]) / state["peak_equity"]
    else:
        dd = 0.0

    # forced exit on global drawdown
    if dd >= MAX_DRAWDOWN and state["position"] != 0:
        side = "sell" if state["position"] == 1 else "buy"
        exit_price = price
        if MODE == "live":
            try:
                order = place_market(exchange, symbol, side, amount_to_precision(exchange, symbol, state["entry_size"]), reduce_only=True)
                exit_price = float(avg_fill_price(order) or price)
            except Exception as e:
                send_telegram_fut(f"❌ {symbol} forced-exit failed: {e}")
        gross = state["entry_size"] * (exit_price - state["entry_price"]) * (1 if state["position"]==1 else -1)
        pos_val = abs(exit_price * state["entry_size"])
        pnl = gross - pos_val*SLIPPAGE_RATE - pos_val*FEE_RATE
        state["capital"] += pnl
        row = {
            "Symbol": symbol, "Entry_DateTime": state["entry_time"],
            "Exit_DateTime": ts, "Position": "Long" if state["position"]==1 else "Short",
            "Entry_Price": round(state["entry_price"],6), "Exit_Price": round(exit_price,6),
            "Take_Profit": round(state["entry_tp"],6), "Stop_Loss": round(state["entry_sl"],6),
            "Position_Size_Base": round(state["entry_size"],8),
            "PnL_$": round(pnl,2), "Win": 1 if pnl>0 else 0,
            "Exit_Reason": "MAX_DRAWDOWN", "Capital_After": round(state["capital"],2), "Mode": MODE
        }
        state.update({"position":0,"entry_price":0.0,"entry_sl":0.0,"entry_tp":0.0,"entry_time":None,"entry_size":0.0})
        return state, row

    trade_row = None

    # ===== EXIT logic: only SL or TP =====
    if state["position"] != 0:
        exit_flag = False; exit_reason = ""; exit_price = price
        if state["position"] == 1:
            if price >= state["entry_tp"]:
                exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
            elif price <= state["entry_sl"]:
                exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"
        else:
            if price <= state["entry_tp"]:
                exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
            elif price >= state["entry_sl"]:
                exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"

        if exit_flag:
            side = "sell" if state["position"]==1 else "buy"
            if MODE == "live":
                try:
                    order = place_market(exchange, symbol, side, amount_to_precision(exchange, symbol, state["entry_size"]), reduce_only=True)
                    exit_price = float(avg_fill_price(order) or price)
                except Exception as e:
                    send_telegram_fut(f"❌ {symbol} exit failed: {e}")
            gross = state["entry_size"] * (exit_price - state["entry_price"]) * (1 if state["position"]==1 else -1)
            pos_val = abs(exit_price * state["entry_size"])
            pnl = gross - pos_val*SLIPPAGE_RATE - pos_val*FEE_RATE
            state["capital"] += pnl

            trade_row = {
                "Symbol": symbol, "Entry_DateTime": state["entry_time"],
                "Exit_DateTime": ts, "Position": "Long" if state["position"]==1 else "Short",
                "Entry_Price": round(state["entry_price"],6), "Exit_Price": round(exit_price,6),
                "Take_Profit": round(state["entry_tp"],6), "Stop_Loss": round(state["entry_sl"],6),
                "Position_Size_Base": round(state["entry_size"],8),
                "PnL_$": round(pnl,2), "Win": 1 if pnl>0 else 0,
                "Exit_Reason": exit_reason, "Capital_After": round(state["capital"],2), "Mode": MODE
            }
            state.update({"position":0,"entry_price":0.0,"entry_sl":0.0,"entry_tp":0.0,"entry_time":None,"entry_size":0.0})
            send_telegram_fut(f"{'💚' if pnl>0 else '🔴'} EXIT {symbol} {exit_reason} @ {exit_price:.4f} | PnL ${pnl:.2f}")

    # ===== UPDATED ENTRY logic: HTF structure/FVG + confluence scoring =====
    if state["position"] == 0:
        # cooldown - ensure both sides naive
        if COOLDOWN_HOURS > 0 and state.get("last_exit_time") is not None:
            try:
                last_exit_ts = pd.to_datetime(state["last_exit_time"], utc=True).tz_convert(None) if state.get("last_exit_time") is not None else None
                if last_exit_ts is not None and (ts - last_exit_ts).total_seconds() / 3600 < COOLDOWN_HOURS:
                    state["last_processed_ts"] = ts
                    return state, trade_row
            except Exception:
                # if anything weird, skip cooldown
                pass

        # need minimum bars
        if len(etf) < 3 or len(htf) < 3:
            state["last_processed_ts"] = ts
            return state, trade_row

        struct = get_htf_structure_level(htf, lookback=30, swing_strength=SWING_STRENGTH)
        fvgs = find_fvgs(htf, min_gap_pct=MIN_FVG_GAP_PCT)
        fvgs = filter_recent_fvgs(fvgs, ts, HTF, max_age_bars=FVG_MAX_AGE_BARS)

        vol_ok_long = vol_ok_short = True
        if USE_VOLUME_FILTER and not pd.isna(etf['Volume'].iloc[i]):
            avgv = etf['Volume'].rolling(VOL_LOOKBACK).mean().iloc[i]
            if not pd.isna(avgv):
                vol_ok_long = etf['Volume'].iloc[i] >= VOL_MIN_RATIO * avgv
                vol_ok_short = vol_ok_long

        rsi = float(etf['RSI'].iloc[i]) if not pd.isna(etf['RSI'].iloc[i]) else None
        rsi_ok_long = True if rsi is None else rsi > RSI_THRESHOLD_LONG
        rsi_ok_short = True if rsi is None else rsi < RSI_THRESHOLD_SHORT

        prevc = etf.iloc[i-1]
        currc = etf.iloc[i]

        pa_confirm_long = (is_bullish_engulfing(prevc, currc) or has_rejection_wick(currc, direction='bull', wick_body_ratio=WICK_BODY_RATIO))
        pa_confirm_short = (is_bearish_engulfing(prevc, currc) or has_rejection_wick(currc, direction='bear', wick_body_ratio=WICK_BODY_RATIO))

        retest_long = False
        retest_short = False

        for z in reversed(fvgs):
            if z['type'] == 'bull' and price_in_zone(price, z['low'], z['high']):
                retest_long = True
                break
            if z['type'] == 'bear' and price_in_zone(price, z['low'], z['high']):
                retest_short = True
                break

        if not retest_long and struct.get('last_sl') is not None:
            retest_long = price_in_zone(price, struct['last_sl'], struct['last_sl'])
        if not retest_short and struct.get('last_sh') is not None:
            retest_short = price_in_zone(price, struct['last_sh'], struct['last_sh'])

        htf_trend = struct.get('trend', 0)
        htf_gate_long = (not USE_HTF_GATE) or (htf_trend == 1)
        htf_gate_short = (not USE_HTF_GATE) or (htf_trend == -1)

        confluence_long = confluence_short = 0
        if retest_long: confluence_long += 1
        if pa_confirm_long: confluence_long += 1
        if vol_ok_long: confluence_long += 1
        if rsi_ok_long: confluence_long += 1
        if htf_gate_long: confluence_long += 1

        if retest_short: confluence_short += 1
        if pa_confirm_short: confluence_short += 1
        if vol_ok_short: confluence_short += 1
        if rsi_ok_short: confluence_short += 1
        if htf_gate_short: confluence_short += 1

        long_ok = confluence_long >= MIN_CONFLUENCE
        short_ok = confluence_short >= MIN_CONFLUENCE

        signal = 1 if long_ok else (-1 if short_ok else 0)

        if signal != 0 and (not USE_ATR_STOPS or (USE_ATR_STOPS and not pd.isna(etf['ATR'].iloc[i]) and etf['ATR'].iloc[i] > 0)):
            if signal == 1:
                sl = price - (ATR_MULT_SL * etf['ATR'].iloc[i]) if USE_ATR_STOPS else price * (1 - min(max(price*0.0005,0.0005),0.0015))
                risk = abs(price - sl)
                rr = RR_FIXED
                if DYNAMIC_RR and USE_ATR_STOPS and i >= 6:
                    recent = float(etf['ATR'].iloc[i-5:i].mean())
                    curr_atr = float(etf['ATR'].iloc[i])
                    if recent > 0:
                        if curr_atr > recent*1.2: rr = MIN_RR
                        elif curr_atr < recent*0.8: rr = MAX_RR
                tp = price + rr * risk
            else:
                sl = price + (ATR_MULT_SL * etf['ATR'].iloc[i]) if USE_ATR_STOPS else price * (1 + min(max(price*0.0005,0.0005),0.0015))
                risk = abs(sl - price)
                rr = RR_FIXED
                if DYNAMIC_RR and USE_ATR_STOPS and i >= 6:
                    recent = float(etf['ATR'].iloc[i-5:i].mean())
                    curr_atr = float(etf['ATR'].iloc[i])
                    if recent > 0:
                        if curr_atr > recent*1.2: rr = MIN_RR
                        elif curr_atr < recent*0.8: rr = MAX_RR
                tp = price - rr * risk

            if risk > 0:
                size = position_size_futures(price, sl, state["capital"], RISK_PERCENT, MAX_TRADE_SIZE)
                if size > 0:
                    entry_price_used = price
                    side = "buy" if signal==1 else "sell"
                    if MODE == "live":
                        try:
                            size = amount_to_precision(exchange, symbol, size)
                            order = place_market(exchange, symbol, side, size, reduce_only=False)
                            ep = avg_fill_price(order)
                            if ep is not None: entry_price_used = float(ep)
                        except Exception as e:
                            send_telegram_fut(f"❌ {symbol} entry failed: {e}")
                            state["last_processed_ts"] = ts
                            return state, trade_row

                    state["position"] = 1 if signal==1 else -1
                    state["entry_price"] = entry_price_used
                    state["entry_sl"] = sl
                    state["entry_tp"] = tp
                    state["entry_time"] = ts
                    state["entry_size"] = size

                    pos_val = abs(entry_price_used * size)
                    state["capital"] -= pos_val*SLIPPAGE_RATE
                    state["capital"] -= pos_val*FEE_RATE

                    tag = "LONG" if signal==1 else "SHORT"
                    conf_score = confluence_long if signal==1 else confluence_short
                    send_telegram_fut(f"🚀 ENTRY {symbol} {tag} @ {entry_price_used:.4f} | SL {sl:.4f} | TP {tp:.4f} | RR {rr:.1f} | Confluence {conf_score}/5")

    # update processed ts
    state["last_processed_ts"] = ts
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    return state, trade_row

# =========================
# WORKER (one per symbol)
# =========================
def worker(symbol):
    state_file, trades_csv = state_files_for_symbol(symbol)
    exchange = get_exchange()
    state = load_state(state_file)

    send_telegram_fut(f"🤖 {symbol} FUTURES bot started | {ENTRY_TF}/{HTF} | cap ${state['capital']:.2f} | Min Confluence {MIN_CONFLUENCE}/5")

    while True:
        try:
            # use timezone-aware now, but keep OHLC indexes tz-naive: convert properly for math
            now = datetime.now(timezone.utc)
            since = now - timedelta(days=LOOKBACK_DAYS)
            since_ms = int(since.timestamp()*1000); until_ms = int(now.timestamp()*1000)

            etf = fetch_ohlcv_range(exchange, symbol, ENTRY_TF, since_ms, until_ms)
            htf = fetch_ohlcv_range(exchange, symbol, HTF, since_ms, until_ms)
            if etf.empty or htf.empty or len(etf) < 3:
                time.sleep(30); continue

            # act on last CLOSED bar (we use -2 index as last closed; we'll pass data trimmed to exclude realtime)
            # etf.index are tz-naive datetimes (we ensured earlier)
            closed_ts = etf.index[-2]

            if state["last_processed_ts"] is not None:
                # parse stored state ts into tz-naive UTC then compare
                last_ts = pd.to_datetime(state["last_processed_ts"], utc=True).tz_convert(None) if state.get("last_processed_ts") is not None else None
                if last_ts is not None and last_ts >= closed_ts:
                    time.sleep(10); continue

            # prepare trimmed dfs so last row is last closed bar
            etf_trim = etf.iloc[:-1]  # exclude current forming bar
            htf_trim = htf.iloc[:-1]

            funding_series = None
            if INCLUDE_FUNDING:
                fdf = fetch_funding_history(exchange, symbol, int(etf_trim.index[0].timestamp()*1000), int(etf_trim.index[-1].timestamp()*1000))
                funding_series = align_funding_to_index(etf_trim.index, fdf) if (fdf is not None and not fdf.empty) else pd.Series(0.0, index=etf_trim.index)

            state, trade = process_bar(symbol, etf_trim, htf_trim, state, exchange=exchange, funding_series=funding_series)
            if trade is not None:
                append_trade(trades_csv, trade)

            save_state(state_file, state)

            # sleep to just after next bar close (ENTRY_TF cadence)
            next_close = etf.index[-1] + (etf.index[-1] - etf.index[-2])  # tz-naive

            # make next_close timezone-aware UTC for safe subtract with now_utc
            if getattr(next_close, "tzinfo", None) is None:
                next_close = next_close.replace(tzinfo=timezone.utc)

            now_utc = datetime.now(timezone.utc)
            sleep_sec = (next_close - now_utc).total_seconds() + 5
            if sleep_sec < 5: sleep_sec = 5
            if sleep_sec > 3600: sleep_sec = SLEEP_CAP
            time.sleep(sleep_sec)

        except ccxt.RateLimitExceeded:
            time.sleep(10)
        except Exception as e:
            msg = f"{LOG_PREFIX} {symbol} ERROR: {e}"
            print(msg)
            traceback.print_exc()
            send_telegram_fut(msg)
            time.sleep(30)

# =========================
# DAILY SUMMARY (IST)
# =========================
def ist_now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))

def generate_daily_summary():
    try:
        now_ist = ist_now()
        start_ist = datetime(now_ist.year, now_ist.month, now_ist.day, 0,0,0, tzinfo=now_ist.tzinfo)
        end_ist = datetime(now_ist.year, now_ist.month, now_ist.day, 23,59,59, tzinfo=now_ist.tzinfo)
        start_utc = start_ist.astimezone(timezone.utc).replace(tzinfo=None)
        end_utc = end_ist.astimezone(timezone.utc).replace(tzinfo=None)

        lines = [f"📊 FUTURES DAILY SUMMARY — {now_ist.strftime('%Y-%m-%d %I:%M %p IST')}", "-"*60]
        total_cap = total_init = pnl_today = n_today = w_today = 0.0

        for sym in SYMBOLS:
            state_file, trades_csv = state_files_for_symbol(sym)
            state = load_state(state_file) if os.path.exists(state_file) else {"capital": PER_COIN_CAP_USD, "position":0}
            cap = float(state.get("capital", PER_COIN_CAP_USD))
            initial = PER_COIN_CAP_USD

            wins = losses = wr = 0.0
            pnl_all = 0.0
            n_trades_today = wins_today = 0
            pnl_today_sym = 0.0

            if os.path.exists(trades_csv):
                df = pd.read_csv(trades_csv)
                if len(df):
                    # robust conversion to naive UTC (handles both tz-aware and naive strings)
                    df['Exit_DateTime'] = pd.to_datetime(df['Exit_DateTime'], utc=True, errors='coerce').dt.tz_convert(None)
                    today = df[(df['Exit_DateTime'] >= start_utc) & (df['Exit_DateTime'] <= end_utc)]
                    n_trades_today = len(today)
                    wins_today = int(today['Win'].sum()) if n_trades_today else 0
                    pnl_today_sym = float(today["PnL_$"].sum()) if n_trades_today else 0.0
                    pnl_all = float(df["PnL_$"].sum()) if "PnL_$" in df.columns else 0.0
                    wins = int(df['Win'].sum()) if 'Win' in df.columns else 0
                    losses = len(df)-wins if len(df) else 0
                    wr = (wins/len(df)*100) if len(df) else 0.0

            total_cap += cap; total_init += initial
            pnl_today += pnl_today_sym; n_today += n_trades_today; w_today += wins_today
            roi = ((cap/initial)-1)*100 if initial>0 else 0.0

            lines.append(f"{sym}: cap ${cap:,.2f} ({roi:+.2f}%) | today {n_trades_today} trades, {wins_today} wins | PnL ${pnl_today_sym:+.2f} | all WR {wr:.1f}%")

        port_roi = ((total_cap/total_init)-1)*100 if total_init>0 else 0.0
        wr_today = (w_today/n_today*100) if n_today>0 else 0.0
        lines += ["-"*60, f"TOTAL: cap ${total_cap:,.2f} ({port_roi:+.2f}%) | today {n_today} trades | WR {wr_today:.1f}% | PnL ${pnl_today:+.2f}"]
        msg = "\n".join(lines)
        print(msg)
        send_telegram_fut(msg)
    except Exception as e:
        send_telegram_fut(f"❌ summary error: {e}")

def summary_scheduler():
    last_sent_date = None
    while True:
        try:
            now = ist_now()
            if now.hour == SUMMARY_HOUR_IST and (last_sent_date is None or last_sent_date != now.date()):
                generate_daily_summary()
                last_sent_date = now.date()
            time.sleep(60)
        except Exception:
            time.sleep(60)

# =========================
# MAIN
# =========================
def main():
    boot = f"""
🚀 Futures Bot Started (UPDATED - Reduced False Signals)
Mode: {MODE.upper()}
Exchange: KuCoin Futures (perps)
Symbols: {", ".join(SYMBOLS)}
TF: {ENTRY_TF}/{HTF}
Cap/coin: ${PER_COIN_CAP_USD:,.2f}
Risk: {RISK_PERCENT*100:.1f}% | Fee: {FEE_RATE*100:.3f}% | Slippage: {SLIPPAGE_RATE*100:.3f}%

UPDATED FILTERS:
✓ Minimum FVG gap: {MIN_FVG_GAP_PCT*100:.2f}%
✓ FVG max age: {FVG_MAX_AGE_BARS} HTF bars
✓ Wick/body ratio: {WICK_BODY_RATIO}x
✓ Volume threshold: {VOL_MIN_RATIO}x average
✓ RSI thresholds: Long>{RSI_THRESHOLD_LONG}, Short<{RSI_THRESHOLD_SHORT}
✓ Retest buffer: {RETEST_BUFFER_PCT*100:.2f}%
✓ Min confluence: {MIN_CONFLUENCE}/5 factors
✓ Swing strength: {SWING_STRENGTH} bars

Strategy: FVG + SMC + PA entry with proper swing detection
Exits: SL/TP only
"""
    print(boot)
    send_telegram_fut(boot)

    threads = []
    for sym in SYMBOLS:
        t = threading.Thread(target=worker, args=(sym,), daemon=True)
        t.start(); threads.append(t)
        time.sleep(1)

    if SEND_DAILY_SUMMARY:
        s = threading.Thread(target=summary_scheduler, daemon=True)
        s.start(); threads.append(s)

    print(f"✅ Running {len(threads)} threads…")
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()

