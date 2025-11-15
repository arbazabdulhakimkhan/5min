# trading_bot.py — final deploy-ready with variable block (paste into repo)
import os
import time
import json
import traceback
import threading
from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import numpy as np
import requests



# =========================
# Parse VAR_BLOCK (simple .env parser)
# Real OS environment variables override these defaults.
# =========================
def parse_env_block(block: str):
    d = {}
    for raw in block.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        # remove surrounding quotes if present
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        d[k] = v
    return d

_defaults = parse_env_block(VAR_BLOCK)

def parse_value(key: str, raw: str):
    """Try to convert raw strings into int/float/bool when appropriate, else return string."""
    if raw is None:
        return None
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    # ints
    try:
        if raw.isdigit():
            return int(raw)
    except Exception:
        pass
    # floats
    try:
        if "." in raw:
            return float(raw)
    except Exception:
        pass
    return raw

def get_var(name: str, default=None):
    # priority: real env > defaults from VAR_BLOCK > explicit default
    if name in os.environ:
        return parse_value(name, os.environ[name])
    if name in _defaults:
        return parse_value(name, _defaults[name])
    return default

# =========================
# CONFIG (from vars)
# =========================
MODE = str(get_var("MODE", "paper")).lower()
EXCHANGE_ID = str(get_var("EXCHANGE_ID", "kucoinfutures"))
SYMBOLS = [s.strip() for s in str(get_var("SYMBOLS", "")).split(",") if s.strip()]

ENTRY_TF = str(get_var("ENTRY_TF", "1h"))
HTF = str(get_var("HTF", "4h"))
LOOKBACK_DAYS = int(get_var("LOOKBACK_DAYS", 180))

TOTAL_PORTFOLIO_CAPITAL = float(get_var("TOTAL_PORTFOLIO_CAPITAL", 10000.0))
PER_COIN_ALLOCATION = float(get_var("PER_COIN_ALLOCATION", 0.20))
PER_COIN_CAP_USD = TOTAL_PORTFOLIO_CAPITAL * PER_COIN_ALLOCATION

RISK_PERCENT = float(get_var("RISK_PERCENT", 0.02))
RR_FIXED = float(get_var("RR_FIXED", 5.0))
DYNAMIC_RR = bool(get_var("DYNAMIC_RR", True))
MIN_RR = float(get_var("MIN_RR", 4.0))
MAX_RR = float(get_var("MAX_RR", 6.0))

ATR_PERIOD = int(get_var("ATR_PERIOD", 14))
ATR_MULT_SL = float(get_var("ATR_MULT_SL", 1.5))
USE_ATR_STOPS = bool(get_var("USE_ATR_STOPS", True))
USE_H1_FILTER = bool(get_var("USE_HTF_GATE", True))  # HTF gate name mapped

USE_VOLUME_FILTER = bool(get_var("USE_VOLUME_FILTER", False))
VOL_LOOKBACK = int(get_var("VOL_LOOKBACK", 20))
VOL_MIN_RATIO = float(get_var("VOL_MIN_RATIO", 0.5))
RSI_PERIOD = int(get_var("RSI_PERIOD", 14))
RSI_OVERSOLD = float(get_var("RSI_THRESHOLD_LONG", 25))  # mapping names
RSI_OVERBOUGHT = float(get_var("RSI_THRESHOLD_SHORT", 75))

BIAS_CONFIRM_BEAR = int(get_var("BIAS_CONFIRM_BEAR", 2))
COOLDOWN_HOURS = float(get_var("COOLDOWN_HOURS", 0.0))

MAX_DRAWDOWN = float(get_var("MAX_DRAWDOWN", 0.20))
MAX_TRADE_SIZE = float(get_var("MAX_TRADE_SIZE", 100000))
SLIPPAGE_RATE = float(get_var("SLIPPAGE_RATE", 0.0005))
FEE_RATE = float(get_var("FEE_RATE", 0.0006))
INCLUDE_FUNDING = bool(get_var("INCLUDE_FUNDING", True))

TELEGRAM_TOKEN_FUT = str(get_var("TELEGRAM_TOKEN_FUT", ""))
TELEGRAM_CHAT_ID_FUT = str(get_var("TELEGRAM_CHAT_ID_FUT", ""))

API_KEY = str(get_var("KUCOIN_API_KEY", ""))
API_SECRET = str(get_var("KUCOIN_SECRET", ""))
API_PASSPHRASE = str(get_var("KUCOIN_PASSPHRASE", ""))

SEND_DAILY_SUMMARY = bool(get_var("SEND_DAILY_SUMMARY", True))
SUMMARY_HOUR_IST = int(get_var("SUMMARY_HOUR", 20))
SLEEP_CAP = int(get_var("SLEEP_CAP", 60))

DEBUG_COMPARE = bool(get_var("DEBUG_COMPARE", False))
DEBUG_CSV = str(get_var("DEBUG_CSV", "debug_signals_live.csv"))

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
    cfg = {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    if MODE == "live":
        cfg.update({"apiKey": API_KEY, "secret": API_SECRET, "password": API_PASSPHRASE})
    return getattr(ccxt, EXCHANGE_ID)(cfg)

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
        except Exception:
            break
        if not batch:
            break
        out.extend(batch)
        newest = batch[-1][0]
        cursor = (newest + tf_ms) if (last is None or newest > last) else cursor + tf_ms
        last = newest
        if newest >= until_ms - tf_ms:
            break
        time.sleep(pause)
    if not out:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    dedup = {r[0]: r for r in out}
    rows = [dedup[k] for k in sorted(dedup.keys()) if since_ms <= k <= until_ms]
    df = pd.DataFrame(rows, columns=["timestamp","Open","High","Low","Close","Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()[["Open","High","Low","Close","Volume"]]

def fetch_funding_history(exchange, symbol, since_ms, until_ms):
    try:
        rates, cursor = [], since_ms
        while cursor < until_ms:
            page = exchange.fetchFundingRateHistory(symbol, since=cursor, limit=1000)
            if not page:
                break
            rates += page
            newest = page[-1].get('timestamp') or page[-1].get('ts')
            if newest is None or newest <= cursor:
                break
            cursor = newest + 1
            time.sleep(0.1)
        if not rates:
            return None
        df = pd.DataFrame([
            {
                "ts": r.get("timestamp") or r.get("ts"),
                "rate": float(
                    r.get("fundingRate", 0.0) or
                    (r.get("info", {}).get("fundingRate", 0.0) if isinstance(r.get("info", {}), dict) else 0.0)
                ),
            }
            for r in rates
            if (r.get("timestamp") or r.get("ts")) is not None
        ])
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.drop(columns=["ts"]).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception:
        return None

def align_funding_to_index(idx, funding_df):
    s = pd.Series(0.0, index=idx)
    if funding_df is None or funding_df.empty:
        return s
    for ts, row in funding_df.iterrows():
        j = s.index.searchsorted(ts)
        if j < len(s):
            s.iloc[j] = row["rate"]
    return s

# =========================
# INDICATORS & MATH
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

def amount_to_precision_wrapper(exchange, symbol, amt):
    try:
        return float(exchange.amount_to_precision(symbol, amt))
    except Exception:
        return float(round(amt, 8))

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
        "last_exit_time": None,
        "bearish_count": 0
    }

def save_state(state_file, state):
    s = dict(state)
    for k in ["entry_time", "last_processed_ts", "last_exit_time"]:
        if s.get(k) is not None:
            s[k] = pd.to_datetime(s[k]).isoformat()
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

# =========================
# CORE PER-BAR PROCESSOR
# =========================
def process_bar(symbol, h1, h4, state, exchange=None, funding_series=None):
    h1 = h1.copy(); h4 = h4.copy()
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
        h1['Avg_Volume'] = h1['Volume'].rolling(VOL_LOOKBACK).mean()
    h1['RSI'] = calculate_rsi(h1['Close'], RSI_PERIOD)

    i = len(h1) - 1
    if i < 0:
        return state, None

    curr = h1.iloc[i]
    prev_close = h1['Close'].iloc[i-1] if i >= 1 else curr['Close']
    price = float(curr['Close']); open_price = float(curr['Open'])
    bias = int(curr['Bias']); h4t = int(curr['H4_Trend'])
    ts = h1.index[i]

    # funding impact
    if INCLUDE_FUNDING and state["position"] != 0 and funding_series is not None:
        try:
            rate = float(funding_series.iloc[i]) if i < len(funding_series) else 0.0
        except Exception:
            rate = 0.0
        if rate != 0.0 and state["entry_price"] and state["entry_size"]:
            notional = abs(state["entry_price"] * state["entry_size"])
            fee = notional * rate * (1 if state["position"] == 1 else -1) * (-1)
            state["capital"] += fee

    # drawdown forced exit
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    dd = (state["peak_equity"] - state["capital"]) / state["peak_equity"] if state["peak_equity"] > 0 else 0.0
    if dd >= MAX_DRAWDOWN and state["position"] != 0:
        side = "sell" if state["position"] == 1 else "buy"
        exit_price = price
        if MODE == "live":
            try:
                order = place_market(exchange, symbol, side, amount_to_precision_wrapper(exchange, symbol, state["entry_size"]), reduce_only=True)
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
            "Exit_Reason": "MAX DRAWDOWN", "Capital_After": round(state["capital"],2), "Mode": MODE
        }
        state.update({"position":0,"entry_price":0.0,"entry_sl":0.0,"entry_tp":0.0,"entry_time":None,"entry_size":0.0,"bearish_count":0})
        state["last_exit_time"] = ts
        return state, row

    trade_row = None

    # ===== EXIT LOGIC =====
    if state["position"] != 0:
        exit_flag = False; exit_reason = ""; exit_price = price
        if state["position"] == 1:
            if price >= state["entry_tp"]:
                exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
                state["bearish_count"] = 0
            elif price <= state["entry_sl"]:
                exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"
                state["bearish_count"] = 0
            elif USE_H1_FILTER and (h4t < 0 and bias < 0):
                exit_flag, exit_price, exit_reason = True, price, "4H Trend Reversal"
            elif bias < 0:
                state["bearish_count"] += 1
                if state["bearish_count"] >= BIAS_CONFIRM_BEAR:
                    exit_flag, exit_price, exit_reason = True, price, "Bias Reversal"
                    state["bearish_count"] = 0
            else:
                state["bearish_count"] = 0
        else:
            if price <= state["entry_tp"]:
                exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
                state["bearish_count"] = 0
            elif price >= state["entry_sl"]:
                exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"
                state["bearish_count"] = 0
            elif USE_H1_FILTER and (h4t > 0 and bias > 0):
                exit_flag, exit_price, exit_reason = True, price, "4H Trend Reversal"
            elif bias > 0:
                state["bearish_count"] += 1
                if state["bearish_count"] >= BIAS_CONFIRM_BEAR:
                    exit_flag, exit_price, exit_reason = True, price, "Bias Reversal"
                    state["bearish_count"] = 0
            else:
                state["bearish_count"] = 0

        if exit_flag:
            side = "sell" if state["position"]==1 else "buy"
            if MODE == "live":
                try:
                    order = place_market(exchange, symbol, side, amount_to_precision_wrapper(exchange, symbol, state["entry_size"]), reduce_only=True)
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
            state["last_exit_time"] = ts

            emoji = "💚" if pnl>0 else "❤️"
            send_telegram_fut(f"{emoji} EXIT {symbol} {exit_reason} @ {exit_price:.4f} | PnL ${pnl:.2f}")

    # ===== ENTRY LOGIC (mirror long/short) =====
    if state["position"] == 0:
        if COOLDOWN_HOURS>0 and state.get("last_exit_time") is not None:
            try:
                if (ts - state["last_exit_time"]).total_seconds()/3600 < COOLDOWN_HOURS:
                    state["last_processed_ts"] = ts
                    return state, trade_row
            except Exception:
                pass

        bullish_sweep = (price > open_price) and (price > prev_close)
        bearish_sweep = (price < open_price) and (price < prev_close)

        vol_ok_long = True
        vol_ok_short = True
        if USE_VOLUME_FILTER and not pd.isna(h1['Volume'].iloc[i]):
            avgv = h1['Volume'].rolling(VOL_LOOKBACK).mean().iloc[i]
            if not pd.isna(avgv):
                vol_ok_long = h1['Volume'].iloc[i] >= VOL_MIN_RATIO * avgv
                vol_ok_short = vol_ok_long

        rsi = float(h1['RSI'].iloc[i]) if not pd.isna(h1['RSI'].iloc[i]) else None
        rsi_ok_long = True if rsi is None else rsi > RSI_OVERSOLD
        rsi_ok_short = True if rsi is None else rsi < RSI_OVERBOUGHT

        long_ok  = bullish_sweep and vol_ok_long  and rsi_ok_long  and ((not USE_H1_FILTER) or h4t == 1)
        short_ok = bearish_sweep and vol_ok_short and rsi_ok_short and ((not USE_H1_FILTER) or h4t == -1)

        signal = 1 if long_ok else (-1 if short_ok else 0)

        if signal != 0 and (not USE_ATR_STOPS or (USE_ATR_STOPS and not pd.isna(h1['ATR'].iloc[i]) and h1['ATR'].iloc[i] > 0)):
            if signal == 1:
                sl = price - (ATR_MULT_SL * h1['ATR'].iloc[i]) if USE_ATR_STOPS else price * (1 - min(max(price*0.0005,0.0005),0.0015))
                risk = abs(price - sl)
                rr = RR_FIXED
                if DYNAMIC_RR and USE_ATR_STOPS and i >= 6:
                    recent = float(h1['ATR'].iloc[i-5:i].mean())
                    curr = float(h1['ATR'].iloc[i])
                    if recent > 0:
                        if curr > recent*1.2: rr = MIN_RR
                        elif curr < recent*0.8: rr = MAX_RR
                tp = price + rr * risk
            else:
                sl = price + (ATR_MULT_SL * h1['ATR'].iloc[i]) if USE_ATR_STOPS else price * (1 + min(max(price*0.0005,0.0005),0.0015))
                risk = abs(sl - price)
                rr = RR_FIXED
                if DYNAMIC_RR and USE_ATR_STOPS and i >= 6:
                    recent = float(h1['ATR'].iloc[i-5:i].mean())
                    curr = float(h1['ATR'].iloc[i])
                    if recent > 0:
                        if curr > recent*1.2: rr = MIN_RR
                        elif curr < recent*0.8: rr = MAX_RR
                tp = price - rr * risk

            if risk > 0:
                size = position_size_futures(price, sl, state["capital"], RISK_PERCENT, MAX_TRADE_SIZE)
                try:
                    size_q = amount_to_precision_wrapper(exchange, symbol, size)
                except Exception:
                    size_q = round(size, 8)
                if size_q <= 0:
                    state["last_processed_ts"] = ts
                    return state, trade_row

                if size_q > 0:
                    entry_price_used = price
                    side = "buy" if signal==1 else "sell"
                    if MODE == "live":
                        try:
                            order = place_market(exchange, symbol, side, size_q, reduce_only=False)
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
                    state["entry_size"] = size_q
                    state["bearish_count"] = 0

                    pos_val = abs(entry_price_used * size_q)
                    state["capital"] -= pos_val*SLIPPAGE_RATE
                    state["capital"] -= pos_val*FEE_RATE

                    tag = "LONG" if signal==1 else "SHORT"
                    send_telegram_fut(f"🚀 ENTRY {symbol} {tag} @ {entry_price_used:.4f} | SL {sl:.4f} | TP {tp:.4f} | RR {rr:.1f}")

                    if DEBUG_COMPARE:
                        line = f"{symbol},{ts},{signal},{price},{sl},{tp},{size},{size_q},{rr},{h1['ATR'].iloc[i] if 'ATR' in h1.columns else ''},{h1['RSI'].iloc[i]}\n"
                        with open(DEBUG_CSV, "a") as f:
                            f.write(line)

    state["last_processed_ts"] = ts
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    return state, trade_row

# =========================
# WORKER THREAD
# =========================
def worker(symbol):
    state_file, trades_csv = state_files_for_symbol(symbol)
    exchange = get_exchange()
    state = load_state(state_file)

    send_telegram_fut(f"🤖 {symbol} FUTURES bot started | {ENTRY_TF}/{HTF} | cap ${state['capital']:.2f}")

    while True:
        try:
            now = datetime.now(timezone.utc)
            if state.get("last_processed_ts") is not None:
                try:
                    last_ts = pd.to_datetime(state["last_processed_ts"], utc=True)
                    buffer_bars = max(200, ATR_PERIOD * 8)
                    since = last_ts - timedelta(milliseconds=(buffer_bars * timeframe_to_ms(ENTRY_TF)))
                except Exception:
                    since = now - timedelta(days=LOOKBACK_DAYS)
            else:
                since = now - timedelta(days=LOOKBACK_DAYS)

            since_ms = int(since.timestamp()*1000); until_ms = int(now.timestamp()*1000)

            h1 = fetch_ohlcv_range(exchange, symbol, ENTRY_TF, since_ms, until_ms)
            h4 = fetch_ohlcv_range(exchange, symbol, HTF, since_ms, until_ms)
            if h1.empty or h4.empty or len(h1) < 3:
                time.sleep(30); continue

            if len(h1) >= 2:
                h1_closed = h1.iloc[:-1].copy()
            else:
                h1_closed = h1.copy()
            if len(h4) >= 2:
                h4_closed = h4.iloc[:-1].copy()
            else:
                h4_closed = h4.copy()

            closed_ts = h1_closed.index[-1]
            if state["last_processed_ts"] is not None and pd.to_datetime(state["last_processed_ts"]) >= closed_ts:
                time.sleep(30); continue

            funding_series = None
            if INCLUDE_FUNDING:
                fdf = fetch_funding_history(exchange, symbol, int(h1_closed.index[0].timestamp()*1000), int(h1_closed.index[-1].timestamp()*1000))
                funding_series = align_funding_to_index(h1_closed.index, fdf) if (fdf is not None and not fdf.empty) else pd.Series(0.0, index=h1_closed.index)

            prev_ts = state.get("last_processed_ts")
            state, trade = process_bar(symbol, h1_closed, h4_closed, state, exchange=exchange, funding_series=funding_series)
            if trade is not None:
                append_trade(trades_csv, trade)

            if state.get("last_processed_ts") != prev_ts or trade is not None:
                save_state(state_file, state)

            next_close = h1.index[-1] + (h1.index[-1] - h1.index[-2])
            now_utc = datetime.now(timezone.utc)
            if getattr(next_close, "tzinfo", None) is None:
                next_close = next_close.replace(tzinfo=timezone.utc)
            sleep_sec = (next_close - now_utc).total_seconds() + 5
            if sleep_sec < 10: sleep_sec = 10
            if sleep_sec > 3600: sleep_sec = SLEEP_CAP
            time.sleep(sleep_sec)

        except ccxt.RateLimitExceeded:
            time.sleep(10)
        except Exception as e:
            msg = f"{LOG_PREFIX} {symbol} ERROR: {e}"
            print(msg)
            traceback.print_exc()
            send_telegram_fut(msg)
            time.sleep(60)

# =========================
# DAILY SUMMARY
# =========================
def ist_now():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))

def generate_daily_summary():
    try:
        now_ist = ist_now()
        start_ist = datetime(now_ist.year, now_ist.month, now_ist.day, 0, 0, 0, tzinfo=now_ist.tzinfo)
        end_ist   = datetime(now_ist.year, now_ist.month, now_ist.day, 23, 59, 59, tzinfo=now_ist.tzinfo)
        start_utc = start_ist.astimezone(timezone.utc).replace(tzinfo=None)
        end_utc   = end_ist.astimezone(timezone.utc).replace(tzinfo=None)

        lines = [f"📊 FUTURES DAILY SUMMARY — {now_ist.strftime('%Y-%m-%d %I:%M %p IST')}", "-"*60]
        total_cap, total_init, pnl_today, n_today, w_today = 0.0, 0.0, 0.0, 0, 0

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
                if df.empty or 'Exit_DateTime' not in df.columns:
                    today = df.iloc[0:0]
                else:
                    try:
                        df['Exit_DateTime'] = pd.to_datetime(df['Exit_DateTime'], utc=True).dt.tz_convert(None)
                        today = df[(df['Exit_DateTime'] >= start_utc) & (df['Exit_DateTime'] <= end_utc)]
                    except Exception:
                        today = df.iloc[0:0]

                n_trades_today = len(today)
                wins_today = int(today['Win'].sum()) if n_trades_today else 0
                pnl_today_sym = float(today['PnL_$'].sum()) if n_trades_today else 0.0

                if not df.empty:
                    pnl_all = float(df['PnL_$'].sum())
                    wins = int(df['Win'].sum()); losses = len(df)-wins
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
🚀 Futures Bot Started
Mode: {MODE.upper()}
Exchange: {EXCHANGE_ID}
Symbols: {", ".join(SYMBOLS)}
TF: {ENTRY_TF}/{HTF}
Cap/coin: ${PER_COIN_CAP_USD:,.2f}
Risk: {RISK_PERCENT*100:.1f}% | Fee: {FEE_RATE*100:.3f}% | Slippage: {SLIPPAGE_RATE*100:.3f}%
Funding: {"ON" if INCLUDE_FUNDING else "OFF"}
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

