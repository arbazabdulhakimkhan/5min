"""
paper_live_bot.py

- Based on your original backtest code, converted to a live-like paper trading loop.
- Paper-mode: it uses live exchange OHLCV data but NEVER places real orders (default).
- Simulates fees, slippage, funding (mark-price based), liquidation checks, ATR SL/TP logic, cooldowns.
- Persists trades to CSV (trades_paper.csv) and prints & logs useful info.
- Configure via top-level variables or .env (optional).
"""

import os
import time
import math
import json
import logging
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ------------------------
# Basic config — edit or set env vars
# ------------------------
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kucoinfutures")
SYMBOL = os.getenv("SYMBOL", "DOGE/USDT:USDT")
TIMEFRAME_ENTRY = os.getenv("TF_ENTRY", "1h")   # candle used for entries/exits
TIMEFRAME_FILTER = os.getenv("TF_FILTER", "4h") # higher timeframe filter
DAYS_BACK = int(os.getenv("DAYS_BACK", "10"))

TOTAL_PORTFOLIO_CAPITAL = float(os.getenv("TOTAL_PORTFOLIO_CAPITAL", "10000.0"))
PER_COIN_ALLOCATION = float(os.getenv("PER_COIN_ALLOCATION", "0.20"))
INITIAL_CAPITAL = TOTAL_PORTFOLIO_CAPITAL * PER_COIN_ALLOCATION

# Strategy params (keep as your original defaults)
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.02"))
RR_FIXED = float(os.getenv("RR_FIXED", "5.0"))
DYNAMIC_RR = os.getenv("DYNAMIC_RR", "true").lower() in ("1","true","yes")
MIN_RR = float(os.getenv("MIN_RR", "4.0"))
MAX_RR = float(os.getenv("MAX_RR", "6.0"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.5"))
USE_ATR_STOPS = os.getenv("USE_ATR_STOPS", "true").lower() in ("1","true","yes")
USE_H1_FILTER = os.getenv("USE_H1_FILTER", "true").lower() in ("1","true","yes")

# Costs / simulation
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.20"))
MAX_TRADE_SIZE = float(os.getenv("MAX_TRADE_SIZE", "100000"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))

# Filters
USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "false").lower() in ("1","true","yes")
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MIN_RATIO = float(os.getenv("VOL_MIN_RATIO", "0.5"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", "25"))
BIAS_CONFIRM_BEAR = int(os.getenv("BIAS_CONFIRM_BEAR", "2"))
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "0.0"))

# Funding & liquidation simulation
INCLUDE_FUNDING = os.getenv("INCLUDE_FUNDING", "true").lower() in ("1","true","yes")
FUNDING_INTERVAL_HOURS = int(os.getenv("FUNDING_INTERVAL_HOURS", "8"))
LIQUIDATION_PENALTY_RATE = float(os.getenv("LIQUIDATION_PENALTY_RATE", "0.005"))
LEVERAGE = float(os.getenv("LEVERAGE", "1.0"))

# Paper/live behavior
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() in ("1","true","yes")
USE_LIVE_MARKET = os.getenv("USE_LIVE_MARKET", "true").lower() in ("1","true","yes")  # when False, only historical backtest

TRADE_CSV_FILENAME = os.getenv("TRADE_CSV_FILENAME", f"{SYMBOL.replace('/', '_').replace(':','_')}_trades_paper.csv")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

# ------------------------
# Helpers
# ------------------------
def timeframe_to_ms(tf: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    num = int(''.join([c for c in tf if c.isdigit()]))
    unit = ''.join([c for c in tf if c.isalpha()])
    return num * units[unit]

def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.headers = {"User-Agent": "paper-live-bot/1.0"}
    return ex

# Robust OHLCV fetcher (keeps your original structure)
def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1500, pause=0.2):
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
            logging.warning(f"fetch_ohlcv error @ {datetime.utcfromtimestamp(cursor/1000)}: {e}")
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

# Indicators
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

# Funding fetch (best-effort; if not supported, returns empty DataFrame)
def fetch_funding_history(exchange, symbol, since_ms, until_ms):
    rates = []
    cursor = since_ms
    while cursor < until_ms:
        try:
            page = exchange.fetchFundingRateHistory(symbol, since=cursor, limit=1000)
        except Exception as e:
            logging.info(f"Funding history fetch error (fallback to none): {e}")
            break
        if not page:
            break
        rates.extend(page)
        newest = page[-1]['timestamp']
        if newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(0.05)
    if not rates:
        return pd.DataFrame(columns=["timestamp","rate"])
    df = pd.DataFrame([{"timestamp": r['timestamp'], "rate": float(r.get('fundingRate', r.get('info', {}).get('fundingRate', 0.0)))} for r in rates])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df.sort_index()

def align_funding_schedule(index, funding_df):
    s = pd.Series(0.0, index=index)
    if funding_df is None or funding_df.empty:
        return s
    for ts, row in funding_df.iterrows():
        idx = s.index.searchsorted(ts)
        if idx < len(s):
            s.iloc[idx] = row["rate"]
    return s

# Liquidation tier fallback
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
        logging.info(f"Leverage tiers fetch failed (using default mmr 0.005): {e}")
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

# Position sizing (uses current mark price)
def calculate_futures_position_size(price, sl, capital, risk_percent, max_trade_size):
    risk_per_trade = capital * risk_percent
    risk_per_contract = abs(price - sl)
    if risk_per_contract <= 0:
        return 0.0
    max_by_risk = (risk_per_trade / risk_per_contract)
    max_by_capital = capital / price
    size_base = min(max_by_risk, max_by_capital, max_trade_size / price)
    return max(size_base, 0.0)

# ------------------------
# Main "paper-live" loop
# ------------------------
def run_paper_live():
    exchange = get_exchange()
    now_utc = datetime.now(timezone.utc)
    since_dt = now_utc - timedelta(days=DAYS_BACK)
    until_dt = now_utc
    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    logging.info(f"Fetching historical candle heads for warmup: {TIMEFRAME_ENTRY} & {TIMEFRAME_FILTER}")
    h1 = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME_ENTRY, since_ms, until_ms, limit=1500, pause=0.12)
    h4 = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME_FILTER, since_ms, until_ms, limit=1500, pause=0.12)
    if h1.empty or h4.empty:
        logging.error("No OHLCV data fetched. Exiting.")
        return

    # Prepare indicators (initial)
    h1['Bias'] = 0
    h1.loc[h1['Close'] > h1['Close'].shift(1), 'Bias'] = 1
    h1.loc[h1['Close'] < h1['Close'].shift(1), 'Bias'] = -1

    h4['Trend'] = 0
    h4.loc[h4['Close'] > h4['Close'].shift(1), 'Trend'] = 1
    h4.loc[h4['Close'] < h4['Close'].shift(1), 'Trend'] = -1

    h1['H4_Trend'] = h4['Trend'].reindex(h1.index, method='ffill').fillna(0).astype(int)

    if USE_ATR_STOPS:
        h1['ATR'] = calculate_atr(h1, ATR_PERIOD)
    else:
        h1['ATR'] = 0.0

    if USE_VOLUME_FILTER:
        h1['Avg_Volume'] = h1['Volume'].rolling(window=VOL_LOOKBACK).mean()

    h1['RSI'] = calculate_rsi(h1['Close'], RSI_PERIOD)

    # fetch funding and tiers once (we'll align funding series to H1 timestamps)
    funding_df = fetch_funding_history(exchange, SYMBOL, since_ms, until_ms) if INCLUDE_FUNDING else pd.DataFrame()
    funding_series = align_funding_schedule(h1.index, funding_df) if INCLUDE_FUNDING else pd.Series(0.0, index=h1.index)
    tiers = fetch_leverage_tiers(exchange, SYMBOL)

    # backtest state variables (but run continuously)
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
    last_exit_time = None

    trades = []
    equity_curve = []

    # helper to persist trade immediately
    def persist_trade(tr):
        nonlocal trades
        trades.append(tr)
        # append to CSV
        df = pd.DataFrame([tr])
        header = not os.path.exists(TRADE_CSV_FILENAME)
        df.to_csv(TRADE_CSV_FILENAME, mode='a', index=False, header=header)

    logging.info("Starting live-paper loop. Polling for new closed candles... (press Ctrl+C to stop)")

    # infinite loop
    try:
        while True:
            # re-fetch latest H1 & H4 head (short window)
            try:
                h1_head = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME_ENTRY,
                                            int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp()*1000),
                                            int(datetime.now(timezone.utc).timestamp()*1000),
                                            limit=500, pause=0.12)
                h4_head = fetch_ohlcv_range(exchange, SYMBOL, TIMEFRAME_FILTER,
                                            int((datetime.now(timezone.utc) - timedelta(days=10)).timestamp()*1000),
                                            int(datetime.now(timezone.utc).timestamp()*1000),
                                            limit=500, pause=0.12)
                if h1_head.empty or h4_head.empty:
                    logging.warning("no fresh data; sleeping 10s")
                    time.sleep(10)
                    continue
            except Exception as e:
                logging.exception("error fetching heads; sleeping 10s")
                time.sleep(10)
                continue

            # recalc indicators on head
            h1 = h1_head.copy()
            h4 = h4_head.copy()
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

            # align funding to latest index
            if INCLUDE_FUNDING:
                funding_series = align_funding_schedule(h1.index, funding_df)

            # evaluate only on the latest closed candle
            if len(h1) < 3:
                time.sleep(5); continue
            i = -1
            ts = h1.index[i]
            price = float(h1['Close'].iat[i])
            open_price = float(h1['Open'].iat[i])
            prev_close = float(h1['Close'].iat[i-1])
            h4_trend = int(h1['H4_Trend'].iat[i])
            bias = int(h1['Bias'].iat[i])
            rsi_val = h1['RSI'].iat[i] if not pd.isna(h1['RSI'].iat[i]) else None

            # apply funding if occurs now for existing position (use mark-price notional)
            if INCLUDE_FUNDING and position != 0:
                # find funding rate aligned to this candle index
                try:
                    idxpos = list(h1.index).index(ts)
                    rate = float(funding_series.iloc[idxpos])
                except Exception:
                    rate = 0.0
                if rate != 0.0:
                    notional = abs(price * entry_size)
                    # longs pay positive rate, shorts receive positive rate
                    funding_pnl = -rate * notional if position == 1 else rate * notional
                    capital += funding_pnl
                    logging.info(f"Funding applied: rate={rate:.8f} funding_pnl={funding_pnl:.4f} capital={capital:.2f}")

            # drawdown check
            if equity_curve:
                peak_equity = max(equity_curve)
                curr_dd = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0.0
                if curr_dd >= MAX_DRAWDOWN and not permanently_stopped:
                    permanently_stopped = True
                    logging.warning(f"PERMANENT STOP at {ts} | DD: {curr_dd*100:.2f}%")
                    if position != 0:
                        exit_price = price
                        gross_pnl = entry_size * (exit_price - entry_price) * (1 if position == 1 else -1)
                        position_value = abs(exit_price * entry_size)
                        exit_slippage = position_value * SLIPPAGE_RATE
                        exit_fee = position_value * FEE_RATE
                        net_pnl = gross_pnl - exit_slippage - exit_fee
                        capital += net_pnl
                        tr = {
                            'Trade_ID': len(trades)+1,
                            'Entry_DateTime': entry_time,
                            'Exit_DateTime': ts,
                            'Position': 'Long' if position==1 else 'Short',
                            'Entry_Price': round(entry_price, 8),
                            'Exit_Price': round(exit_price, 8),
                            'Take_Profit': round(entry_tp, 8),
                            'Stop_Loss': round(entry_sl, 8),
                            'Position_Size_Base': round(entry_size, 8),
                            'PnL_$': round(net_pnl, 6),
                            'Win': 1 if net_pnl>0 else 0,
                            'Exit_Reason': 'MAX_DRAWDOWN',
                            'Capital_After': round(capital,2)
                        }
                        persist_trade(tr)
                        position = 0
                        liq_lower = liq_upper = None
                        entry_notional = 0.0

            # liquidation check (conservative)
            if position != 0 and not permanently_stopped:
                if position == 1 and liq_lower is not None and price <= liq_lower:
                    position_value = abs(price * entry_size)
                    penalty = position_value * LIQUIDATION_PENALTY_RATE
                    gross_pnl = entry_size * (price - entry_price)
                    net_pnl = gross_pnl - penalty
                    capital += net_pnl
                    tr = {
                        'Trade_ID': len(trades)+1,
                        'Entry_DateTime': entry_time,
                        'Exit_DateTime': ts,
                        'Position': 'Long',
                        'Entry_Price': round(entry_price, 8),
                        'Exit_Price': round(price, 8),
                        'Take_Profit': round(entry_tp, 8),
                        'Stop_Loss': round(entry_sl, 8),
                        'Position_Size_Base': round(entry_size, 8),
                        'PnL_$': round(net_pnl, 6),
                        'Win': 1 if net_pnl>0 else 0,
                        'Exit_Reason': 'LIQUIDATION',
                        'Capital_After': round(capital,2)
                    }
                    persist_trade(tr)
                    logging.warning("LIQUIDATION occurred: %s", tr)
                    position = 0
                    liq_lower = liq_upper = None
                    entry_notional = 0.0
                elif position == -1 and liq_upper is not None and price >= liq_upper:
                    position_value = abs(price * entry_size)
                    penalty = position_value * LIQUIDATION_PENALTY_RATE
                    gross_pnl = entry_size * (entry_price - price)
                    net_pnl = gross_pnl - penalty
                    capital += net_pnl
                    tr = {
                        'Trade_ID': len(trades)+1,
                        'Entry_DateTime': entry_time,
                        'Exit_DateTime': ts,
                        'Position': 'Short',
                        'Entry_Price': round(entry_price, 8),
                        'Exit_Price': round(price, 8),
                        'Take_Profit': round(entry_tp, 8),
                        'Stop_Loss': round(entry_sl, 8),
                        'Position_Size_Base': round(entry_size, 8),
                        'PnL_$': round(net_pnl, 6),
                        'Win': 1 if net_pnl>0 else 0,
                        'Exit_Reason': 'LIQUIDATION',
                        'Capital_After': round(capital,2)
                    }
                    persist_trade(tr)
                    logging.warning("LIQUIDATION occurred: %s", tr)
                    position = 0
                    liq_lower = liq_upper = None
                    entry_notional = 0.0

            # normal exit logic (close-based SL/TP and trend/bias flips)
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
                    tr = {
                        'Trade_ID': len(trades)+1,
                        'Entry_DateTime': entry_time,
                        'Exit_DateTime': ts,
                        'Position': 'Long' if position==1 else 'Short',
                        'Entry_Price': round(entry_price, 8),
                        'Exit_Price': round(exit_price, 8),
                        'Take_Profit': round(entry_tp, 8),
                        'Stop_Loss': round(entry_sl, 8),
                        'Position_Size_Base': round(entry_size, 8),
                        'PnL_$': round(net_pnl, 6),
                        'Win': 1 if net_pnl>0 else 0,
                        'Exit_Reason': exit_reason,
                        'Capital_After': round(capital,2)
                    }
                    persist_trade(tr)
                    logging.info("Exit executed (paper): %s", tr)
                    position = 0
                    entry_price = entry_sl = entry_tp = 0.0
                    entry_time = None
                    entry_size = 0.0
                    liq_lower = liq_upper = None
                    entry_notional = 0.0
                    last_exit_time = ts

            # entry logic (only if flat)
            if position == 0 and not permanently_stopped:
                # compute entry signal (your original rules)
                bullish_sweep = (price > open_price) and (price > prev_close)
                vol_ok_long = True
                if USE_VOLUME_FILTER and not pd.isna(h1['Avg_Volume'].iat[i]):
                    vol_ok_long = h1['Volume'].iat[i] >= VOL_MIN_RATIO * h1['Avg_Volume'].iat[i]
                rsi_ok_long = True if pd.isna(rsi_val) else (rsi_val > RSI_OVERSOLD)
                h4_ok_long = (not USE_H1_FILTER) or (h4_trend == 1)

                bearish_sweep = (price < open_price) and (price < prev_close)
                vol_ok_short = True
                if USE_VOLUME_FILTER and not pd.isna(h1['Avg_Volume'].iat[i]):
                    vol_ok_short = h1['Volume'].iat[i] >= VOL_MIN_RATIO * h1['Avg_Volume'].iat[i]
                rsi_ok_short = True if pd.isna(rsi_val) else (rsi_val < (100 - RSI_OVERSOLD))
                h4_ok_short = (not USE_H1_FILTER) or (h4_trend == -1)

                signal = 0
                if bullish_sweep and vol_ok_long and rsi_ok_long and h4_ok_long:
                    signal = 1
                elif bearish_sweep and vol_ok_short and rsi_ok_short and h4_ok_short:
                    signal = -1

                if signal != 0:
                    # cooldown check
                    if last_exit_time is not None:
                        if (ts - last_exit_time) < timedelta(hours=COOLDOWN_HOURS):
                            logging.info("In cooldown period; skipping entry.")
                            equity_curve.append(capital)
                            time.sleep(2)
                            continue

                    # require ATR if enabled
                    if USE_ATR_STOPS and (pd.isna(h1['ATR'].iat[i]) or h1['ATR'].iat[i] <= 0):
                        equity_curve.append(capital)
                        time.sleep(1)
                        continue

                    # SL/TP
                    if signal == 1:
                        sl = price - (ATR_MULT_SL * h1['ATR'].iat[i]) if USE_ATR_STOPS else price * (1 - 0.0005)
                        risk = abs(price - sl)
                        if risk <= 0:
                            equity_curve.append(capital); time.sleep(1); continue
                        rr_ratio = RR_FIXED
                        if DYNAMIC_RR and USE_ATR_STOPS and not pd.isna(h1['ATR'].iat[i]) and len(h1) >= 7:
                            recent_atr = float(h1['ATR'].iloc[max(0, len(h1)-6):len(h1)-1].mean())
                            current_atr = float(h1['ATR'].iat[i])
                            if recent_atr > 0:
                                if current_atr > recent_atr * 1.2: rr_ratio = MIN_RR
                                elif current_atr < recent_atr * 0.8: rr_ratio = MAX_RR
                        tp = price + rr_ratio * risk
                    else:
                        sl = price + (ATR_MULT_SL * h1['ATR'].iat[i]) if USE_ATR_STOPS else price * (1 + 0.0005)
                        risk = abs(sl - price)
                        if risk <= 0:
                            equity_curve.append(capital); time.sleep(1); continue
                        rr_ratio = RR_FIXED
                        if DYNAMIC_RR and USE_ATR_STOPS and not pd.isna(h1['ATR'].iat[i]) and len(h1) >= 7:
                            recent_atr = float(h1['ATR'].iloc[max(0, len(h1)-6):len(h1)-1].mean())
                            current_atr = float(h1['ATR'].iat[i])
                            if recent_atr > 0:
                                if current_atr > recent_atr * 1.2: rr_ratio = MIN_RR
                                elif current_atr < recent_atr * 0.8: rr_ratio = MAX_RR
                        tp = price - rr_ratio * risk

                    # position sizing (mark price based)
                    size_base = calculate_futures_position_size(price, sl, capital, RISK_PERCENT, MAX_TRADE_SIZE)
                    if size_base > 0:
                        # simulate entry fees + slippage
                        position = 1 if signal == 1 else -1
                        entry_price = price
                        entry_sl = sl
                        entry_tp = tp
                        entry_time = ts
                        entry_size = size_base
                        position_value = abs(entry_price * entry_size)
                        entry_slippage = position_value * SLIPPAGE_RATE
                        entry_fee = position_value * FEE_RATE
                        capital -= (entry_slippage + entry_fee)
                        entry_notional = position_value
                        mmr = get_mmr_for_notional(tiers, entry_notional)
                        liq_lower, liq_upper = estimate_liquidation_bounds(entry_price, entry_notional, position, mmr, leverage=LEVERAGE)
                        logging.info(f"PAPER ENTRY { 'LONG' if position==1 else 'SHORT'} @{entry_price:.8f} size={entry_size:.6f} SL={entry_sl:.8f} TP={entry_tp:.8f} capital={capital:.2f}")
                        # persist entry as a partial record (final saved on exit)
                    else:
                        logging.info("Calculated size <= 0. Skipping entry.")

            # append equity curve and sleep a bit
            equity_curve.append(capital)
            time.sleep(5)  # poll every 5 seconds (tune as required)

    except KeyboardInterrupt:
        logging.info("Interrupted by user — exiting loop and printing summary.")
    except Exception as e:
        logging.exception("Unhandled exception in main loop: %s", e)

    # final summary
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = int(trades_df['Win'].sum()) if 'Win' in trades_df.columns else 0
        total = len(trades_df)
        total_pnl = float(trades_df['PnL_$'].sum()) if 'PnL_$' in trades_df.columns else trades_df['pnl'].sum()
        final_capital = capital
        logging.info("PAPER RUN SUMMARY: trades=%d wins=%d total_pnl=%.2f final_capital=%.2f", total, wins, total_pnl, final_capital)
    else:
        logging.info("No trades executed in this paper run.")

if __name__ == "__main__":
    run_paper_live()
