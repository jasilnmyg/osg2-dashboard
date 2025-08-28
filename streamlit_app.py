import streamlit as st
import ccxt
import pandas as pd
import requests
import time
from datetime import datetime, timezone

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="SMC Alert Bot", layout="wide")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
chat_id = st.sidebar.text_input("Telegram Chat ID")
exchange_id = st.sidebar.selectbox("Exchange", ["binance"])
symbol = st.sidebar.text_input("Trading Pair", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1h","4h","1d"], index=1)
fetch_limit = st.sidebar.slider("Candles to fetch", 100, 500, 200)
poll_seconds = st.sidebar.slider("Refresh seconds", 10, 300, 60)

# Strategy parameters
bos_lookback = st.sidebar.slider("BOS Lookback", 10, 100, 30)
min_body_pct = st.sidebar.slider("Min OB Body %", 0.1, 1.0, 0.25)
fvg_min_pct = st.sidebar.slider("Min FVG %", 0.001, 0.01, 0.002)
retest_pct = st.sidebar.slider("Retest tolerance %", 0.001, 0.02, 0.005)
max_ob_age = st.sidebar.slider("Max OB Age (candles)", 10, 100, 50)

# ----------------- INIT -----------------
exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

# ----------------- HELPERS -----------------
def send_telegram(message: str):
    if not bot_token or not chat_id:
        return
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(telegram_url, data=payload, timeout=10)
    except Exception as e:
        st.error(f"Telegram error: {e}")

def fetch_ohlcv(symbol, timeframe, limit=200):
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def is_bullish(c): return c["close"] > c["open"]
def is_bearish(c): return c["close"] < c["open"]
def body_size(c): return abs(c["close"]-c["open"])
def rng(c): return c["high"]-c["low"]

def detect_bos(df):
    if len(df) < bos_lookback+2:
        return None
    recent = df.tail(bos_lookback+1)
    last = recent.iloc[-1]["close"]
    if last > recent.iloc[:-1]["close"].max():
        return "bull"
    if last < recent.iloc[:-1]["close"].min():
        return "bear"
    return None

def find_order_block(df, direction="bull"):
    for idx in range(len(df)-2, -1, -1):
        c = df.iloc[idx]
        if rng(c) == 0: continue
        if body_size(c)/rng(c) < min_body_pct: continue
        age = len(df)-1-idx
        if age > max_ob_age: continue
        if direction=="bull" and is_bearish(c):
            if idx+2 < len(df) and is_bullish(df.iloc[idx+1]) and df.iloc[idx+1]["close"] > c["high"]:
                return {"zone_high":c["high"],"zone_low":c["low"],"type":"bullish_OB"}
        if direction=="bear" and is_bullish(c):
            if idx+2 < len(df) and is_bearish(df.iloc[idx+1]) and df.iloc[idx+1]["close"] < c["low"]:
                return {"zone_high":c["high"],"zone_low":c["low"],"type":"bearish_OB"}
    return None

def detect_fvg(df):
    if len(df)<4: return None
    for i in range(len(df)-3,0,-1):
        a,b,c = df.iloc[i-1],df.iloc[i],df.iloc[i+1]
        if is_bearish(b) and a["high"]<c["low"]:  # bullish FVG
            gap_high,gap_low = c["low"],a["high"]
            if (gap_high-gap_low)/c["close"]>=fvg_min_pct:
                return {"zone_high":gap_high,"zone_low":gap_low,"type":"bullish_FVG"}
        if is_bullish(b) and a["low"]>c["high"]:  # bearish FVG
            gap_high,gap_low = a["low"],c["high"]
            if (gap_high-gap_low)/c["close"]>=fvg_min_pct:
                return {"zone_high":gap_high,"zone_low":gap_low,"type":"bearish_FVG"}
    return None

def price_in_zone(price, high, low, pct):
    tol=pct*price
    return low-tol<=price<=high+tol

def analyze_signal(df):
    bos=detect_bos(df)
    latest=df.iloc[-1]["close"]
    if not bos: return {"signal":None}

    ob=find_order_block(df,direction="bull" if bos=="bull" else "bear")
    fvg=detect_fvg(df)

    if bos=="bull" and ob and fvg:
        if price_in_zone(latest, max(ob["zone_high"],fvg["zone_high"]), min(ob["zone_low"],fvg["zone_low"]), retest_pct):
            stop=ob["zone_low"]-(ob["zone_high"]-ob["zone_low"])*0.5
            target=latest+(latest-stop)*1.5
            return {"signal":"BUY","reason":"Bullish BOS + OB+FVG retest","price":latest,"stop":stop,"target":target}
    if bos=="bear" and ob and fvg:
        if price_in_zone(latest, max(ob["zone_high"],fvg["zone_high"]), min(ob["zone_low"],fvg["zone_low"]), retest_pct):
            stop=ob["zone_high"]+(ob["zone_high"]-ob["zone_low"])*0.5
            target=latest-(stop-latest)*1.5
            return {"signal":"SELL","reason":"Bearish BOS + OB+FVG retest","price":latest,"stop":stop,"target":target}
    return {"signal":None}

# ----------------- UI -----------------
st.title("📊 Smart Money Concept (SMC) Alert Bot")
st.markdown("Alerts generated using **BOS + OB + FVG Retest** strategy.")

log_box = st.empty()
chart_placeholder = st.empty()

if st.sidebar.button("▶️ Start Bot"):
    st.success("Bot started! Leave this app running.")
    last_alert=None
    while True:
        try:
            df=fetch_ohlcv(symbol,timeframe,limit=fetch_limit)
            signal=analyze_signal(df)

            if signal["signal"]:
                msg=(f"<b>{signal['signal']} ALERT</b>\n"
                     f"{symbol} {timeframe}\n"
                     f"Reason: {signal['reason']}\n"
                     f"Price: {signal['price']:.2f}\n"
                     f"Stop: {signal['stop']:.2f}\n"
                     f"Target: {signal['target']:.2f}\n"
                     f"Time: {datetime.now(timezone.utc).astimezone().isoformat()}")
                if msg!=last_alert:
                    send_telegram(msg)
                    log_box.info(msg)
                    last_alert=msg

            chart_placeholder.line_chart(df.set_index("dt")["close"])
        except Exception as e:
            log_box.error(f"Error: {e}")

        time.sleep(poll_seconds)
