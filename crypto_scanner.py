import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime

# ── WATCHLIST ────────────────────────────────────────
watchlist = [
    "LDO-USD",
    "PROVE-USD",
    "SUI20947-USD",
    "DOGE-USD",
    "WLD-USD",
    "AAVE-USD",
    "BTC-USD",
    "ETH-USD",
    "ONDO-USD",
    "UB38339-USD",
    "JTO-USD",
    "PEPE-22478",
    "BNB-USD",
    "TAO22974-USD",
    "FIDA-USD",
    "PENDLE-USD"
]

# ── PARAMETERS ───────────────────────────────────────
turnover_threshold    = 2
force_exit_turnover   = 1.0
max_prev_candle_move  = 0.05   # skip if prev candle already moved 5%+
recent_candle_window  = 3      # check last N candles for signal

# ── ROLLING SLOPE ────────────────────────────────────
def rolling_slope(series, window):
    slopes = []
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
        else:
            y = series.iloc[i - window:i].values
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)

# ── GET SIGNAL ───────────────────────────────────────
def get_signal(ticker):
    try:
        df = yf.download(
            ticker,
            interval="1h",
            period="60d",
            progress=False
        )

        if len(df) < 200:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.dropna()

        if len(df) < 336:
            return None

        # ── INDICATORS ───────────────────────────────
        df['hour']             = df.index.hour
        df['dollar_volume']    = df['Volume']
        df['regression_slope'] = rolling_slope(df['Close'], 336)

        df['avg_dollar_volume'] = (
            df['dollar_volume']
            .rolling(24)
            .mean()
            .shift(1)
        )
        df['turnover_ratio'] = (
            df['dollar_volume'].shift(1)
            / df['avg_dollar_volume']
        )

        df['ema']          = df['Close'].ewm(span=200, adjust=False).mean()
        df['returns']      = df['Close'].pct_change()
        df['volatility']   = df['returns'].rolling(24).std()
        df['avg_volatility'] = df['volatility'].rolling(168).mean()
        df['price_change'] = df['Close'].pct_change().shift(1)

        # ── MARKET STRUCTURE ─────────────────────────
        window = 168
        df['local_high'] = df['High'].rolling(window).max().shift(1)
        df['local_low']  = df['Low'].rolling(window).min().shift(1)
        df['lower_high'] = df['local_high'] < df['local_high'].shift(window)
        df['lower_low']  = df['local_low']  < df['local_low'].shift(window)
        df['bearish']    = df['lower_high'] & df['lower_low']

        df = df.dropna()

        if len(df) == 0:
            return None

        # ── CONDITIONS ───────────────────────────────
        buy_condition = (
            (df['turnover_ratio'] > turnover_threshold) &
            (df['hour'].between(0, 15)) &
            (df['regression_slope'] > 0) &
            (~df['bearish']) &
            (df['price_change']>0) &  # fixed
            (df['volatility'] > df['avg_volatility'])
        )

        exit_condition = (
            (df['turnover_ratio'] < force_exit_turnover) &
            (df['Close'] < df['ema'])
        )

        buy_signals  = df[buy_condition].index
        exit_signals = df[exit_condition].index

        # ── CHECK RECENT CANDLES ─────────────────────
        recent_candles = df.index[-recent_candle_window:]

        signal      = "STAND STILL"
        signal_time = None

        if len(buy_signals) > 0 and buy_signals[-1] in recent_candles:
            signal      = "BUY"
            signal_time = buy_signals[-1]
        elif len(exit_signals) > 0 and exit_signals[-1] in recent_candles:
            signal      = "EXIT"
            signal_time = exit_signals[-1]

        # ── OUTPUT ───────────────────────────────────
        latest      = df.iloc[-1]
        candle_move = abs(float(latest['price_change'])) * 100

        return {
            "ticker":      ticker,
            "price":       round(float(latest['Close']), 4),
            "turnover":    round(float(latest['turnover_ratio']), 2),
            "slope":       round(float(latest['regression_slope']), 6),
            "volatility":  round(float(latest['volatility']), 4),
            "candle_move": round(candle_move, 2),
            "signal":      signal,
            "signal_time": str(signal_time) if signal_time else None
        }

    except Exception as e:
        print(f"  ❌ {ticker} error: {e}")
        return None

# ── MAIN ─────────────────────────────────────────────
print("🔍 Scanning crypto watchlist...")
print("=" * 50)

signals_found = []

for ticker in watchlist:
    print(f"  Checking {ticker}...", end=" ")
    result = get_signal(ticker)

    if result and result["signal"] != "STAND STILL":
        signals_found.append(result)
        print(f"✅ {result['signal']} at ${result['price']}")
    else:
        print("❌ No signal")

print("=" * 50)
print(f"📊 Scan complete. Found {len(signals_found)} signals.")

# ── SAVE OUTPUT ──────────────────────────────────────
if signals_found:
    with open('signal_output.txt', 'w') as f:
        for s in signals_found:
            f.write(f"Ticker:      {s['ticker']}\n")
            f.write(f"Price:       ${s['price']}\n")
            f.write(f"Signal:      {s['signal']}\n")
            f.write(f"Turnover:    {s['turnover']}x\n")
            f.write(f"Candle Move: {s['candle_move']}%\n")
            f.write(f"Volatility:  {s['volatility']}\n")
            f.write(f"Time:        {s['signal_time']}\n")
            f.write("━━━━━━━━━━━━━━━━━━━━\n")

    with open('signal_output.json', 'w') as f:
        json.dump(signals_found, f, indent=2)

    print("\n🚨 SIGNALS DETECTED:")
    for s in signals_found:
        emoji = "🟢" if s['signal'] == "BUY" else "🔴"
        print(f"  {emoji} {s['ticker']}: {s['signal']} @ ${s['price']} | candle move: {s['candle_move']}%")

else:
    with open('signal_output.txt', 'w') as f:
        f.write("NO_SIGNAL")
    print("\n😴 No active signals at this time")
