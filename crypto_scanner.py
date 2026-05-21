import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime

# ── WATCHLIST (FIXED) ────────────────────────────────────
watchlist = [
    "LDO-USD",
    "SUI20947-USD",
    "DOGE-USD",
    "RPL-USD",
    "AAVE-USD",
    "BTC-USD",
    "ETH-USD",
    "ONDO-USD",
    "TRX-USD",
    "JTO-USD",
    "TAO22974-USD",           # Fixed from TAO22974-USD
    "FIDA-USD",
    "PENDLE-USD"
]

# ── PARAMETERS ───────────────────────────────────
turnover_threshold   = 2
force_exit_turnover  = 1.0

# ── FUNCTION: ROLLING SLOPE ─────────────────────
def rolling_slope(series, window):
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)

# ── FUNCTION: GET SIGNAL FOR TICKER ─────────────
def get_signal(ticker):
    try:
        # Download data
        df = yf.download(ticker, interval="1h", period="60d", progress=False)
        
        if len(df) < 200:  # Need minimum data
            return None
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = df.dropna()
        
        if len(df) < 336:  # Need enough for 14-day slope
            return None
        
        # ── FIXED: DOLLAR VOLUME (Volume * Price) ──
        df['dollar_volume'] = df['Volume'] * df['Close']
        
        # ── FEATURES ─────────────────────────────
        df['hour'] = df.index.hour
        df['regression_slope'] = rolling_slope(df['Close'], 336)
        
        df['avg_dollar_volume'] = df['dollar_volume'].rolling(24).mean().shift(1)
        df['turnover_ratio'] = df['dollar_volume'].shift(1) / df['avg_dollar_volume']
        
        df['ema'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std()
        df['avg_volatility'] = df['volatility'].rolling(168).mean()
        
        # ── FIXED: NO LOOK-AHEAD BIAS ────────────
        window = 168
        df['local_high'] = df['High'].rolling(window).max()
        df['local_low'] = df['Low'].rolling(window).min()
        df['lower_high'] = df['local_high'] < df['local_high'].shift(window)
        df['lower_low'] = df['local_low'] < df['local_low'].shift(window)
        df['bearish'] = df['lower_high'] & df['lower_low']
        
        # Drop NaN rows from rolling calculations
        df = df.dropna()
        
        if len(df) == 0:
            return None
        
        # ── BUY CONDITION ────────────────────────
        buy_condition = (
            (df['turnover_ratio'] > turnover_threshold) &
            (df['hour'] >= 0) & (df['hour'] <= 15) &
            (df['regression_slope'] > 0) &
            (~df['bearish']) &
            (df['volatility'] > df['avg_volatility'])
        )
        
        # ── EXIT CONDITION ───────────────────────
        exit_condition = (
            (df['turnover_ratio'] < force_exit_turnover) &
            (df['Close'] < df['ema'])
        )
        
        # Get most recent signals
        buy_signals = df[buy_condition].index
        exit_signals = df[exit_condition].index
        
        signal = "STAND STILL"
        signal_time = None
        
        # Check for BUY first (most recent signal wins)
        if len(buy_signals) > 0:
            last_buy = buy_signals[-1]
            last_exit = exit_signals[-1] if len(exit_signals) > 0 else None
            
            if last_exit is None or last_buy > last_exit:
                signal = "BUY"
                signal_time = last_buy
        
        # If no BUY, check for EXIT
        if signal == "STAND STILL" and len(exit_signals) > 0:
            signal = "EXIT"
            signal_time = exit_signals[-1]
        
        latest = df.iloc[-1]
        
        return {
            "ticker": ticker,
            "price": round(latest['Close'], 4),
            "turnover": round(latest['turnover_ratio'], 2),
            "slope": round(latest['regression_slope'], 6),
            "volatility": round(latest['volatility'], 4),
            "signal": signal,
            "signal_time": str(signal_time) if signal_time else None
        }
        
    except Exception as e:
        print(f"  ❌ {ticker} error: {e}")
        return None

# ── MAIN EXECUTION ─────────────────────────────
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

# ── SAVE OUTPUT FOR GITHUB ACTIONS ─────────────
if signals_found:
    # Save first signal for Telegram (take highest turnover or first)
    primary = signals_found[0]
    
    with open('signal_output.txt', 'w') as f:
        f.write(f"Ticker: {primary['ticker']}\n")
        f.write(f"Price: {primary['price']}\n")
        f.write(f"Signal: {primary['signal']}\n")
        f.write(f"Turnover: {primary['turnover']}\n")
        f.write(f"Volatility: {primary['volatility']}\n")
        f.write(f"Time: {primary['signal_time']}\n")
    
    with open('signal_output.json', 'w') as f:
        json.dump(signals_found, f, indent=2)
    
    print("\n🚨 SIGNALS DETECTED:")
    for s in signals_found:
        print(f"  🔔 {s['ticker']}: {s['signal']} @ ${s['price']}")
else:
    with open('signal_output.txt', 'w') as f:
        f.write("NO_SIGNAL")
    print("\n😴 No active signals at this time")
