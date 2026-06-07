import pandas as pd
import numpy as np
import yfinance as yf

# =======================================================================
# 1. C-SPEED CONVOLUTIONAL SLOPE CALCULATOR
# =======================================================================
def calculate_fast_slope(series, window):
    y = np.asarray(series.values, dtype=np.float64)
    if len(y) < window:
        return np.nan
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_weights = x - x_mean
    x_var = (x_weights ** 2).sum()
    
    slope_filter = x_weights[::-1] / x_var
    raw_slopes = np.convolve(y, slope_filter, mode='valid')
    return raw_slopes[-1]

# =======================================================================
# 2. DEFINITIONS & TARGET POOL
# =======================================================================
# Pure yfinance pool—no more finicky exchange API keys needed!
token_pool = [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'AAVE-USD',
    'JTO-USD',
    'LINK-USD',
    'XRP-USD',
    'PENDLE-USD',
    'TON11419-USD',
    'HYPE32196-USD',
    'FET-USD',
    'SUI20947-USD',
    'ONDO-USD',
    'ENA-USD',
]

dashboard_ledger = []
print("🤖 Compiling clean daily quantitative market regime matrix...")

# =======================================================================
# 3. SCANNER PROCESSING PIPELINE
# =======================================================================
for yf_ticker in token_pool:
    try:
        # --- A. DATA ACQUISITION & CLEANING ---
        df = yf.download(yf_ticker, interval="1d", period="120d", progress=False)
        df.columns = [col[0] for col in df.columns]
        df = df.dropna()
        
        if len(df) < 60:
            continue
            
        df['log_close'] = np.log(df['Close'])
        df['returns'] = df['Close'].pct_change()
        
        # --- B. STRUCTURAL TREND REGIMES (60d Macro vs 3d Micro) ---
        slope_2m = calculate_fast_slope(df['log_close'], 60)
        slope_3d = calculate_fast_slope(df['log_close'], 3)
        
        macro_regime = "🟢 BULL" if slope_2m > 0 else "🔴 BEAR"
        micro_momentum = "🟢 UP" if slope_3d > 0 else "🔴 DOWN"
        
        # --- C. 24H PRICE CHANGE TRACKER ---
        current_close = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2]
        price_change_24h = ((current_close - previous_close) / previous_close) * 100
        
        # --- D. LIQUIDITY TURNOVER RATIOS (Current vs 1-Week Average) ---
        current_vol = df['Volume'].iloc[-1]
        avg_weekly_vol = df['Volume'].iloc[-7:].mean()
        volume_turnover = current_vol / avg_weekly_vol if avg_weekly_vol > 0 else 1.0
        
        current_volatility = df['returns'].iloc[-3:].std()
        avg_weekly_volatility = df['returns'].iloc[-7:].std()
        volatility_turnover = current_volatility / avg_weekly_volatility if avg_weekly_volatility > 0 else 1.0

        # --- E. COMPILE ENTRY LINE INTO LEDGER ---
        dashboard_ledger.append({
            'TOKEN': yf_ticker.replace('-USD', ''),
            '24H PRICE': f"{price_change_24h:+.2f}%",
            'MACRO REGIME': macro_regime,
            'SHORT MOMENTUM': micro_momentum,
            'VOL TURNOVER': f"{volume_turnover:.2f}x",
            'VOLATILITY TO': f"{volatility_turnover:.2f}x"
        })
        
    except Exception as e:
        print(f"⚠️ Could not sync metrics for {yf_ticker}: {str(e)}")

# =======================================================================
# 4. EXPORT & DISPLAY THE AUTOMATED LEDGER
# =======================================================================
final_df = pd.DataFrame(dashboard_ledger)

# Display to the local console / GitHub Actions terminal logs
print("\n" + "="*95)
print("                          DAILY QUANT REGIME & LIQUIDITY MATRIX               ")
print("="*95)
print(final_df.to_string(index=False))
print("="*95)

# Auto-write to your GitHub markdown home page documentation
with open("MARKET_REPORT.md", "w", encoding="utf-8") as f:
    f.write("# 🤖 Daily Quantitative Market Regime & Liquidity Matrix\n\n")
    f.write(f"**Last Data Audit Verification (UTC):** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("> **Next-Step Playbook:** Identify tokens with high `VOL TURNOVER` and cross-examine them on Coinglass/Velo to check the live CVD order-flow delta context.\n\n")
    f.write(final_df.to_markdown(index=False))
