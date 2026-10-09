import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def sanity_check():
    print("Loading datasets...")
    gmsi = pd.read_csv(PROCESSED / 'gmsi_exogenous.csv')
    gmsi['date'] = pd.to_datetime(gmsi['date'])
    
    btc = pd.read_csv(PROCESSED / 'btc_vsi_full.csv')
    btc['date'] = pd.to_datetime(btc['date'])
    
    nifty = pd.read_csv(PROCESSED / 'nifty_vsi_full.csv')
    nifty['date'] = pd.to_datetime(nifty['date'])
    
    # Merge for correlation checking
    df_btc = pd.merge(gmsi[['date', 'pure_gmsi']], btc[['date', 'volatility_30d']], on='date', how='inner')
    df_nifty = pd.merge(gmsi[['date', 'pure_gmsi']], nifty[['date', 'volatility_30d']], on='date', how='inner')
    
    print("\n--- SAME-DAY CORRELATION (Should be low!) ---")
    corr_btc = df_btc['pure_gmsi'].corr(df_btc['volatility_30d'])
    corr_nifty = df_nifty['pure_gmsi'].corr(df_nifty['volatility_30d'])
    print(f"GMSI vs BTC Volatility (30d): {corr_btc:.3f}")
    print(f"GMSI vs NIFTY Volatility (30d): {corr_nifty:.3f}")
    
    if abs(corr_btc) > 0.5 or abs(corr_nifty) > 0.5:
        print("WARNING: High same-day correlation detected! GMSI might not be purely exogenous.")
    else:
        print("PASS: Same-day correlation is reasonably low.")
        
    print("\n--- STATIONARITY TESTS ---")
    # ADF test
    result = adfuller(gmsi['pure_gmsi'].dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4e}")
    if result[1] < 0.05:
        print("PASS: GMSI is stationary (p < 0.05).")
    else:
        print("WARNING: GMSI is non-stationary.")
        
    # Plotting
    print("\nGenerating plots...")
    os.makedirs(ROOT / 'reports' / 'figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribution
    sns.histplot(gmsi['pure_gmsi'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Pure Exogenous GMSI')
    
    # 2. Time Series
    axes[0, 1].plot(gmsi['date'], gmsi['pure_gmsi'], alpha=0.7)
    axes[0, 1].set_title('Pure GMSI Over Time')
    
    # 3. Autocorrelation
    plot_acf(gmsi['pure_gmsi'].dropna(), ax=axes[1, 0], lags=30)
    axes[1, 0].set_title('Autocorrelation (ACF) up to 30 Lags')
    
    # 4. GMSI vs Volatility Overlay (BTC)
    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(df_btc['date'], df_btc['pure_gmsi'], color='blue', alpha=0.5, label='GMSI')
    ax2.plot(df_btc['date'], df_btc['volatility_30d'], color='red', alpha=0.5, label='BTC Volatility')
    axes[1, 1].set_ylabel('GMSI', color='blue')
    ax2.set_ylabel('Volatility', color='red')
    axes[1, 1].set_title('GMSI vs BTC Volatility (Time Series)')
    
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / 'gmsi_sanity_checks.png')
    print("Saved plots to reports/figures/gmsi_sanity_checks.png")

if __name__ == '__main__':
    sanity_check()
