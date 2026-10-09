import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def create_vsi_dataset(price_path, text_path, events_path, asset_name, output_path):
    print(f"\n{'='*50}")
    print(f"Processing {asset_name} VSI Dataset")
    print(f"{'='*50}")
    
    # Task 1 - Load Base Price Data
    print("\n[Task 1] Loading Base Price Data")
    df = pd.read_csv(price_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['date'])
    
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Row count: {len(df)}")
    
    # Task 2 - Compute Returns & Volatility
    print("\n[Task 2] Computing Returns & Volatility")
    # Log return formula: ln(P_t / P_t-1)
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility (annualized or just raw std, we'll use raw std of log returns for daily)
    df['volatility_7d'] = df['return'].rolling(window=7, min_periods=7).std()
    df['volatility_14d'] = df['return'].rolling(window=14, min_periods=14).std()
    df['volatility_30d'] = df['return'].rolling(window=30, min_periods=30).std()
    
    print(f"  NaNs in volatility_7d: {df['volatility_7d'].isna().sum()}")
    print(f"  NaNs in volatility_14d: {df['volatility_14d'].isna().sum()}")
    print(f"  NaNs in volatility_30d: {df['volatility_30d'].isna().sum()}")
    
    # Task 3 - Attach Sentiment Data
    print("\n[Task 3] Attaching Sentiment Data")
    rows_before = len(df)
    
    # Process text sentiment
    text_df = pd.read_csv(text_path)
    if 'timestamp' in text_df.columns:
        text_df['date'] = pd.to_datetime(text_df['timestamp']).dt.normalize()
    elif 'Date' in text_df.columns:
        text_df['date'] = pd.to_datetime(text_df['Date']).dt.normalize()
        
    # Standardize asset names for filtering
    text_asset = text_df[text_df['asset'].str.contains(asset_name, case=False, na=False)]
    
    if len(text_asset) > 0:
        # We will define sentiment_core as the daily average of finbert_score (or vader if missing)
        if 'finbert_score' in text_asset.columns:
            daily_sentiment = text_asset.groupby('date')['finbert_score'].mean().reset_index()
            daily_sentiment.rename(columns={'finbert_score': 'sentiment_core'}, inplace=True)
        else:
            daily_sentiment = text_asset.groupby('date')['vader_score'].mean().reset_index()
            daily_sentiment.rename(columns={'vader_score': 'sentiment_core'}, inplace=True)
    else:
        # Fallback if no sentiment for asset (create empty)
        daily_sentiment = pd.DataFrame(columns=['date', 'sentiment_core'])
        
    # Process daily events
    events_df = pd.read_csv(events_path)
    events_df['date'] = pd.to_datetime(events_df['date']).dt.tz_localize(None)
    events_asset = events_df[events_df['asset'].str.contains(asset_name, case=False, na=False)]
    if len(events_asset) > 0:
        daily_attention = events_asset.groupby('date')['event_intensity'].mean().reset_index()
        daily_attention.rename(columns={'event_intensity': 'attention'}, inplace=True)
    else:
        daily_attention = pd.DataFrame(columns=['date', 'attention'])
        
    # Remove timezone from main df too
    df['date'] = df['date'].dt.tz_localize(None)
    daily_sentiment['date'] = daily_sentiment['date'].dt.tz_localize(None)
        
    # Merge
    df = pd.merge(df, daily_sentiment, on='date', how='left')
    df = pd.merge(df, daily_attention, on='date', how='left')
    
    # Calculate sentiment_surprise:
    # surprise = actual today - moving average (expected). Let's use a 30-day expanding or rolling mean
    # We will compute a 30d rolling expanding mean on non-nulls.
    # To compute rolling without forward leakage, we just take rolling mean of past 30 days
    if 'sentiment_core' in df.columns:
        df['sentiment_surprise'] = df['sentiment_core'] - df['sentiment_core'].rolling(30, min_periods=1).mean()
    else:
        df['sentiment_surprise'] = np.nan
        df['sentiment_core'] = np.nan
        
    rows_after = len(df)
    print(f"  Row count before merge: {rows_before}")
    print(f"  Row count after merge: {rows_after}")
    
    # Task 4 - Feature Normalization (Z-score expanding to avoid leakage)
    print("\n[Task 4] Feature Normalization")
    def expanding_z_score(series, min_periods=30):
        mean = series.expanding(min_periods=min_periods).mean()
        std = series.expanding(min_periods=min_periods).std()
        # Avoid division by zero
        return (series - mean) / std.replace(0, np.nan)
        
    # Compute z-score for volatility_30d which is usually the most stable vol
    df['vol_z'] = expanding_z_score(df['volatility_30d'], min_periods=30)
    
    # Compute z-score for sentiment (we only compute on non-null values, then reindex)
    if df['sentiment_core'].notna().sum() > 5:
        df['sentiment_z'] = expanding_z_score(df['sentiment_core'], min_periods=5)
    else:
        df['sentiment_z'] = np.nan
        
    attention_exists = 'attention' in df.columns and df['attention'].notna().sum() > 5
    if attention_exists:
        df['attention_z'] = expanding_z_score(df['attention'], min_periods=5)
    else:
        df['attention_z'] = np.nan
        
    # Task 5 - Construct VSI
    print("\n[Task 5] Constructing VSI")
    # Formula: VSI = w1 * vol_z + w2 * sentiment_z + w3 * attention_z
    # Weights = equal where available. Z-scores already handle scale.
    # To properly compute without losing rows, we treat NaNs as 0 (neutral) for sentiment/attention,
    # or just use available signals. A transparent approach:
    # VSI = mean of available z-scores for that day.
    
    # Let's fill NaNs in z-scores with 0 (neutral expected value) for the VSI calculation
    df_scores = df[['vol_z', 'sentiment_z', 'attention_z']].fillna(0)
    # Give them equal weights assuming all 3 exist. If attention doesn't exist, we just add 0 which is fine if weights are equal, but let's take mean of non-NaN components.
    
    # "Default weights = equal unless stated. Clearly comment the formula."
    # We will use explicit weights.
    # We use a 1/3, 1/3, 1/3 split if all present, or 1/2, 1/2 if attention missing.
    if attention_exists:
        # VSI = (1/3)*vol_z + (1/3)*sentiment_z + (1/3)*attention_z
        df['VSI'] = (1/3) * df_scores['vol_z'] + (1/3) * df_scores['sentiment_z'] + (1/3) * df_scores['attention_z']
    else:
        # VSI = 0.5 * vol_z + 0.5 * sentiment_z
        df['VSI'] = 0.5 * df_scores['vol_z'] + 0.5 * df_scores['sentiment_z']
        
    print("  Calculated VSI using equal weights for available z-score features.")
    
    # Task 6 - Final Dataset Integrity Check
    print("\n[Task 6] Final Dataset Integrity Check")
    # "Drop rows only if VSI cannot be computed"
    # Actually, if vol_z is NaN (e.g. first 30 days), VSI might be 0 because we filled with 0. 
    # But truly, if 'volatility_30d' is NaN, we probably shouldn't have a VSI.
    # Let's clean up: If vol_z is truly NaN because of insufficient history, VSI should be NaN.
    # Let's re-apply NaN to VSI where vol_z is NaN.
    df.loc[df['vol_z'].isna(), 'VSI'] = np.nan
    
    final_df = df.dropna(subset=['VSI']).copy()
    final_count = len(final_df)
    original_count = len(df)
    retained_pct = (final_count / original_count) * 100
    
    print(f"  Final row count: {final_count}")
    print(f"  % of original data retained: {retained_pct:.2f}%")
    
    # Reorder columns as requested
    required_cols = [
        'date', 'close', 'volume', 'return',
        'volatility_7d', 'volatility_14d', 'volatility_30d',
        'sentiment_core', 'sentiment_surprise',
        'vol_z', 'sentiment_z'
    ]
    if attention_exists:
        required_cols.append('attention_z')
    required_cols.append('VSI')
    
    # ensure all columns exist
    for col in required_cols:
        if col not in final_df.columns:
            print(f"  WARNING: missing required column {col}")
            
    final_cols = [c for c in required_cols if c in final_df.columns] + [c for c in final_df.columns if c not in required_cols]
    final_df = final_df[final_cols]
    
    final_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    # BTC
    create_vsi_dataset(
        price_path=ROOT / 'data/raw/btc_prices.csv',
        text_path=PROCESSED / 'text_with_sentiment.csv',
        events_path=PROCESSED / 'daily_events.csv',
        asset_name='BTC',
        output_path=PROCESSED / 'btc_vsi_full.csv'
    )
    
    # NIFTY
    create_vsi_dataset(
        price_path=ROOT / 'data/raw/nifty_prices.csv',
        text_path=PROCESSED / 'text_with_sentiment.csv',
        events_path=PROCESSED / 'daily_events.csv',
        asset_name='NIFTY',
        output_path=PROCESSED / 'nifty_vsi_full.csv'
    )
    
    print("\nDone!")
