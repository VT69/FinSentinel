"""
scripts/reconstruct_gmsi.py — the exogenous Global Market Stress Index (GMSI).

GMSI_t = mean of six expanding z-scores (equal weights, NO PCA): event count,
negative-event share, −Goldstein, −sentiment, −sentiment surprise, attention.
Missing components are set to 0 ("neutral") before averaging. Sentiment exists
only for 2024-10 → 2025-01, so on most days three of the six terms are 0 and the
index is effectively a 3-component index scaled by 1/2 (docs/ISSUES.md P1-7).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def expanding_z_score(series, min_periods=30):
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)

def reconstruct_gmsi():
    print("Reconstructing Pure Exogenous GMSI...")
    
    # 1. Global Daily Events (Counts & Metrics)
    events_daily = pd.read_csv(PROCESSED / 'events_daily_2015_2025.csv')
    events_daily['date'] = pd.to_datetime(events_daily['date']).dt.normalize()
    
    # 2. Text Sentiment (Average globally per day)
    text_df = pd.read_csv(PROCESSED / 'text_with_sentiment.csv')
    text_df['date'] = pd.to_datetime(text_df['timestamp']).dt.tz_localize(None).dt.normalize()
    # Average finbert and vader
    daily_sentiment = text_df.groupby('date')[['finbert_score', 'vader_score']].mean().reset_index()
    # Use finbert strictly representing sentiment core since VADER can be fallback
    daily_sentiment['sentiment_core'] = daily_sentiment['finbert_score'].fillna(daily_sentiment['vader_score'])
    
    # 3. Attention proxy based on event intensity (btc and nifty)
    daily_events = pd.read_csv(PROCESSED / 'daily_events.csv')
    daily_events['date'] = pd.to_datetime(daily_events['date']).dt.normalize()
    global_attention = daily_events.groupby('date')['event_intensity'].mean().reset_index()
    global_attention.rename(columns={'event_intensity': 'attention'}, inplace=True)
    
    # Merge all into one Master GMSI DataFrame
    df = pd.merge(events_daily, daily_sentiment[['date', 'sentiment_core']], on='date', how='left')
    df = pd.merge(df, global_attention, on='date', how='left')
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate sentiment surprise (expanding/rolling mean)
    df['sentiment_surprise'] = df['sentiment_core'] - df['sentiment_core'].rolling(30, min_periods=1).mean()
    
    # Normalize features using expanding z-score to prevent data leakage
    print("Normalizing features...")
    df['event_count_z'] = expanding_z_score(df['event_count'], min_periods=30)
    df['neg_event_share_z'] = expanding_z_score(df['neg_event_share'], min_periods=30)
    
    # Goldstein: lower is more stressful, so we invert it (-z)
    df['avg_goldstein_z_inv'] = -expanding_z_score(df['avg_goldstein'], min_periods=30)
    
    # Sentiment: lower finbert score = negative sentiment = more stress, so we invert it
    if df['sentiment_core'].notna().sum() > 5:
        df['sentiment_z_inv'] = -expanding_z_score(df['sentiment_core'], min_periods=5)
        df['sentiment_surprise_z_inv'] = -expanding_z_score(df['sentiment_surprise'], min_periods=5)
    else:
        df['sentiment_z_inv'] = np.nan
        df['sentiment_surprise_z_inv'] = np.nan
        
    if df['attention'].notna().sum() > 5:
        df['attention_z'] = expanding_z_score(df['attention'], min_periods=5)
    else:
        df['attention_z'] = np.nan

    # Fill NaNs with 0 for standardized features before averaging
    z_features = ['event_count_z', 'neg_event_share_z', 'avg_goldstein_z_inv', 
                  'sentiment_z_inv', 'sentiment_surprise_z_inv', 'attention_z']
    
    # Only keep the ones fully calculated
    coverage = df[z_features].notna().mean()
    print("Component coverage (share of days with a value):\n" + coverage.round(3).to_string())
    df_scores = df[z_features].fillna(0)
    
    # Equal weight sum/average of these stress indicators
    # All features are aligned so that HIGHER value = MORE STRESS.
    print("Combining features (Equal Weight Baseline)...")
    df['pure_gmsi'] = df_scores.mean(axis=1)
    
    # We should only have a valid GMSI if standardisation ran cleanly (at least 30 days of data)
    # Set to NaN if event_count_z is NaN
    df.loc[df['event_count_z'].isna(), 'pure_gmsi'] = np.nan
    
    final_df = df.dropna(subset=['pure_gmsi']).reset_index(drop=True)
    
    # Retain strictly exogenous columns and the pure GMSI
    out_cols = [
        'date', 'event_count', 'avg_goldstein', 'neg_event_share', 
        'conflict_events', 'economic_events', 'political_events',
        'sentiment_core', 'sentiment_surprise', 'attention',
        'event_count_z', 'neg_event_share_z', 'avg_goldstein_z_inv', 
        'sentiment_z_inv', 'sentiment_surprise_z_inv', 'attention_z',
        'pure_gmsi'
    ]
    
    final_df = final_df[[c for c in out_cols if c in final_df]]
    
    output_path = PROCESSED / 'gmsi_exogenous.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total rows: {len(final_df)}")
    print(df[['date', 'pure_gmsi']].tail())
    
if __name__ == '__main__':
    reconstruct_gmsi()
