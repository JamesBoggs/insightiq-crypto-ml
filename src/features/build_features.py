import argparse, pandas as pd

def add_price_features(df):
    df = df.sort_values(['coin','date'])
    df['ret_1d'] = df.groupby('coin')['price_usd'].pct_change()
    for w in [3,7,14,30]:
        df[f'vol_{w}d'] = df.groupby('coin')['ret_1d'].rolling(w).std().reset_index(level=0, drop=True)
        df[f'ret_{w}d'] = df.groupby('coin')['price_usd'].pct_change(w)
        df[f'ma_{w}d']  = df.groupby('coin')['price_usd'].rolling(w).mean().reset_index(level=0, drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prices', default='data/prices.parquet')
    ap.add_argument('--sent', default='data/sentiment_features.parquet')
    ap.add_argument('--out', default='data/features.parquet')
    args = ap.parse_args()

    p = pd.read_parquet(args.prices)
    s = pd.read_parquet(args.sent)

    df = p.merge(s, on='date', how='left')
    df = add_price_features(df)

    df[['comp_mean','pos_mean','neg_mean','neu_mean']] = df[['comp_mean','pos_mean','neg_mean','neu_mean']].fillna(method='ffill')
    df['n_posts'] = df['n_posts'].fillna(0)

    df = df.sort_values(['coin','date'])
    df['target_up'] = (df.groupby('coin')['price_usd'].shift(-1) > df['price_usd']).astype(int)

    df = df.dropna().reset_index(drop=True)
    df.to_parquet(args.out)
    print(f"Wrote features to {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
