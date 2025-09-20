import argparse, pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inpath', default='data/sentiment.csv')
    ap.add_argument('--out', dest='outpath', default='data/sentiment_features.parquet')
    args = ap.parse_args()

    df = pd.read_csv(args.inpath)
    df['date'] = pd.to_datetime(df['date']).dt.date
    sia = SentimentIntensityAnalyzer()
    scores = df['text'].fillna('').apply(sia.polarity_scores).apply(pd.Series)
    df = pd.concat([df[['date','source']].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    daily = df.groupby(['date']).agg(
        comp_mean=('compound','mean'),
        pos_mean=('pos','mean'),
        neg_mean=('neg','mean'),
        neu_mean=('neu','mean'),
        n_posts=('compound','size')
    ).reset_index()

    daily.to_parquet(args.out)
    print(f"Saved sentiment features to {args.out}")

if __name__ == "__main__":
    main()
