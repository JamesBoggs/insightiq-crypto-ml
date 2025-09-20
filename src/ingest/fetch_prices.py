import argparse, pandas as pd, time
from pycoingecko import CoinGeckoAPI

def fetch_coin(coin_id: str, days: int, interval: str):
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days, interval=interval)
    df = pd.DataFrame(data['prices'], columns=['ts_ms','price_usd'])
    df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.date
    df = df.groupby('date', as_index=False)['price_usd'].mean()
    df['coin'] = coin_id.upper()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coins', default='DOGE,PEPE', help='Comma-separated CoinGecko ids')
    ap.add_argument('--days', type=int, default=180)
    ap.add_argument('--interval', default='daily')
    ap.add_argument('--out', default='data/prices.parquet')
    args = ap.parse_args()

    coins = [c.strip() for c in args.coins.split(',') if c.strip()]
    frames = []
    for c in coins:
        df = fetch_coin(c.lower(), args.days, args.interval)
        frames.append(df)
        time.sleep(1.2)
    allp = pd.concat(frames, ignore_index=True)
    allp.to_parquet(args.out)
    print(f"Saved {len(allp)} rows to {args.out}")

if __name__ == "__main__":
    main()
