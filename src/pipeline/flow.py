from prefect import flow, task
import subprocess

@task
def ingest_prices():
    subprocess.check_call("python src/ingest/fetch_prices.py --coins DOGE,PEPE --days 180 --interval daily --out data/prices.parquet", shell=True)

@task
def build_sentiment():
    subprocess.check_call("python src/ingest/make_sentiment_features.py --in data/sentiment.csv --out data/sentiment_features.parquet", shell=True)

@task
def features():
    subprocess.check_call("python src/features/build_features.py --prices data/prices.parquet --sent data/sentiment_features.parquet --out data/features.parquet", shell=True)

@task
def train():
    subprocess.check_call("python src/models/train.py --features data/features.parquet --register_name MemeCoinAlpha", shell=True)

@flow
def crypto_pipeline():
    ingest_prices()
    build_sentiment()
    features()
    train()

if __name__ == "__main__":
    crypto_pipeline()
