from prefect import flow, task
import subprocess

@task
def ingest_prices():
    subprocess.check_call(
        "python src/ingest/fetch_prices.py --coins DOGE,PEPE --days 180 --interval daily --out data/prices.parquet",
        shell=True
    )

@task
def build_sentiment():
    subprocess.check_call(
        "python src/ingest/make_sentiment_features.py --in data/sentiment.csv --out data/sentiment_features.parquet",
        shell=True
    )

@task
def features():
    subprocess.check_call(
        "python src/features/build_features.py --prices data/prices.parquet --sent data/sentiment_features.parquet --out data/features.parquet",
        shell=True
    )

@task
def train():
    subprocess.check_call(
        "python src/models/train.py --features data/features.parquet --register_name MemeCoinAlpha",
        shell=True
    )

@task
def save_predictions():
    import pandas as pd, mlflow, os

    # connect to MLflow (defaults to local server if set)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001'))
    client = mlflow.tracking.MlflowClient()

    # find latest registered version
    vers = client.search_model_versions("name='MemeCoinAlpha'")
    version = max(int(v.version) for v in vers) if vers else 1

    # load model
    model = mlflow.pyfunc.load_model(f"models:/MemeCoinAlpha/{version}")

    # load features
    df = pd.read_parquet("data/features.parquet")
    X = df[['ret_1d','vol_3d','vol_7d','vol_14d','vol_30d',
            'ret_3d','ret_7d','ret_14d','ret_30d',
            'ma_3d','ma_7d','ma_14d','ma_30d',
            'comp_mean','pos_mean','neg_mean','neu_mean','n_posts']]

    # predict
    probs = model.predict(X).tolist()

    # handle both numpy arrays and lists of lists
    df['prob_up'] = [p[1] if isinstance(p, (list, tuple)) else p for p in probs]

    # save last 50 rows
    out = df[['date','coin','prob_up']].tail(50)
    out.to_csv("predictions.csv", index=False)

    print("✅ Wrote predictions.csv with shape", out.shape)

@flow
def crypto_pipeline():
    ingest_prices()
    build_sentiment()
    features()
    train()
    save_predictions()

if __name__ == "__main__":
    crypto_pipeline()
