# InsightIQ Crypto ML (Meme Coin Forecasting)

End-to-end pipeline you can run locally and show in your portfolio:
- 📈 Prices from **CoinGecko**
- 🗣️ Sentiment from your CSV (Reddit/Twitter/Discord posts → VADER)
- ⚙️ Feature engineering (returns, vol, moving averages, sentiment)
- 🧠 PyTorch model with **MLflow** tracking/registry
- 🚀 Serve predictions via **FastAPI**
- 📓 A clean **Jupyter notebook** to demo the whole thing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOURNAME/insightiq-crypto-ml/blob/main/notebooks/demo_crypto_pipeline.ipynb)

## Quickstart (macOS)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
jupyter lab
```

## CLI (no notebook)
```bash
python src/ingest/fetch_prices.py --coins DOGE,PEPE --days 180 --interval daily --out data/prices.parquet
python src/ingest/make_sentiment_features.py --in data/sentiment.csv --out data/sentiment_features.parquet
python src/features/build_features.py --prices data/prices.parquet --sent data/sentiment_features.parquet --out data/features.parquet
python src/models/train.py --features data/features.parquet --register_name MemeCoinAlpha
```

## Serve the model (FastAPI)
```bash
export MODEL_NAME=MemeCoinAlpha MODEL_VERSION=1
uvicorn src.api.app:app --port 8000 --reload
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"   -d '{"features":[[0.01,0.02,0.03,0.04,0.05,-0.01,-0.02,-0.03,-0.04,0.1,0.1,0.1,0.1,0.2,0.1,0.05,0.65,12]]}'
```

Customize coins, sentiment, features, and model as you like.
MIT © YOUR NAME
