import argparse, os, mlflow, mlflow.pyfunc, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES = [
 'ret_1d','vol_3d','vol_7d','vol_14d','vol_30d','ret_3d','ret_7d','ret_14d','ret_30d',
 'ma_3d','ma_7d','ma_14d','ma_30d',
 'comp_mean','pos_mean','neg_mean','neu_mean','n_posts'
]

class TinyNet(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='data/features.parquet')
    ap.add_argument('--register_name', default='MemeCoinAlpha')
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    X = df[FEATURES].values.astype('float32')
    y = df['target_up'].values.astype('int64')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = TinyNet(in_dim=X.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train); yt = torch.tensor(y_train)
    Xv = torch.tensor(X_test); yv = torch.tensor(y_test)

    for epoch in range(200):
        model.train()
        opt.zero_grad()
        logits = model(Xt)
        loss = crit(logits, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xv)
        pred = logits.argmax(1).numpy()
        probs = torch.softmax(logits, dim=1)[:,1].numpy()
    acc = float(accuracy_score(yv, pred))
    try:
        auc = float(roc_auc_score(yv, probs))
    except Exception:
        auc = float('nan')

    mlflow.set_experiment("/Shared/CryptoMeme")
    with mlflow.start_run() as run:
        mlflow.log_metric("acc", acc)
        if not np.isnan(auc):
            mlflow.log_metric("auc", auc)

        ts = torch.jit.script(model)
        ts_path = "model.ts"
        ts.save(ts_path)

        class CryptoPyFunc(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import torch
                self.model = torch.jit.load(context.artifacts['torchscript'])
            def predict(self, context, model_input):
                import torch
                X = torch.tensor(model_input[FEATURES].values, dtype=torch.float32)
                with torch.no_grad():
                    probs = torch.softmax(self.model(X), dim=1).numpy()
                return probs

        artifacts = {"torchscript": ts_path}
        example = df[FEATURES].head(3)
        mlflow.pyfunc.log_model("model", python_model=CryptoPyFunc(), artifacts=artifacts, input_example=example)
        result = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.register_name)
        print("Registered:", result.name, result.version, "ACC:", acc, "AUC:", auc)

if __name__ == "__main__":
    main()
