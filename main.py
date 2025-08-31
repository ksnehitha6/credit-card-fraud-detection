import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils import print_and_save_report, ensure_dir
from models import model_rf, model_dt, model_knn, model_lof, model_kmeans, map_clusters_to_labels

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")

def load_dataset_or_synthetic(path):
    if os.path.exists(path):
        print(f"[data] found dataset at {path}")
        df = pd.read_csv(path)
        return df, False

    # synthetic fallback (small; for sanity-check only)
    print(f"[data] {path} not found. generating small synthetic sample for a dry-run...")
    rng = np.random.RandomState(42)
    n = 5000
    # synthetic PCA-like features V1..V28
    Xsyn = rng.randn(n, 28)
    time = rng.randint(0, 172800, size=n)  # two days in seconds
    amt = np.abs(rng.normal(50, 30, size=n))
    # fraud labels (rare ~0.5%)
    y = (rng.rand(n) < 0.005).astype(int)
    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(Xsyn, columns=cols)
    df["Time"] = time
    df["Amount"] = amt
    df["Class"] = y
    return df, True

def build_pipeline(base_estimator):
    numeric_to_scale = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_to_scale)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("prep", preprocessor), ("clf", base_estimator)])

def run_supervised(X, y, name, estimator):
    pipe = build_pipeline(estimator)
    pipe.fit(X["train"], y["train"])
    y_pred = pipe.predict(X["test"])
    y_proba = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X["test"])[:, 1]
    print_and_save_report(name, y["test"], y_pred, OUT_DIR, y_proba)
    return pipe

def run_lof(X, y, name="lof"):
    lof = model_lof()
    # LOF uses fit_predict on the *same* data (unsupervised). We'll fit on train, predict on test by refitting.
    lof.fit(X["train"])             # not used for prediction later (novelty=False)
    y_pred = lof.fit_predict(X["test"])
    y_pred = np.where(y_pred == -1, 1, 0)  # -1 outlier -> fraud(1)
    print_and_save_report(name, y["test"], y_pred, OUT_DIR)
    return None

def run_kmeans(X, y, name="kmeans"):
    km = model_kmeans(n_clusters=2)
    # unsupervised: fit on train, predict clusters on test, map clusters -> labels via majority of y_train
    km.fit(X["train"])
    clusters_test = km.predict(X["test"])
    clusters_train = km.predict(X["train"])
    y_pred = map_clusters_to_labels(y["train"].values, clusters_test)
    print_and_save_report(name, y["test"], y_pred, OUT_DIR)
    return km

def main(models_csv):
    df, synthetic = load_dataset_or_synthetic(DATA_PATH)

    required_cols = {"Class", "Amount", "Time"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    target = "Class"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    order = ["Time", "Amount"] + [c for c in X.columns if c not in ("Time", "Amount")]
    X_train = X_train[order]
    X_test  = X_test[order]

    Xsplit = {"train": X_train, "test": X_test}
    ysplit = {"train": y_train, "test": y_test}

    ensure_dir(OUT_DIR)

    selected = [m.strip().lower() for m in models_csv.split(",") if m.strip()]
    valid = {"rf", "dt", "knn", "lof", "kmeans"}
    if not selected:
        selected = ["rf", "dt", "knn", "lof", "kmeans"]
    for m in selected:
        if m not in valid:
            raise ValueError(f"Unknown model '{m}'. Choose from: {sorted(valid)}")

    if "rf" in selected:
        run_supervised(Xsplit, ysplit, name="random_forest", estimator=model_rf())
    if "dt" in selected:
        run_supervised(Xsplit, ysplit, name="decision_tree", estimator=model_dt())
    if "knn" in selected:
        run_supervised(Xsplit, ysplit, name="knn", estimator=model_knn())

    if "lof" in selected:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False)
        X_train_u = scaler.fit_transform(X_train)
        X_test_u  = scaler.transform(X_test)
        X_u = {"train": X_train_u, "test": X_test_u}
        run_lof(X_u, ysplit, name="lof")

    if "kmeans" in selected:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False)
        X_train_u = scaler.fit_transform(X_train)
        X_test_u  = scaler.transform(X_test)
        X_u = {"train": X_train_u, "test": X_test_u}
        run_kmeans(X_u, ysplit, name="kmeans")

    if synthetic:
        print("\\n[notice] You ran on a synthetic sample. Put the real dataset at data/creditcard.csv and run again for real results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")
    parser.add_argument("--models", default="rf,dt,knn,lof,kmeans",
                        help="comma-separated list from {rf,dt,knn,lof,kmeans}")
    args = parser.parse_args()
    main(args.models)
