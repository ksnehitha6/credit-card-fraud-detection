import pandas as pd
import os
from models import train_and_evaluate_all_models

DATA_FILE = "creditcard.csv"

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print("Dataset not found. Generating synthetic dataset...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=5000, n_features=30, n_informative=10, 
                                   n_classes=2, weights=[0.99, 0.01], random_state=42)
        data = pd.DataFrame(X)
        data['Class'] = y
        data.to_csv(DATA_FILE, index=False)
        print(f"Synthetic data saved to {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    train_and_evaluate_all_models(df)
