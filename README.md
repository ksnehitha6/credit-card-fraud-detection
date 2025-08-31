# Credit Card Fraud Detection (ML)

End-to-end implementation of multiple algorithms (Random Forest, Decision Tree, KNN, LOF, K-Means) for detecting fraudulent credit-card transactions.
Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, and saves a confusion matrix image per model.

## Project Structure
```
credit-card-fraud-detection/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ creditcard.csv           # place dataset here (not committed)
├─ outputs/                    # results & plots saved here at runtime
└─ src/
   ├─ main.py                  # entry point
   ├─ models.py                # model definitions & training wrappers
   └─ utils.py                 # helpers (metrics, io, plotting)
```

## Quickstart
```bash
# (optional) create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
python src/main.py
```
If `data/creditcard.csv` isn’t found, the script generates a small synthetic dataset so you can test the pipeline end-to-end. Put the real dataset for real results.

## Dataset
This project expects the well-known **credit card fraud** dataset (`creditcard.csv`) originally released by ULB (Kaggle).
Download it and place the file at:
```
data/creditcard.csv
```
Columns typically include `Time`, `V1..V28`, `Amount`, and target label `Class` (0 = normal, 1 = fraud).

## Run specific models
```bash
python src/main.py --models rf,dt,knn,lof,kmeans
# or just one model:
python src/main.py --models rf
```

## Outputs
- Metrics saved to `outputs/metrics_<model>.json`
- Confusion matrix images saved to `outputs/confusion_matrix_<model>.png`

## Notes
- `data/` and `outputs/` are ignored by Git via `.gitignore` to keep the repo light.
- Random Forest often performs best on this dataset when tuned.
