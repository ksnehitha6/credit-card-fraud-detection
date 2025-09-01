import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import KMeans
import seaborn as sns

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_confusion_matrix.png")
    plt.close()

def train_and_evaluate_all_models(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results = {}

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        save_confusion_matrix(y_test, y_pred, name)
        results[name] = report

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    y_pred_lof = lof.fit_predict(X)
    results["LOF"] = {"outlier_count": int((y_pred_lof==-1).sum())}

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    results["KMeans"] = {"inertia": kmeans.inertia_}

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Training complete..")
