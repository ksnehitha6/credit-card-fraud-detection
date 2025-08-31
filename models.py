import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import KMeans

# ---------------- Supervised ---------------- #

def model_rf(random_state=42, n_estimators=200, n_jobs=-1, class_weight="balanced"):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight
    )

def model_dt(random_state=42, class_weight="balanced"):
    return DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)

def model_knn(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

# ---------------- Unsupervised / Outlier ---------------- #

def model_lof(n_neighbors=20, contamination="auto"):
    # Note: LOF is unsupervised; returns -1 for outliers, 1 for inliers
    return LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)

def model_kmeans(n_clusters=2, random_state=42):
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

def map_clusters_to_labels(y_true, clusters):
    \"\"\"Map each cluster id -> majority label from y_true for those samples.\"\"\"
    mapping = {}
    for k in np.unique(clusters):
        majority = Counter(y_true[clusters == k]).most_common(1)[0][0]
        mapping[k] = majority
    return np.array([mapping[c] for c in clusters])
