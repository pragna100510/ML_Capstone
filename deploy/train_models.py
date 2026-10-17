"""Train and save every model the SmartCommerce AI app can use.

It repeats the setup from your notebooks (same cleaning, same split, same
hyperparameters) and saves one file per algorithm in ./models:

    models/purchase_<algorithm>.pkl    (classification_finall.ipynb)
    models/diamond_<algorithm>.pkl     (regression.ipynb)

Each file is a full scikit-learn Pipeline (preprocessing + model), so the app
just passes it a one-row DataFrame.

Usage (run from the folder that contains your data files):

    python train_models.py                         # everything
    python train_models.py --only classification   # purchase models only
    python train_models.py --only regression       # diamond models only
    python train_models.py --shoppers path/online_shoppers_intention.csv \
                           --diamonds path/diamonds.csv

Run it with the same scikit-learn version you use to run the app, otherwise
the pickles may fail to load.

Clustering is not trained here because it needs the Online Retail II workbook.
Add the cell in clustering_export_cell.py to the end of clustering.ipynb.
"""
import argparse
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42
MODEL_DIR = Path("models")


def slug(name):
    """'Random Forest' -> 'random_forest'. app.py uses the same rule."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save(prefix, name, model):
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{prefix}_{slug(name)}.pkl"
    joblib.dump(model, path, compress=3)
    return path


# ==================================================
# CLASSIFICATION  (classification_finall.ipynb)
# ==================================================
def classification_preprocessor(numerical, categorical, dense=False):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=not dense
            )),
        ]), categorical),
    ])


def train_classification(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["TotalPages"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)
    categorical = ["Month", "VisitorType", "Weekend"]
    numerical = [c for c in X.columns if c not in categorical]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # name -> (classifier, needs dense one-hot output)
    classifiers = {
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=5, random_state=42), False),
        "Random Forest": (RandomForestClassifier(
            n_estimators=50, max_depth=15, random_state=42, n_jobs=1,
            class_weight="balanced"), False),
        "AdaBoost": (AdaBoostClassifier(
            n_estimators=30, learning_rate=0.5, random_state=42), False),
        "Gradient Boosting": (GradientBoostingClassifier(
            n_estimators=30, learning_rate=0.1, max_depth=2,
            random_state=42), False),
        "Bagging": (BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
            n_estimators=10, random_state=42, n_jobs=1), False),
        "Support Vector Machine": (
            SVC(C=1.0, kernel="rbf", random_state=42), False),
        "MLP Neural Network": (MLPClassifier(
            hidden_layer_sizes=(30,), activation="relu", max_iter=50,
            early_stopping=True, random_state=42), False),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42), False),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), False),
        "Gaussian Naive Bayes": (GaussianNB(), True),
    }

    print("\nPURCHASE INTENTION (classification)")
    print(f"{'Algorithm':<26}{'Accuracy':>10}{'Weighted F1':>13}{'Seconds':>9}")
    for name, (clf, dense) in classifiers.items():
        start = time.time()
        model = Pipeline([
            ("preprocessor",
             classification_preprocessor(numerical, categorical, dense)),
            ("classifier", clf),
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        save("purchase", name, model)
        print(f"{name:<26}{accuracy_score(y_test, pred):>10.4f}"
              f"{f1_score(y_test, pred, average='weighted'):>13.4f}"
              f"{time.time() - start:>9.1f}")


# ==================================================
# REGRESSION  (regression.ipynb)
# ==================================================
def train_regression(csv_path):
    # The notebook reads the file with sep="\t"; accept comma files too.
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    invalid = (df[["x", "y", "z"]] <= 0).any(axis=1)
    df = df.loc[~invalid].copy()
    df = df.loc[~((df["y"] > 20) | (df["z"] > 10))].copy()
    df["volume"] = df["x"] * df["y"] * df["z"]

    X = df.drop(columns=["price"])
    y = df["price"]
    categorical = ["cut", "color", "clarity"]
    numerical = [c for c in X.columns if c not in categorical]

    price_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=price_bins
    )

    def preprocessor():
        return ColumnTransformer([
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            ), categorical),
        ])

    # Best hyperparameters found by GridSearchCV in the notebook.
    regressors = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1),
        "Polynomial Regression (Degree 2)": None,  # built below
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.2, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Support Vector Regressor": SVR(C=100, kernel="rbf", epsilon=0.1),
        "ElasticNet Regression": ElasticNet(
            alpha=0.001, l1_ratio=0.8, max_iter=10000),
        "Ridge Regression": Ridge(alpha=0.1),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Linear Regression": LinearRegression(),
    }

    print("\nDIAMOND PRICE (regression)")
    print(f"{'Algorithm':<34}{'R2':>8}{'RMSE':>10}{'MAE':>9}{'Seconds':>9}")
    for name, reg in regressors.items():
        start = time.time()
        if reg is None:
            steps = [
                ("preprocessor", preprocessor()),
                ("polynomial", PolynomialFeatures(degree=2,
                                                  include_bias=False)),
                ("model", LinearRegression()),
            ]
        else:
            steps = [("preprocessor", preprocessor()), ("model", reg)]
        model = Pipeline(steps)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        path = save("diamond", name, model)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"{name:<34}{r2_score(y_test, pred):>8.4f}{rmse:>10.1f}"
              f"{mean_absolute_error(y_test, pred):>9.1f}"
              f"{time.time() - start:>9.1f}   ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shoppers", default="online_shoppers_intention.csv")
    parser.add_argument("--diamonds", default="diamonds.csv")
    parser.add_argument("--only", choices=["classification", "regression"])
    args = parser.parse_args()

    if args.only in (None, "classification"):
        train_classification(args.shoppers)
    if args.only in (None, "regression"):
        train_regression(args.diamonds)
    print("\nDone. Models are in the 'models' folder.")