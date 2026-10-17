import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SmartCommerce AI",
    page_icon="🛍️",
    layout="wide"
)
# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0f1115;
    --card: #181b22;
    --field: #11141a;
    --line: #2a2f3a;
    --text: #e8eaed;
    --muted: #a8b0be;
    --accent: #ff6b3d;
    --accent-dark: #e8532b;
    --field-height: 60px;
}

/* Scale everything up: Streamlit sizes most things in rem */
html {
    font-size: 19px;
}
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
.stApp { background-color: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2.5rem 3rem 4rem 3rem;
}
@media (max-width: 768px) {
    html { font-size: 17px; }
    .block-container { padding: 1.5rem 1rem 3rem 1rem; }
}

/* ---- heading ---- */
.brand {
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 10px;
}
.brand span { color: var(--accent); }
.tagline {
    color: var(--muted);
    font-size: 1.3rem;
    line-height: 1.5;
    margin-bottom: 36px;
}

/* ---- tabs: big, equal-width boxes ----
   Selectors cover several Streamlit versions (BaseWeb tabs, role="tab",
   and data-testid="stTab"), and !important beats Streamlit's own styles. */
[data-testid="stTabs"] [role="tablist"],
[data-baseweb="tab-list"],
[role="tablist"] {
    gap: 16px !important;
    border-bottom: none !important;
    box-shadow: none !important;
    padding-bottom: 8px !important;
}
[data-testid="stTab"],
[role="tab"],
[data-baseweb="tab"] {
    flex: 1 1 0 !important;           /* three equal boxes across the row */
    height: 64px !important;
    min-height: 64px !important;
    padding: 0 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    transition: background 0.15s, border-color 0.15s;
}
[data-testid="stTab"]:hover,
[role="tab"]:hover,
[data-baseweb="tab"]:hover {
    border-color: var(--accent) !important;
    background: #1f232d !important;
}
[data-testid="stTab"] *,
[role="tab"] *,
[data-baseweb="tab"] *,
[data-testid="stTab"],
[role="tab"] {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
    color: var(--muted) !important;
    text-align: center;
}
[data-testid="stTab"] p,
[role="tab"] p {
    margin: 0 !important;
}
[data-testid="stTab"][aria-selected="true"],
[role="tab"][aria-selected="true"],
[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="stTab"][aria-selected="true"] *,
[role="tab"][aria-selected="true"] *,
[data-baseweb="tab"][aria-selected="true"] *,
[data-testid="stTab"][aria-selected="true"],
[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
}
/* hide the default grey line and red underline */
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"],
[data-testid="stTabs"] [role="tablist"] + div[aria-hidden="true"],
[data-testid="stTabs"] [role="tablist"]::after {
    display: none !important;
}
@media (max-width: 768px) {
    [role="tablist"] { flex-wrap: wrap !important; }
    [role="tab"] { flex: 1 1 100% !important; height: 52px !important; }
}

/* ---- section headings ---- */
h2, h3 {
    font-size: 2.2rem !important;
    font-weight: 600;
    line-height: 1.25;
    padding: 1rem 0 0.4rem 0;
    color: var(--text);
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: var(--muted);
    font-size: 1.15rem;
    line-height: 1.5;
}

/* ---- form card ---- */
[data-testid="stForm"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 36px 36px 24px 36px;
}

/* ---- labels: same height everywhere so fields line up ---- */
[data-testid="stWidgetLabel"] {
    min-height: 1.8rem;
    margin-bottom: 6px;
    align-items: flex-end;
}
label p,
[data-testid="stWidgetLabel"] p {
    font-size: 1.15rem !important;
    font-weight: 600;
    line-height: 1.4;
    color: var(--text);
}

/* ---- inputs and dropdowns: one orange border, always visible ----
   Number inputs: the border sits on the outer container, so it wraps the
   whole box including the +/- buttons, selected or not. */
[data-testid="stNumberInputContainer"],
[data-testid="stNumberInputContainer"]:hover,
[data-testid="stNumberInputContainer"]:focus-within {
    background-color: var(--field) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 8px !important;
    height: var(--field-height) !important;
    min-height: var(--field-height) !important;
    align-items: center !important;
    box-shadow: none !important;
    outline: none !important;
    overflow: hidden;
}
/* everything inside the container has no border of its own */
[data-testid="stNumberInputContainer"] > div,
[data-testid="stNumberInputContainer"] div[data-baseweb="input"],
[data-testid="stNumberInputContainer"] div[data-baseweb="base-input"],
[data-testid="stNumberInputContainer"] input,
[data-testid="stNumberInputContainer"] input:focus {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    height: 100%;
    align-items: center;
}
[data-testid="stNumberInput"] input {
    font-size: 1.3rem !important;
    padding: 0 16px;
    color: var(--text);
}

/* dropdowns: the border sits on the outer wrapper of the control (same idea
   as the number inputs), so it shows in every state: idle, hover, focus, open */
[data-testid="stSelectbox"] > div:last-child {
    display: flex !important;
    align-items: center !important;
    background-color: var(--field) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 8px !important;
    height: var(--field-height) !important;
    min-height: var(--field-height) !important;
    padding: 0 6px !important;
    box-shadow: none !important;
    outline: none !important;
    overflow: hidden;
}
/* everything inside the wrapper: no border / glow / background of its own */
[data-testid="stSelectbox"] > div:last-child * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--text) !important;
    font-size: 1.3rem !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] > div:last-child > div {
    flex: 1 1 auto;
    min-height: 0 !important;
}
[data-testid="stSelectbox"] > div:last-child svg {
    width: 22px !important;
    height: 22px !important;
}

/* +/- stepper buttons */
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    background-color: #1f232d;
    color: var(--text);
    height: 100%;
    width: 44px;
    border: none !important;
    box-shadow: none !important;
}

/* dropdown menu items */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background-color: var(--card);
    color: var(--text);
    font-size: 1.2rem;
    padding-top: 10px;
    padding-bottom: 10px;
}

/* space between rows of fields */
[data-testid="stForm"] [data-testid="stVerticalBlock"] {
    gap: 1.1rem;
}

/* ---- button ---- */
div.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] > button,
div.stButton > button {
    background-color: var(--accent);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    height: 60px;
    margin-top: 12px;
    transition: background 0.15s;
}
div.stFormSubmitButton > button p,
[data-testid="stFormSubmitButton"] > button p,
div.stButton > button p {
    font-size: 1.3rem;
    font-weight: 600;
}
div.stFormSubmitButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
div.stButton > button:hover {
    background-color: var(--accent-dark);
    color: #ffffff;
    border: none;
}

/* ---- result card ---- */
.result {
    max-width: 640px;
    margin: 32px auto 0 auto;
    padding: 28px 32px;
    text-align: center;
    background: var(--card);
    border: 1px solid var(--line);
    border-top: 5px solid var(--accent);
    border-radius: 8px;
}
.result.good { border-top-color: #4ade80; }
.result.neutral { border-top-color: #8b93a3; }
.result-label {
    font-size: 1.05rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: var(--muted);
    margin-bottom: 8px;
}
.result-value {
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
}

/* hide the "Press Enter to submit form" hint inside every input */
[data-testid="InputInstructions"],
[data-testid="stNumberInput"] [data-testid="InputInstructions"],
div[data-baseweb="input"] + div[data-testid="InputInstructions"],
div[data-testid="InputInstructions"] {
    display: none !important;
    visibility: hidden !important;
}

/* alert boxes (warning / error) */
[data-testid="stAlert"] p {
    font-size: 1.15rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="brand">SmartCommerce AI<span>.</span></div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Predict, Classify and Segment '
    'for Smarter E-Commerce</div>',
    unsafe_allow_html=True
)

# Folder with the saved models (created by train_models.py)
MODEL_DIR = Path("models")


@st.cache_resource(show_spinner="Loading model...")
def _read_model(path):
    return joblib.load(path)


def load_model(filename):
    """Load a saved model once and reuse it. Returns None if it is missing."""
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return _read_model(str(path))


def model_file(prefix, algorithm):
    """('diamond', 'Random Forest') -> 'diamond_random_forest.pkl'.
    train_models.py names its files with the same rule."""
    name = re.sub(r"[^a-z0-9]+", "_", algorithm.lower()).strip("_")
    return f"{prefix}_{name}.pkl"


def algorithm_picker(options, key):
    """Dropdown to choose the algorithm, with its notebook score underneath.
    `options` maps algorithm name -> short score text (first one = default).
    It sits outside the form so the score updates as soon as you change it."""
    left, _ = st.columns([1.2, 1.8])
    with left:
        choice = st.selectbox("Algorithm", list(options), key=key)
        st.caption(options[choice])
    return choice


# Scores below are the test-set results from your notebooks, best first.
DIAMOND_ALGOS = {
    "Random Forest": "Test R² 0.976 · RMSE $620",
    "Polynomial Regression (Degree 2)": "Test R² 0.967 · RMSE $722",
    "Gradient Boosting": "Test R² 0.962 · RMSE $778",
    "K-Nearest Neighbors": "Test R² 0.957 · RMSE $825",
    "Decision Tree": "Test R² 0.950 · RMSE $895",
    "Support Vector Regressor": "Test R² 0.948 · RMSE $916",
    "ElasticNet Regression": "Test R² 0.917 · RMSE $1,151",
    "Ridge Regression": "Test R² 0.917 · RMSE $1,151",
    "Lasso Regression": "Test R² 0.917 · RMSE $1,151",
    "Linear Regression": "Test R² 0.917 · RMSE $1,151",
}

PURCHASE_ALGOS = {
    "Decision Tree": "Test accuracy 90.2% · weighted F1 0.901",
    "Random Forest": "Test accuracy 90.0% · weighted F1 0.900",
    "AdaBoost": "Test accuracy 90.0% · weighted F1 0.898",
    "Gradient Boosting": "Test accuracy 90.3% · weighted F1 0.898",
    "Bagging": "Test accuracy 89.6% · weighted F1 0.893",
    "Support Vector Machine": "Test accuracy 89.4% · weighted F1 0.886",
    "MLP Neural Network": "Test accuracy 89.4% · weighted F1 0.885",
    "Logistic Regression": "Test accuracy 88.9% · weighted F1 0.875",
    "K-Nearest Neighbors": "Test accuracy 88.2% · weighted F1 0.871",
    "Gaussian Naive Bayes": "Test accuracy 72.5% · weighted F1 0.761",
}

CLUSTER_ALGOS = {
    "K-Means (k=4)": "Silhouette 0.274 · Davies-Bouldin 1.290",
    "Agglomerative (Ward, k=4)": "Silhouette 0.244 · Davies-Bouldin 1.436",
}


def form_columns(n):
    """Columns for the form fields, top-aligned so rows line up."""
    try:
        return st.columns(n, gap="large", vertical_alignment="top")
    except TypeError:
        # older Streamlit versions
        return st.columns(n, gap="large")


def show_result(label, value, tone=""):
    st.markdown(
        f'<div class="result {tone}">'
        f'<div class="result-label">{label}</div>'
        f'<div class="result-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def centered_submit(label):
    """Submit button placed in the middle of the form."""
    _, middle, _ = st.columns([1.2, 1, 1.2])
    with middle:
        try:
            return st.form_submit_button(label, width="stretch")
        except TypeError:
            # older Streamlit versions
            return st.form_submit_button(label, use_container_width=True)


tab_diamond, tab_purchase, tab_cluster = st.tabs(
    ["Regression", "Classification", "Clustering"]
)

# ==================================================
# 1. DIAMOND PRICE - REGRESSION
# ==================================================
with tab_diamond:
    st.subheader("Diamond price estimator")
    st.caption("Enter the stone's details to get an instant price.")

    diamond_algo = algorithm_picker(DIAMOND_ALGOS, "diamond_algo")

    with st.form("diamond_form"):
        col1, col2, col3 = form_columns(3)

        with col1:
            carat = st.number_input(
                "Carat", min_value=0.1, value=0.7, step=0.1
            )
            cut = st.selectbox(
                "Cut",
                ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                index=4
            )
            color = st.selectbox(
                "Color", ["D", "E", "F", "G", "H", "I", "J"], index=3
            )

        with col2:
            clarity = st.selectbox(
                "Clarity",
                ["I1", "SI2", "SI1", "VS2", "VS1",
                 "VVS2", "VVS1", "IF"],
                index=2
            )
            depth = st.number_input(
                "Depth (%)", min_value=0.1, value=61.5
            )
            table = st.number_input(
                "Table (%)", min_value=0.1, value=57.0
            )

        with col3:
            x = st.number_input(
                "X dimension (mm)", min_value=0.1, value=5.7
            )
            y = st.number_input(
                "Y dimension (mm)", min_value=0.1, value=5.8
            )
            z = st.number_input(
                "Z dimension (mm)", min_value=0.1, value=3.5
            )

        diamond_predict = centered_submit("Estimate price")

    if diamond_predict:
        diamond_file = model_file("diamond", diamond_algo)
        model = load_model(diamond_file)

        if model is None:
            st.warning(
                f"{diamond_algo} is not trained yet. "
                f"Run train_models.py to create models/{diamond_file}."
            )
        else:
            # Each saved model is a full pipeline: it takes the original
            # columns and does its own scaling and encoding.
            features = pd.DataFrame([{
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "depth": depth,
                "table": table,
                "x": x,
                "y": y,
                "z": z,
                "volume": x * y * z
            }])

            try:
                prediction = model.predict(features)[0]
                sign = "-" if prediction < 0 else ""
                show_result(
                    f"Estimated price · {diamond_algo}",
                    f"{sign}${abs(prediction):,.2f}"
                )
            except Exception as e:
                st.error(
                    "Model input format needs to match your "
                    f"training pipeline. Details: {e}"
                )


# ==================================================
# 2. PURCHASE INTENTION - CLASSIFICATION
# ==================================================
# Only the 7 inputs that matter most are shown (chosen with the Decision
# Tree, then checked on all 10 algorithms). Every other column is held at
# a typical (median / most common) value from the training data, so the
# saved pipelines still get the full set of columns they were trained on.
PURCHASE_CONSTANTS = {
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 18,
    "SpecialDay": 0.0,
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}

with tab_purchase:
    st.subheader("Purchase predictor")
    st.caption(
        "Enter the session details to see if this visitor will buy."
    )

    purchase_algo = algorithm_picker(PURCHASE_ALGOS, "purchase_algo")

    with st.form("purchase_form"):
        col1, col2, col3, col4 = form_columns(4)

        with col1:
            page_values = st.number_input(
                "Page Values", min_value=0.0, value=0.0
            )
            bounce_rate = st.number_input(
                "Bounce Rate", min_value=0.0, max_value=1.0,
                value=0.02, format="%.3f"
            )

        with col2:
            exit_rate = st.number_input(
                "Exit Rate", min_value=0.0, max_value=1.0,
                value=0.05, format="%.3f"
            )
            product_duration = st.number_input(
                "Product-related Duration", min_value=0.0, value=600.0
            )

        with col3:
            administrative = st.number_input(
                "Administrative Pages", min_value=0, value=1
            )
            admin_duration = st.number_input(
                "Administrative Duration", min_value=0.0, value=10.0
            )

        with col4:
            month = st.selectbox(
                "Month",
                ["Feb", "Mar", "May", "June", "Jul",
                 "Aug", "Sep", "Oct", "Nov", "Dec"],
                index=2
            )

        purchase_predict = centered_submit("Check intention")

    if purchase_predict:
        purchase_file = model_file("purchase", purchase_algo)
        model = load_model(purchase_file)

        if model is None:
            st.warning(
                f"{purchase_algo} is not trained yet. "
                f"Run train_models.py to create models/{purchase_file}."
            )
        else:
            c = PURCHASE_CONSTANTS
            features = pd.DataFrame([{
                "Administrative": administrative,
                "Administrative_Duration": admin_duration,
                "Informational": c["Informational"],
                "Informational_Duration": c["Informational_Duration"],
                "ProductRelated": c["ProductRelated"],
                "ProductRelated_Duration": product_duration,
                "BounceRates": bounce_rate,
                "ExitRates": exit_rate,
                "PageValues": page_values,
                "SpecialDay": c["SpecialDay"],
                "Month": month,
                "OperatingSystems": c["OperatingSystems"],
                "Browser": c["Browser"],
                "Region": c["Region"],
                "TrafficType": c["TrafficType"],
                "VisitorType": c["VisitorType"],
                "Weekend": c["Weekend"],
                "TotalPages": (
                    administrative
                    + c["Informational"]
                    + c["ProductRelated"]
                )
            }])

            try:
                prediction = model.predict(features)[0]
                will_buy = (
                    str(prediction).lower() in ["true", "1", "yes"]
                    or prediction is True
                )
                show_result(
                    f"Prediction · {purchase_algo}",
                    "Likely to buy" if will_buy else "Unlikely to buy",
                    "good" if will_buy else "neutral"
                )
            except Exception as e:
                st.error(
                    "Model input format needs to match your "
                    f"training pipeline. Details: {e}"
                )


# ==================================================
# 3. CUSTOMER SEGMENTATION - CLUSTERING
# ==================================================
def assign_cluster(bundle, algorithm, values):
    """Preprocess one customer exactly like clustering.ipynb did, then
    return the cluster number for the chosen algorithm.

    Steps: log1p on the skewed features -> clip to the training 0.5% / 99.5%
    range -> StandardScaler. K-Means has .predict(). Agglomerative clustering
    has none, so the customer joins the cluster with the nearest centre."""
    features = bundle["features"]
    row = pd.DataFrame([values], columns=features)
    row[bundle["log_features"]] = np.log1p(row[bundle["log_features"]])
    row = row.clip(
        lower=bundle["clip_lower"], upper=bundle["clip_upper"], axis=1
    )
    scaled = bundle["scaler"].transform(row)

    if algorithm.startswith("K-Means"):
        return int(bundle["kmeans"].predict(scaled)[0])

    centres = bundle["agglomerative_centroids"]
    return int(np.linalg.norm(centres - scaled, axis=1).argmin())


with tab_cluster:
    st.subheader("Customer segmentation")
    st.caption(
        "Enter the buying history to see which group they belong to."
    )

    cluster_algo = algorithm_picker(CLUSTER_ALGOS, "cluster_algo")

    with st.form("clustering_form"):
        col1, col2, col3, col4 = form_columns(4)

        with col1:
            recency = st.number_input(
                "Recency (days)", min_value=0.0, value=30.0
            )
            frequency = st.number_input(
                "Frequency (orders)", min_value=0.0, value=12.0
            )

        with col2:
            monetary = st.number_input(
                "Monetary (total spend)", min_value=0.0, value=1500.0
            )
            avg_order_value = st.number_input(
                "Average Order Value", min_value=0.0, value=125.0
            )

        with col3:
            unique_products = st.number_input(
                "Unique Products", min_value=0.0, value=15.0
            )
            tenure_days = st.number_input(
                "Tenure (days)", min_value=0.0, value=365.0
            )

        with col4:
            cancel_rate = st.number_input(
                "Cancel Rate", min_value=0.0,
                max_value=1.0, value=0.05
            )

        cluster_predict = centered_submit("Find cluster")

    if cluster_predict:
        bundle = load_model("clustering_models.pkl")

        if bundle is None:
            st.warning(
                "Clustering models not connected yet. Add the cell from "
                "clustering_export_cell.py to the end of clustering.ipynb, "
                "run it, and put clustering_models.pkl in the models folder."
            )
        else:
            values = {
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "AvgOrderValue": avg_order_value,
                "UniqueProducts": unique_products,
                "TenureDays": tenure_days,
                "CancelRate": cancel_rate,
            }

            try:
                cluster = assign_cluster(bundle, cluster_algo, values)
                show_result(
                    f"Customer group · {cluster_algo}", f"Cluster {cluster}"
                )
                st.caption(
                    "Cluster numbers are separate for each algorithm, so "
                    "Cluster 1 in K-Means is not the same group as "
                    "Cluster 1 in Agglomerative."
                )
            except Exception as e:
                st.error(
                    "Check that the feature names, order, and "
                    f"preprocessing match training. Details: {e}"
                )