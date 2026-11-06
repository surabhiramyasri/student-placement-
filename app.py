
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Student Placement Analysis",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Analysis")
st.markdown(
    "### Higher-Education Placement Data Mining"
)
st.write(
    "An interactive dashboard using Classification "
    "and Clustering techniques."
)

# ---------------- LOAD DATASET ----------------

st.sidebar.title("Navigation")

uploaded_file = st.sidebar.file_uploader(
    "Upload Placement CSV (optional)",
    type=["csv"]
)

@st.cache_data
def load_data(file_bytes=None):
    if file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes))

    file_path = Path(__file__).parent / "Placement_Data_Full_Class.csv"

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


try:
    df = load_data(
        uploaded_file.getvalue() if uploaded_file else None
    )
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

if df is None:
    st.error(
        "Dataset not found. Keep Placement_Data_Full_Class.csv "
        "in the same folder as app.py."
    )
    st.stop()

if "status" not in df.columns:
    st.error("Dataset must contain a 'status' column.")
    st.stop()

df = df.copy()
df["status"] = df["status"].astype(str).str.strip()

# Remove unnecessary columns
df = df.drop(
    columns=["sl_no", "salary"],
    errors="ignore"
)

# ---------------- SIDEBAR MENU ----------------

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "Dataset Analysis",
        "Classification",
        "Clustering",
        "Student Prediction"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Placement Overview")

    total = len(df)
    placed = (df["status"].str.lower() == "placed").sum()
    not_placed = total - placed
    percentage = placed / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total)
    col2.metric("Placed Students", int(placed))
    col3.metric("Not Placed", int(not_placed))
    col4.metric("Placement Rate", f"{percentage:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Placement Status")

        counts = df["status"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Placement Status")
        ax.set_ylabel("Number of Students")
        ax.set_title("Student Placement Status")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Average Academic Performance")

        academic_cols = [
            c for c in ["ssc_p", "hsc_p", "degree_p"]
            if c in df.columns
        ]

        if academic_cols:
            averages = df[academic_cols].mean()

            fig, ax = plt.subplots()
            averages.plot(kind="bar", ax=ax)
            ax.set_xlabel("Academic Level")
            ax.set_ylabel("Average Percentage")
            ax.set_title("Academic Performance")
            ax.tick_params(axis="x", rotation=0)

            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ---------------- DATASET ANALYSIS ----------------

elif page == "Dataset Analysis":

    st.subheader("Dataset Analysis")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Complete Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "placement_dataset.csv",
        "text/csv"
    )


# ---------------- CLASSIFICATION ----------------

elif page == "Classification":

    st.subheader("Student Placement Classification")

    X = df.drop(columns=["status"])
    y = df["status"]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        st.error("At least two classes with sufficient records are required.")
        st.stop()

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    transformers = []

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    if numerical_cols:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append(
            ("numerical", numerical_pipeline, numerical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Training Records", len(X_train))
    col3.metric("Testing Records", len(X_test))

    st.divider()

    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

    st.subheader("Actual vs Predicted")

    result = pd.DataFrame({
        "Actual Status": y_test.values,
        "Predicted Status": predictions
    })

    st.dataframe(result, use_container_width=True)


# ---------------- CLUSTERING ----------------

elif page == "Clustering":

    st.subheader("Student Academic Performance Clustering")

    features = [
        c for c in [
            "ssc_p",
            "hsc_p",
            "degree_p",
            "etest_p",
            "mba_p"
        ]
        if c in df.columns
    ]

    if len(features) < 2:
        st.error("At least two academic columns are required.")
        st.stop()

    X_cluster = df[features].apply(
        pd.to_numeric, errors="coerce"
    )

    X_cluster = X_cluster.fillna(
        X_cluster.mean()
    ).fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_cluster)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(scaled_data)

    cluster_df = df.copy()
    cluster_df["Cluster"] = clusters

    st.write("The algorithm divides students into three groups.")

    counts = cluster_df["Cluster"].value_counts().sort_index()

    col1, col2, col3 = st.columns(3)

    for col, cluster in zip(
        [col1, col2, col3], range(3)
    ):
        col.metric(
            f"Cluster {cluster}",
            int(counts.get(cluster, 0))
        )

    st.divider()

    st.subheader("Cluster Visualization")

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=clusters,
        cmap="viridis",
        s=50
    )

    ax.set_title("Student Academic Performance Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(
        *scatter.legend_elements(),
        title="Clusters"
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Performance by Cluster")

    cluster_means = cluster_df.groupby(
        "Cluster"
    )[features].mean().round(2)

    st.dataframe(cluster_means, use_container_width=True)

    st.subheader("Cluster Results")
    st.dataframe(cluster_df, use_container_width=True)

    st.download_button(
        "Download Cluster Results",
        cluster_df.to_csv(index=False).encode("utf-8"),
        "placement_clustering_results.csv",
        "text/csv"
    )


# ---------------- STUDENT PREDICTION ----------------

elif page == "Student Prediction":

    st.subheader("Predict Student Placement Status")

    st.write(
        "Enter a student's academic and personal details "
        "to estimate their placement status."
    )

    X = df.drop(columns=["status"])
    y = df["status"]

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    transformers = []

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        ))

    if numerical_cols:
        transformers.append((
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_cols
        ))

    model = Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    if y.nunique() < 2 or y.value_counts().min() < 2:
        st.error("Insufficient records for prediction.")
        st.stop()

    model.fit(X, y)

    with st.form("prediction_form"):

        user_data = {}

        col1, col2 = st.columns(2)

        for i, column in enumerate(X.columns):

            with (col1 if i % 2 == 0 else col2):

                if column in numerical_cols:
                    values = pd.to_numeric(
                        X[column], errors="coerce"
                    ).dropna()

                    if values.empty:
                        user_data[column] = 0.0
                    else:
                        minimum = float(values.min())
                        maximum = float(values.max())
                        default = float(values.median())

                        user_data[column] = st.number_input(
                            column,
                            min_value=minimum,
                            max_value=maximum,
                            value=min(max(default, minimum), maximum)
                        )

                else:
                    options = sorted(
                        X[column].dropna().astype(str).unique()
                    )

                    if options:
                        user_data[column] = st.selectbox(
                            column, options
                        )
                    else:
                        user_data[column] = ""

        submitted = st.form_submit_button(
            "Predict Placement"
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_data],
            columns=X.columns
        )

        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        if str(prediction).lower() == "placed":
            st.success(f"Predicted Status: {prediction}")
        else:
            st.warning(f"Predicted Status: {prediction}")

        st.caption(
            "This is a machine-learning estimate based on historical "
            "dataset patterns, not a guarantee of actual placement."
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Student Placement Analysis | "
    "Classification and Clustering | B.Tech AI & Data Science"
)
