
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Remove columns not needed for prediction
df = df.drop(columns=["sl_no", "salary"], errors="ignore")

# Separate input features and target
X = df.drop(columns=["status"])
y = df["status"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Convert text categories into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),
        ("numerical", "passthrough", numerical_cols)
    ]
)

# Create classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42
    ))
])

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Predict placement status
predictions = model.predict(X_test)

# Display results
accuracy = accuracy_score(y_test, predictions)

print("STUDENT PLACEMENT CLASSIFICATION")
print("--------------------------------")
print("Total students:", len(df))
print("Training students:", len(X_train))
print("Testing students:", len(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
