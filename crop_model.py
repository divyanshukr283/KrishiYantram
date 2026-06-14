import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load crop dataset
df = pd.read_csv("crop.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Input features
features = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = df[features]
y = df["label"]

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save trained model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Crop recommendation model saved successfully.")