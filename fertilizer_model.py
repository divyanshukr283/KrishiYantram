import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

print("Loading fertilizer dataset...")

# Load dataset
df = pd.read_csv("fertilizer.csv")
df.columns = df.columns.str.strip()

# Handle different column names in datasets
col_mapping = {col.lower(): col for col in df.columns}

temp_col = col_mapping.get("temperature", col_mapping.get("temparature"))
humidity_col = col_mapping.get("humidity")
moisture_col = col_mapping.get("moisture")
soil_col = col_mapping.get("soil type")
crop_col = col_mapping.get("crop type")
n_col = col_mapping.get("nitrogen")
k_col = col_mapping.get("potassium")
p_col = col_mapping.get("phosphorous", col_mapping.get("phosphorus"))
target_col = col_mapping.get("fertilizer name")

# Remove extra spaces from text columns
for col in [soil_col, crop_col, target_col]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Encode categorical values
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df[soil_col] = le_soil.fit_transform(df[soil_col])
df[crop_col] = le_crop.fit_transform(df[crop_col])
df[target_col] = le_fert.fit_transform(df[target_col])

# Input features
features = [
    temp_col,
    humidity_col,
    moisture_col,
    soil_col,
    crop_col,
    n_col,
    k_col,
    p_col
]

X = df[features]
y = df[target_col]

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model and encoders together
fert_package = {
    "model": model,
    "le_soil": le_soil,
    "le_crop": le_crop,
    "le_fert": le_fert
}

with open("fert_model.pkl", "wb") as file:
    pickle.dump(fert_package, file)

print("Fertilizer recommendation model saved successfully.")