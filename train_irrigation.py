import joblib
import pandas as pd

model = joblib.load(
    "models/yield_model.pkl"
)

input_data = pd.DataFrame([{
    "State_Name": "Telangana",
    "District_Name": "HYDERABAD",
    "Crop_Year": 2026,
    "Season": "Kharif",
    "Crop": "Rice",
    "Area": 2.0
}])

prediction = model.predict(input_data)[0]

print(
    "Expected Yield:",
    round(prediction, 2),
    "tonnes/hectare"
)
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    "data/irrigation_project_ready_10crop_harmonized.csv"
)

features = [
    "Crop",
    "Soil_Type",
    "Crop_Growth_Stage",
    "Irrigation_Method",
    "Water_Availability",
    "Soil_Moisture_pct",
    "Temperature_C",
    "Relative_Humidity_pct",
    "Rainfall_24h_mm",
    "Rainfall_7d_mm",
    "ETo_mm_day",
    "Land_Area_ha"
]

target = "Irrigation_Amount_mm_day"

X = df[features]
y = df[target]

categorical_features = [
    "Crop",
    "Soil_Type",
    "Crop_Growth_Stage",
    "Irrigation_Method",
    "Water_Availability"
]

numeric_features = [
    "Soil_Moisture_pct",
    "Temperature_C",
    "Relative_Humidity_pct",
    "Rainfall_24h_mm",
    "Rainfall_7d_mm",
    "ETo_mm_day",
    "Land_Area_ha"
]

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "num",
        "passthrough",
        numeric_features
    )
])

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print(
    "MAE:",
    mean_absolute_error(y_test, pred)
)

print(
    "RMSE:",
    mean_squared_error(
        y_test,
        pred
    ) ** 0.5
)

print(
    "R2:",
    r2_score(y_test, pred)
)

joblib.dump(
    pipeline,
    "models/irrigation_amount_model.pkl"
)