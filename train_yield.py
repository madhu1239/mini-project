import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    "data/yield_model_ready_10crop_harmonized.csv"
)

features = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area"
]

target = "Yield_tonnes_per_hectare"

X = df[features]
y = df[target]

categorical_features = [
    "State_Name",
    "District_Name",
    "Season",
    "Crop"
]

numeric_features = [
    "Crop_Year",
    "Area"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    verbose=1
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

print("Training yield model...", flush=True)
pipeline.fit(X_train, y_train)
print("Training complete. Evaluating...", flush=True)

pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(
    y_test,
    pred
) ** 0.5

r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

joblib.dump(
    pipeline,
    "models/yield_model.pkl"
)

print("Yield model saved.")
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