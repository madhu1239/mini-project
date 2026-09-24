import pandas as pd

crop_df = pd.read_csv("data/Crop_Recommendation_10crop_harmonized.csv")
yield_df = pd.read_csv("data/yield_model_ready_10crop_harmonized.csv")
irr_df = pd.read_csv("data/irrigation_project_ready_10crop_harmonized.csv")

print("Crop dataset:", crop_df.shape)
print("Yield dataset:", yield_df.shape)
print("Irrigation dataset:", irr_df.shape)

print(crop_df.head())
print(yield_df.head())
print(irr_df.head())
print(crop_df.isnull().sum())
print(yield_df.isnull().sum())
print(irr_df.isnull().sum())
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/Crop_Recommendation_10crop_harmonized.csv")

features = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH_Value",
    "Rainfall"
]

X = df[features]
y = df["Crop"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Crop Recommendation Accuracy:", accuracy)
print(classification_report(
    y_test,
    pred,
    target_names=encoder.classes_
))

joblib.dump(model, "models/crop_model.pkl")
joblib.dump(encoder, "models/crop_encoder.pkl")

print("Crop model saved.")
import joblib
import pandas as pd

model = joblib.load("models/crop_model.pkl")
encoder = joblib.load("models/crop_encoder.pkl")

input_data = pd.DataFrame([{
    "Nitrogen": 90,
    "Phosphorus": 50,
    "Potassium": 45,
    "Temperature": 27,
    "Humidity": 75,
    "pH_Value": 6.5,
    "Rainfall": 100
}])

prediction = model.predict(input_data)

crop = encoder.inverse_transform(prediction)[0]

print("Recommended Crop:", crop)
probabilities = model.predict_proba(input_data)[0]

results = pd.DataFrame({
    "Crop": encoder.classes_,
    "Probability": probabilities
})

results = results.sort_values(
    "Probability",
    ascending=False
)

print(results.head(5))