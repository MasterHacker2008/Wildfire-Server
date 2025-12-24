import joblib
import numpy as np
import sklearn

model = joblib.load("model.pkl")

print(model)

# Make predictions
# [Temperature, Humidity]
new_data = np.array([[21, 100-75]])

# Predict class and probability
prediction = model.predict(new_data)
prediction_proba = model.predict_proba(new_data)[:, 1]

print("Prediction (0 = low risk, 1 = high risk):", prediction[0])
print("Predicted risk probability:", prediction_proba[0] * 100)
