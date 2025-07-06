import numpy as np
from flask import Flask, request, render_template
import joblib

# Create flask app
flask_app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get form values and convert to float
    float_features = [float(x) for x in request.form.values()]
    
    # Convert to 2D numpy array for scaler
    features = np.array(float_features).reshape(1, -1)
    
    # Scale input features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return prediction result
    return render_template("index.html", prediction_text=f"The Predicted Crop is {prediction[0]}")

if __name__ == "__main__":
    flask_app.run(debug=True)
