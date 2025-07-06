import numpy as np
from flask import Flask, request, render_template
import joblib
from pyngrok import ngrok

ngrok.kill()  # <-- Add this to clean old sessions
port_no = 5050
ngrok.set_auth_token("2zVOmKNVMzCREJ19Inm342g1GXi_39JpUqkvFb6a8oFU61vWZ")
public_url = ngrok.connect(port_no).public_url
print(" * Ngrok tunnel URL:", public_url)

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
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return render_template("index.html", prediction_text=f"The Predicted Crop is {prediction[0]}")

if __name__ == "__main__":
    flask_app.run(debug=True, port=port_no)
