from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import joblib
from aimodel.insurance import predict_charges

app = Flask(__name__)

# Load the model
model_path = "insurance_model.pkl"
model = joblib.load(model_path)

# Function to preprocess input
def preprocess_input(age, sex, bmi, smoker):
    le = LabelEncoder()
    sex = le.fit_transform([sex])[0]
    smoker = le.fit_transform([smoker])[0]
    return age, sex, bmi, smoker

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    sex = int(request.form["sex"])
    bmi = float(request.form["bmi"])
    smoker = int(request.form["smoker"])

    age, sex, bmi, smoker = preprocess_input(age, sex, bmi, smoker)
    input_data = [[age, sex, bmi, smoker]]
    predicted_charges = model.predict(input_data)[0]

    return render_template("index.html", prediction=f"${predicted_charges:.2f}")

if __name__ == "__main__":
    app.run(debug=True)

