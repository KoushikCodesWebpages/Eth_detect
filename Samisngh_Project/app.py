from flask import Flask, render_template, request

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


#install requirements using,
 
#pip install flask pandas scikit-learn

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv("Samisngh_Project\Ethereum_transaction_dataset.csv")

# Prepare the data
X = dataset.drop(columns=["Index", "Address", "FLAG"])
y = dataset["FLAG"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor for categorical and numerical variables
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the logistic regression model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),  # Scale numerical features
    ('classifier', LogisticRegression())
])

# Train the logistic regression model
model.fit(X_train, y_train)

# Function to predict fraud/non-fraud based on input address
def predict_fraud(address):
    # Locate the row corresponding to the input address
    row = dataset[dataset['Address'] == address]

    if len(row) == 0:
        return "Prediction: Fraudulent"

    # Extract features from the row
    input_data = row.drop(columns=["Index", "Address", "FLAG"])

    # Predict using the trained model
    prediction = model.predict(input_data)

    # Output the prediction
    if prediction[0] == 0:
        return "Prediction: Non-Fraudulent"
    else:
        return "Prediction: Fraudulent"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/output", methods=["POST"])
def predict():
    address = request.form.get("address")
    prediction = predict_fraud(address)
    if prediction == "Non-Fraudulent":
        css_class = "non-fraudulent"
    else:
        css_class = "fraudulent"
    return render_template("output.html", prediction=prediction, css_class=css_class)
if __name__ == "__main__":
    app.run(debug=True)
