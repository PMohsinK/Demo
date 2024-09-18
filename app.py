from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
run_with_ngrok(app)  # This will set up ngrok for the Flask app

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Define the route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        DELS_Cation = float(request.form['DELS_Cation'])
        SpMax8_Bh = float(request.form['SpMax8_Bh(e)(cation)'])
        RDF035p = float(request.form['RDF035p(cation)'])
        Mor05m = float(request.form['Mor05m(cation)'])
        QXXi = float(request.form['QXXi(anion)'])

        # Create input feature array for the model
        input_features = np.array([[DELS_Cation, SpMax8_Bh, RDF035p, Mor05m, QXXi]])

        # Use the model to predict
        prediction = model.predict(input_features)

        # Render the result on the web page
        return render_template('index.html', prediction_text=f'Predicted logEC50(AChE): {prediction[0]:.4f}')
    except Exception as e:
        return str(e)

# Run the app
if __name__ == "__main__":
    app.run()
