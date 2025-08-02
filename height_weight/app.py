from flask import Flask , request,jsonify , render_template
import pickle
import numpy as np

app=Flask(__name__)  #creating a Flask app

# Load trained model
model = pickle.load(open('height_weight_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = float(request.form['weight'])  # user input
    prediction = model.predict(np.array([[weight]]))  # 2D input
    height = round(prediction[0], 2)  # round to 2 decimals
    return render_template('index.html', prediction_text=f"Predicted height: {height} cm")

if __name__ == "__main__":
    app.run(debug=True)

