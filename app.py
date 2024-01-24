from flask import Flask
from flask import Flask, render_template, request, redirect, url_for, flash, session,app,jsonify,url_for
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)
dataset = pd.read_csv('temp212.csv')  

regmodel = pickle.load(open('temp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predct_api',methods=['POST'])
def predct_api():
    data = request.get_json(force=True)
    # Access the temperature values within the 'data' key
    min_temp = data['data']['Min Temperature (°C)']
    max_temp = data['data']['Max Temperature (°C)']
    prediction = regmodel.predict([[min_temp, max_temp]])
    output = prediction[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=regmodel.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='Predicted Soil Mositure: {}%'.format(output))
if __name__ == "__main__":
    app.run(debug=True) 
