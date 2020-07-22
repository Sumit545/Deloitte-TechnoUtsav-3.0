# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:21:02 2020

@author: Sumit Keshav
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template,Response,send_from_directory
import pickle as p
import pickle
import csv
import os

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/make_predictions')
def make_predictions():
    return render_template('make_predictions.html')

@app.route('/enter_value')
def enter_value():
    return render_template('predict.html')

@app.route('/upload_csv')
def upload_csv():
    return render_template('uploadcsv.html')

@app.route('/Cloud_ICU_Model')
def Cloud_ICU_Model():
    return render_template('Cloud_ICU_Model.html')

@app.route('/download_test_data')
def download_test_data():
    csv = '78,64.6,0.68,39.26666667,0.536363636,11.84615385,126.2,13.6,33.23333333,70.94520548,3.125124025,4.38,64.76666667,2.633333333,139.6,30.53333333,130.4,96.33333333,0,125.55,37.00555556,43.81081081,20,48.4,7.274\n65,36.77777778,2.18,75.19565217,0.535238542,12.125,195.1111111,24.77777778,36,84.47916667,4.29925197,3.671428571,105.4347826,2.133333333,164.5333333,50,98.5,92.5,1,158.4782609,37.99285714,202.96875,9.85,66.38333333,7.35\n48,6,2.85,50.08333333,0.566524229,15,86.5,23,24.45,93.85,2.824802129,3.65,64.14861111,1.3,136.5,37.36301976,169.9131127,145.75,0,92.27777778,37.2125,137.4534064,6.275,42.3,7.478900353\n25,12,0.6,61.91304348,0.516666667,13.22222222,93.5,26.66666667,24.725,110.0952381,2.47942704,4.15,76.39130435,2,139,41.7,315.5,243.75,1,107.0434783,36.78095238,173.0540541,9.95,78,7.401818182\n94,29.14285714,0.8,42.91935484,0.392857143,5.333333333,180.5714286,20,27.25,48.67241379,3.332625207,4.114285714,71.38709677,2.4,141.8571429,34.75,133.75,234.25,0,125.3387097,34.82545455,43.79411765,9.975,59,7.3825'
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=testdata.csv"})

@app.route('/predict_using_csv',methods=['POST','GET'])
def predict_using_csv():
    
    f = request.files['csvfile']
    f = pd.read_csv(f,header = None)
    f = list(f.to_numpy())

    data = []
    for row in f:
        data.append(row)
    
  
    count = 1
    predictionstrings = []
    predictionstate = []

    for d in data:

        form_value = [float(x) for x in d]
        feature_array = np.array(form_value).reshape(1,-1)
        std=p.load(open("scaler.p","rb"))
        #res = std.transform(form_value)
        res = np.array(form_value).reshape(1,-1)
        res = std.transform(res)
        model_prob = pickle.load(open("rf.pkl","rb"))
        model_reg = pickle.load(open("LoS_model.pkl","rb"))
        prediction_survival = model_prob.predict_proba(res)
        prediction_LoS = model_reg.predict(feature_array) # Need not apply scaling in this, since its Gradient Boosting
        output = '{0:.{1}f}'.format(prediction_survival[0][1],4)
        current_prediction = ""

        if output>str(0.311):
            current_prediction = 'Patient number {} is in CRITICAL STATE. Patient survives with a LOW probability of :{:.2f} %. The Length of Stay of the patient is: {} days'.format(count,(1.0-float(output))*100,int(prediction_LoS))
            predictionstate.append(True)
        else:
            current_prediction='Patient number {} is in SAFE STATE. Patient survives with a HIGH probability of: {:.2f} %. The Length of Stay of the patient is : {} days'.format(count,(1.0-float(output))*100,int(prediction_LoS))
            predictionstate.append(False)
        
        predictionstrings.append(current_prediction)
        count = count + 1

 
    return render_template('output_multi.html',pred = predictionstrings,predstate = predictionstate)


@app.route('/predict',methods=['POST','GET'])
def predict():
    
    form_value = [float(x) for x in request.form.values()]
    feature_array = np.array(form_value).reshape(1,-1)
    std=p.load(open("scaler.p","rb"))
    #res = std.transform(form_value)
    res = np.array(form_value).reshape(1,-1)
    res = std.transform(res)
    model_prob = pickle.load(open("rf.pkl","rb"))
    model_reg = pickle.load(open("LoS_model.pkl","rb"))
    prediction_survival = model_prob.predict_proba(res)
    prediction_LoS = model_reg.predict(feature_array) # Need not apply scaling in this, since its Gradient Boosting
    output = '{0:.{1}f}'.format(prediction_survival[0][1],4)
    
    if output>str(0.311):
        return render_template('output.html',pred = 'Patient is in CRITICAL STATE. Patient survives with a LOW probability of :{:.2f} %. The Length of Stay of the patient is: {} days'.format( (1.0-float(output))*100 , int(prediction_LoS) ), danger = True) 
    else:
        return render_template('output.html',pred = 'Patient is in SAFE STATE. Patient survives with a HIGH probability of: {:.2f} %. The Length of Stay of the patient is : {} days'.format((1.0-float(output))*100,int(prediction_LoS)), danger = False )
        
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
