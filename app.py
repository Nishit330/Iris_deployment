from flask import Flask,jsonify,request
import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

import flask
app = Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('/index.html')


@app.route('/predict',methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    # pred_list = request.form.to_dict()
    sl = request.form['sl']
    pl = request.form['pl']
    sw = request.form['sw']
    pw = request.form['pw']
    array = np.array([float(sl),float(pl),float(sw),float(pw)])
    print(array)
    pred = clf.predict(array.reshape(1,-1))
    # print(sl)
    print(pred) 
    targetnames = ['Setosa', 'Versicolor', 'Virginica']
    if pred[0]==0:
        prediction = targetnames[0]
    elif pred[0]==1:
        prediction = targetnames[1]
    else:
        prediction = targetnames[2]
    

    return ("Flower is :"+ prediction)

if __name__ == '__main__':
    app.run()