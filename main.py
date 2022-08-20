import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for, flash
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import gc

app = Flask(__name__)
app.secret_key = 'team-deep-learning'
class encode_columns(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        cols_to_transform = list(X.columns)
        if self.columns:
            cols_to_transform  = self.columns
        encoding = {
            'Classification_Size' : {
                'XL' : 1,
                'L' : 2,
                'M' : 3,
                'S' : 4},
            'Research_Intensity' : {
                'VH' : 1,
                'HI' : 2,
                'MD' : 3,
                'LO' : 4},
            'Status' : {
                'A' : 1,
                'B' : 2,
                'C' : 3}
            }
        for col in cols_to_transform:
            if col not in list(encoding.keys()):
                val_dict = {v: i for i, v in enumerate(np.unique(X[col]))}
                #print(val_dict)
            else:
                val_dict = encoding[col]
            X[col] = X[col].map(val_dict)
        return X
with open('final_model.pickle', 'rb') as f:
    model = pickle.load(f)





@app.route('/', methods = ['GET', 'POST'])
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST'])
def prediction():
    feature_list = request.form.to_dict()
    print(feature_list)
    final = pd.DataFrame(feature_list, index = [1])

    prediction = model.predict(final)
    print(prediction)
    return render_template('index.html', prediction = np.round(prediction, 3)[0])

@app.route('/help')
def help():
    return render_template('help.html')
    

if __name__ == "__main__":
    app.run(debug = False)