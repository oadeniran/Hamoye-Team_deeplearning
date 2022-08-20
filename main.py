import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for, flash
from sklearn import preprocessing
import pickle
import gc

app = Flask(__name__)
app.secret_key = 'team-deep-learning'
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
    feature_list = list(feature_list.values())
    feature_list = list(map(float, feature_list))
    final = np.array(feature_list).reshape(1,-1)

    prediction = model.predict(final)
    return render_template('index.html', prediction = np.round(prediction, 3)[0])

@app.route('/help')
def help():
    return render_template('help.html')
    

if __name__ == "__main__":
    app.run(debug = True)