import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for, flash
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
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
    final = pd.DataFrame(feature_list, index = [1])

    prediction = model.predict(final)
    print(prediction)
    return render_template('index.html', prediction = np.round(prediction, 3)[0])

@app.route('/help')
def help():
    return render_template('help.html')
    

if __name__ == "__main__":
    app.run(debug = False)