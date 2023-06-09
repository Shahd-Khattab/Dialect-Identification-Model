from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from joblib import load
from fastapi import FastAPI
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from pyarabic.araby import strip_tashkeel
import uvicorn
import pandas as pd
import io

import requests
import csv
headers= {
    "Accept": "application/json",
    "Content-Type": "multipart/form"
}
# Load LSTM model
lstm_model = load_model('lstm.h5')
svm_model = load('svm.joblib')
app = Flask(__name__)

@app.route('/svm', methods=['POST'])
def svm_predict():
    #csv_data = request.files['csvfile'].read().decode('utf-8')
    #data = pd.read_csv(io.StringIO(csv_data))
    data= pd.read_csv("preprocessedData.csv")
    #y = request.get_json()
    #features = np.array(data['features'])
    data= data.dropna(axis=0)
    data['fineText'] = data['fineText'].apply(lambda x: [str(word.split()) for word in x.split()]) 
    data['fineText'] = data['fineText'].apply(lambda x: ' '.join(map(str, x)))
    
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(data['fineText'])
    prediction = svm_model.predict(embeddings)

    return jsonify({'prediction': prediction.tolist()})

@app.route('/lstm', methods=['POST'])
def lstm_predict():
    data= pd.read_csv("preprocessedData.csv")
    #y = request.get_json()
    #features = np.array(data['features'])
    data= data.dropna(axis=0)
    data['fineText'] = data['fineText'].apply(lambda x: [str(word.split()) for word in x.split()]) 
    data['fineText'] = data['fineText'].apply(lambda x: ' '.join(map(str, x)))
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(data['fineText'])
    # Converting the splitted data to arrays
    embeddings= embeddings.toarray()

    # Converting the X_train and X_test to type float as they are written 0. not 0.0 
    # Reshaping them back to their original shape as they were flattened 
    embeddings=np.asarray(embeddings).astype('float32').reshape(embeddings.shape[0],1,embeddings.shape[1])

    prediction = lstm_model.predict(embeddings)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=8000)


if __name__ != '__main__':
    uvicorn.run(app, host='localhost', port=3000)  