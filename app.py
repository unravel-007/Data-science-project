pip install Flask
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('winequality.csv')
df = df.fillna(df.mean())
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)
features = df.drop(['quality', 'best quality', 'total sulfur dioxide'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

models = {
    'LogisticRegression': LogisticRegression(),
    'XGBClassifier': XGBClassifier(),
    'SVC': SVC(kernel='rbf')
}

for model_name in models:
    models[model_name].fit(xtrain, ytrain)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = np.array([float(data[feature]) for feature in features.columns]).reshape(1, -1)
    input_data = norm.transform(input_data)
    
    predictions = {model_name: models[model_name].predict(input_data)[0] for model_name in models}
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
