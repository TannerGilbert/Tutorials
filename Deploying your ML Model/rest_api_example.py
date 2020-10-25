from flask import Flask, request, jsonify # loading in Flask
from ludwig.api import LudwigModel # loading in Ludwig
import pandas as pd # loading pandas for reading csv

# creating a Flask application
app = Flask(__name__)

# Load the model
model = LudwigModel.load('model')

# creating predict url and only allowing post requests.
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # Make prediction
    df = pd.DataFrame([str(data['text'])], columns=['content'])
    print(df.head())
    # making predictions
    pred = model.predict(dataset=df, data_format='df')
    print(pred)
    # returning the predictions as json
    return jsonify(pred['airline_sentiment_predictions'][0])

if __name__ == '__main__':
    app.run(port=3000, debug=True)