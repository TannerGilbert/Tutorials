from flask import Flask, request, render_template
from ludwig.api import LudwigModel
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load the model
model = LudwigModel.load('model')

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('text')
        # Make prediction
        df = pd.DataFrame([str(data)], columns=['content'])
        print(df.head())
        pred = model.predict(dataset=df, data_format='df')
        print(pred)
        return render_template('index.html', sentiment=pred['airline_sentiment_predictions'][0])
    return render_template('index.html', sentiment='')
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)