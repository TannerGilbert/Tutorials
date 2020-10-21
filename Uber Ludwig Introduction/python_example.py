from ludwig.api import LudwigModel
import pandas as pd

df = pd.read_csv('Tweets.csv')
print(df.head())

model_definition = {
    'input_features':[
        {'name':'text', 'type':'text'},
    ],
    'output_features': [
        {'name': 'airline_sentiment', 'type': 'category'}
    ]
}

print('creating model')
model = LudwigModel(model_definition)
print('training model')
train_stats, _, _ = model.train(dataset=df)