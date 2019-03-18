from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

analyzer = SentimentIntensityAnalyzer()
translator = Translator()

def sentiment_analyzer_scores(text):
    trans = translator.translate(text).text

    score = analyzer.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 'positive'
    elif (lb > -0.05) and (lb < 0.05):
        return 'neutral'
    else:
        return 'negative'

print(sentiment_analyzer_scores('programmieren ist lustig'))