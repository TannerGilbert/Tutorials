from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# create sentiment analyzer object
analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(text):
    score = analyzer.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 'positive'
    elif (lb > -0.05) and (lb < 0.05):
        return 'neutral'
    else:
        return 'negative'

print(sentiment_analyzer_scores('very bad'))