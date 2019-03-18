from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# create sentiment analyzer object
analyzer = SentimentIntensityAnalyzer()

score = analyzer.polarity_scores('programming is fun :)')
print(score)