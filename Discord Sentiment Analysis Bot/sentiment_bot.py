import discord
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

client = discord.Client()
analyser = SentimentIntensityAnalyzer()
translator = Translator()

def sentiment_analyzer_scores(text):
    trans = translator.translate(text).text

    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 'positive'
    elif (lb > -0.05) and (lb < 0.05):
        return 'neutral'
    else:
        return 'negative'

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    sentiment = sentiment_analyzer_scores(message.content)
    print('sentiment: ' + str(sentiment))
    await message.channel.send('The sentiment of your text is ' + str(sentiment))


client.run('<your token>')