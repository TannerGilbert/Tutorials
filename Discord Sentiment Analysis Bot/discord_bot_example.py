import discord

# create discord client
client = discord.Client()

# on message event-handler
@client.event
async def on_message(message):
    # ignore if the bot is the author
    if message.author == client.user:
        return
    await message.channel.send(message.content)

# run our bot
client.run('<your token>')
