# import libraries
import nest_asyncio
import asyncio
from telethon import TelegramClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

# read lists of sources
with open("news.txt", "r") as file:
    news = [line.strip() for line in file]
    
with open("technology.txt", "r") as file:
    technology = [line.strip() for line in file]
    
with open("sport.txt", "r") as file:
    sport = [line.strip() for line in file]
    
# Telegram api and hash
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

# date threshold for data collecting
date_threshold = datetime.now(timezone.utc) - timedelta(days=100)

# functions for data collecting from Telegram
async def fetch_recent_posts(client, channel_username):
    messages_data = []
    async for message in client.iter_messages(channel_username):
        if message.date >= date_threshold:
            reactions = message.reactions
            message_text = message.text
            views = message.views
            message_link = f'{channel_username}/{message.id}'

            if reactions:
                reaction_count = {str(reaction.reaction): reaction.count for reaction in reactions.results}
            else:
                reaction_count = {}

            messages_data.append({'Date': message.date, 'Message': message_text, 'Reactions': reaction_count, 'Views': views, 'Link': message_link, 'Channel': channel_username})
        else:
            break

    if messages_data:
        df = pd.DataFrame(messages_data)
        if 'Message' in df.columns:
            df = df[df['Message'] != '']
        return df
    else:
        return pd.DataFrame()
    
async def fetch_all_channels():
    channel_dataframes = {}
    async with TelegramClient(f'session_{current_timestamp}', api_id, api_hash) as client:
        for channel in channels:
            df = await fetch_recent_posts(client, channel)
            if not df.empty:
                channel_dataframes[channel] = df
    return channel_dataframes

nest_asyncio.apply()

# news part
current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
channels = news

channel_dataframes = asyncio.get_event_loop().run_until_complete(fetch_all_channels())

columns = ['Message', 'Reactions', 'Views', 'Link', 'Date', 'Channel']
df_news = pd.DataFrame(columns=columns)
for channel, df in channel_dataframes.items():
    df_news = pd.concat([df_news, df], ignore_index=True)
    
df_news.to_pickle(f'df_news_{current_timestamp}.pkl')

# technology part
current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
channels = technology

channel_dataframes = asyncio.get_event_loop().run_until_complete(fetch_all_channels())

columns = ['Message', 'Reactions', 'Views', 'Link', 'Date', 'Channel']
df_news = pd.DataFrame(columns=columns)
for channel, df in channel_dataframes.items():
    df_news = pd.concat([df_news, df], ignore_index=True)
    
df_news.to_pickle(f'df_technology_{current_timestamp}.pkl')

# sport part
current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
channels = sport

channel_dataframes = asyncio.get_event_loop().run_until_complete(fetch_all_channels())

columns = ['Message', 'Reactions', 'Views', 'Link', 'Date', 'Channel']
df_news = pd.DataFrame(columns=columns)
for channel, df in channel_dataframes.items():
    df_news = pd.concat([df_news, df], ignore_index=True)
    
df_news.to_pickle(f'df_sport_{current_timestamp}.pkl')