# import libraries
import nest_asyncio
import asyncio
from telethon import TelegramClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import re
import openai
import argparse
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Telegram api and hash
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

# define file paths in the /app directory
NEWS_FILE_PATH = "/app/pickle_data/news_selec.pkl"
TECHNOLOGY_FILE_PATH = "/app/pickle_data/technology_selec.pkl"
SPORT_FILE_PATH = "/app/pickle_data/sport_selec.pkl"

# define historical median files in the /app directory
NEWS_MED_FILE_PATH = "/app/pickle_data/df_news_hist_med.pkl"
TECHNOLOGY_MED_FILE_PATH = "/app/pickle_data/df_technology_hist_med.pkl"
SPORT_MED_FILE_PATH = "/app/pickle_data/df_sport_hist_med.pkl"

# read lists of sources
with open("/app/news.txt", "r") as file:
    news = [line.strip() for line in file]
    
with open("/app/technology.txt", "r") as file:
    technology = [line.strip() for line in file]
    
with open("/app/sport.txt", "r") as file:
    sport = [line.strip() for line in file]

# date threshold for data collecting (last 24 hours news)
date_threshold = datetime.now(timezone.utc) - timedelta(days=1)

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
    async with TelegramClient('my_bot_session', api_id, api_hash) as client:
        for channel in channels:
            df = await fetch_recent_posts(client, channel)
            if not df.empty:
                channel_dataframes[channel] = df
    return channel_dataframes

nest_asyncio.apply()

# news part
channels = news

channel_dataframes = asyncio.get_event_loop().run_until_complete(fetch_all_channels())

columns = ['Message', 'Reactions', 'Views', 'Link', 'Date', 'Channel']
df_news = pd.DataFrame(columns=columns)
for channel, df in channel_dataframes.items():
    df_news = pd.concat([df_news, df], ignore_index=True)
    
# technology part
channels = technology

channel_dataframes = asyncio.get_event_loop().run_until_complete(fetch_all_channels())

columns = ['Message', 'Reactions', 'Views', 'Link', 'Date', 'Channel']
df_technology = pd.DataFrame(columns=columns)
for channel, df in channel_dataframes.items():
    df_technology = pd.concat([df_technology, df], ignore_index=True)
    
# sport part
channels = sport

channel_dataframes = asyncio.get_event_loop().run_until_complete(fetch_all_channels())

columns = ['Message', 'Reactions', 'Views', 'Link', 'Date', 'Channel']
df_sport = pd.DataFrame(columns=columns)
for channel, df in channel_dataframes.items():
    df_sport = pd.concat([df_sport, df], ignore_index=True)
    
# clean collected data
df_news_cleaned = df_news.dropna(subset=['Views']).copy()
df_technology_cleaned = df_technology.dropna(subset=['Views']).copy()
df_sport_cleaned = df_sport.dropna(subset=['Views']).copy()

df_news_cleaned.reset_index(drop=True, inplace=True)
df_technology_cleaned.reset_index(drop=True, inplace=True)
df_sport_cleaned.reset_index(drop=True, inplace=True)

df_news_cleaned['Views'] = df_news_cleaned['Views'].astype(int)
df_technology_cleaned['Views'] = df_technology_cleaned['Views'].astype(int)
df_sport_cleaned['Views'] = df_sport_cleaned['Views'].astype(int)

df_news_cleaned['Date'] = pd.to_datetime(df_news_cleaned['Date']).dt.tz_localize(None)
df_technology_cleaned['Date'] = pd.to_datetime(df_technology_cleaned['Date']).dt.tz_localize(None)
df_sport_cleaned['Date'] = pd.to_datetime(df_sport_cleaned['Date']).dt.tz_localize(None)

df_news_cleaned['Date'] = df_news_cleaned['Date'] + pd.Timedelta(hours=3)
df_technology_cleaned['Date'] = df_technology_cleaned['Date'] + pd.Timedelta(hours=3)
df_sport_cleaned['Date'] = df_sport_cleaned['Date'] + pd.Timedelta(hours=3)

# add day of week and channel name columns
df_news_cleaned['Day_of_Week'] = df_news_cleaned['Date'].dt.day_name()
df_technology_cleaned['Day_of_Week'] = df_technology_cleaned['Date'].dt.day_name()
df_sport_cleaned['Day_of_Week'] = df_sport_cleaned['Date'].dt.day_name()

df_news_cleaned['Channel_Name'] = df_news_cleaned['Channel'].str[13:]
df_technology_cleaned['Channel_Name'] = df_technology_cleaned['Channel'].str[13:]
df_sport_cleaned['Channel_Name'] = df_sport_cleaned['Channel'].str[13:]

# read files with medians by week of day/channel
df_news_med = pd.read_pickle(NEWS_MED_FILE_PATH)
df_technology_med = pd.read_pickle(TECHNOLOGY_MED_FILE_PATH)
df_sport_med = pd.read_pickle(SPORT_MED_FILE_PATH)

# merge dfs
df_news_final = pd.merge(df_news_cleaned, df_news_med, on=['Channel', 'Channel_Name', 'Day_of_Week'], how='left')
df_technology_final = pd.merge(df_technology_cleaned, df_technology_med, on=['Channel', 'Channel_Name', 'Day_of_Week'], how='left')
df_sport_final = pd.merge(df_sport_cleaned, df_sport_med, on=['Channel', 'Channel_Name', 'Day_of_Week'], how='left')

# calculate views coefficient
df_news_final['Views_Coef'] = df_news_final['Views'] / df_news_final['Median_Views_Per_Day_of_Week']
df_technology_final['Views_Coef'] = df_technology_final['Views'] / df_technology_final['Median_Views_Per_Day_of_Week']
df_sport_final['Views_Coef'] = df_sport_final['Views'] / df_sport_final['Median_Views_Per_Day_of_Week']

# select hottest news
news_selec = df_news_final.sort_values(by='Views_Coef', ascending=False)[:15]
technology_selec = df_technology_final.sort_values(by='Views_Coef', ascending=False)[:5]
sport_selec = df_sport_final.sort_values(by='Views_Coef', ascending=False)[:15]

# remove links
news_selec['Message_Cleaned'] = news_selec['Message'].apply(lambda x: re.sub(r'\[.*?\]\(.*?\)', '', str(x)) if isinstance(x, str) else '')
technology_selec['Message_Cleaned'] = technology_selec['Message'].apply(lambda x: re.sub(r'\[.*?\]\(.*?\)', '', str(x)) if isinstance(x, str) else '')
sport_selec['Message_Cleaned'] = sport_selec['Message'].apply(lambda x: re.sub(r'\[.*?\]\(.*?\)', '', str(x)) if isinstance(x, str) else '')

# remove emojis and **
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

news_selec['Message_Cleaned'] = news_selec['Message_Cleaned'].apply(lambda x: remove_emojis(x) if isinstance(x, str) else '')
technology_selec['Message_Cleaned'] = technology_selec['Message_Cleaned'].apply(lambda x: remove_emojis(x) if isinstance(x, str) else '')
sport_selec['Message_Cleaned'] = sport_selec['Message_Cleaned'].apply(lambda x: remove_emojis(x) if isinstance(x, str) else '')

news_selec['Message_Cleaned'] = news_selec['Message_Cleaned'].apply(lambda x: x.replace("**", "") if isinstance(x, str) else '')
technology_selec['Message_Cleaned'] = technology_selec['Message_Cleaned'].apply(lambda x: x.replace("**", "") if isinstance(x, str) else '')
sport_selec['Message_Cleaned'] = sport_selec['Message_Cleaned'].apply(lambda x: x.replace("**", "") if isinstance(x, str) else '')

# custom ad block
client = openai.OpenAI()

def create_prompt(text):
    instructions = "Содержит ли данный пост рекламу/раздачу призов/ставки?"
    formatting = 'Пожалуйста, в ответе используй только одно слово без точки: "Да" или "Нет"'
    return f'Text:{text}\n{instructions}\nAnswer ({formatting}):'


def call_llm(prompt):
    messages = [{'content':prompt, 'role':'user'}]
    response = client.chat.completions.create(messages=messages, model='gpt-4o')
    return response.choices[0].message.content


def classify(text):
    prompt = create_prompt(text)
    answer = call_llm(prompt)
    return answer

news_selec['Ad?'] = news_selec['Message_Cleaned'].apply(classify)
technology_selec['Ad?'] = technology_selec['Message_Cleaned'].apply(classify)
sport_selec['Ad?'] = sport_selec['Message_Cleaned'].apply(classify)

news_selec_final = news_selec[news_selec['Ad?'] == 'Нет']
technology_selec_final = technology_selec[technology_selec['Ad?'] == 'Нет']
sport_selec_final = sport_selec[sport_selec['Ad?'] == 'Нет']

news_selec_final.reset_index(drop=True, inplace=True)
technology_selec_final.reset_index(drop=True, inplace=True)
sport_selec_final.reset_index(drop=True, inplace=True)

news_selec_final = news_selec_final.copy()
technology_selec_final = technology_selec_final.copy()
sport_selec_final = sport_selec_final.copy()

# lemmatization
def clean_message(message):
    cleaned_message = re.sub(r'\[.*?\]', '', message)
    cleaned_message = re.sub(r'\(http\S+\)', '', cleaned_message)
    cleaned_message = re.sub(r'http\S+', '', cleaned_message)
    cleaned_message = re.sub(r'@\w+', '', cleaned_message)
    cleaned_message = re.sub(r'#\w+', '', cleaned_message)
    cleaned_message = re.sub(r'_', '', cleaned_message)
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()
    cleaned_message = re.sub(r'[^\w\s]', '', cleaned_message)
    return cleaned_message.lower()

news_selec_final['Message_Cleaned'] = news_selec_final['Message_Cleaned'].apply(lambda x: clean_message(x) if isinstance(x, str) else '')
technology_selec_final['Message_Cleaned'] = technology_selec_final['Message_Cleaned'].apply(lambda x: clean_message(x) if isinstance(x, str) else '')
sport_selec_final['Message_Cleaned'] = sport_selec_final['Message_Cleaned'].apply(lambda x: clean_message(x) if isinstance(x, str) else '')

morph = pymorphy2.MorphAnalyzer()
news_selec_final['Message_Lemma'] = news_selec_final['Message_Cleaned'].apply(lambda text: ' '.join([morph.parse(word)[0].normal_form for word in text.split()]))
technology_selec_final['Message_Lemma'] = technology_selec_final['Message_Cleaned'].apply(lambda text: ' '.join([morph.parse(word)[0].normal_form for word in text.split()]))
sport_selec_final['Message_Lemma'] = sport_selec_final['Message_Cleaned'].apply(lambda text: ' '.join([morph.parse(word)[0].normal_form for word in text.split()]))

# drop news duplicates
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(news_selec_final['Message_Lemma'])

vectors = tfidf_matrix.toarray()
cos_sim_matrix = cosine_similarity(vectors)

to_drop = set()

for i in range(len(cos_sim_matrix)):
    for j in range(i + 1, len(cos_sim_matrix)):
        if cos_sim_matrix[i][j] > 0.2:
            if len(news_selec_final['Message_Lemma'][i]) > len(news_selec_final['Message_Lemma'][j]):
                to_drop.add(j)
            else:
                to_drop.add(i)

news_selec_very_final = news_selec_final.drop(list(to_drop)).reset_index(drop=True)

# drop technology duplicates
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(technology_selec_final['Message_Lemma'])

vectors = tfidf_matrix.toarray()
cos_sim_matrix = cosine_similarity(vectors)

to_drop = set()

for i in range(len(cos_sim_matrix)):
    for j in range(i + 1, len(cos_sim_matrix)):
        if cos_sim_matrix[i][j] > 0.2:
            if len(technology_selec_final['Message_Lemma'][i]) > len(technology_selec_final['Message_Lemma'][j]):
                to_drop.add(j)
            else:
                to_drop.add(i)

technology_selec_very_final = technology_selec_final.drop(list(to_drop)).reset_index(drop=True)

# drop sport duplicates
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sport_selec_final['Message_Lemma'])

vectors = tfidf_matrix.toarray()
cos_sim_matrix = cosine_similarity(vectors)

to_drop = set()

for i in range(len(cos_sim_matrix)):
    for j in range(i + 1, len(cos_sim_matrix)):
        if cos_sim_matrix[i][j] > 0.2:
            if len(sport_selec_final['Message_Lemma'][i]) > len(sport_selec_final['Message_Lemma'][j]):
                to_drop.add(j)
            else:
                to_drop.add(i)

sport_selec_very_final = sport_selec_final.drop(list(to_drop)).reset_index(drop=True)

news_selec_very_final = news_selec_very_final.copy()
technology_selec_very_final = technology_selec_very_final.copy()
sport_selec_very_final = sport_selec_very_final.copy()

# select column for mailing
news_selec_very_final = news_selec_very_final[['Link']]
technology_selec_very_final = technology_selec_very_final[['Link']]
sport_selec_very_final = sport_selec_very_final[['Link']]

# save the results in the /app directory
news_selec_very_final.to_pickle(NEWS_FILE_PATH)
technology_selec_very_final.to_pickle(TECHNOLOGY_FILE_PATH)
sport_selec_very_final.to_pickle(SPORT_FILE_PATH)