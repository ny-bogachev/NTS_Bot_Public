# import libraries
import asyncio
import nest_asyncio
import pandas as pd
from datetime import time, datetime, timedelta, timezone
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, ContextTypes
import pytz
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import subprocess
from telethon import TelegramClient
import re
import openai
import argparse
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# bot username and token
TOKEN = os.getenv("TELEGRAM_NTS_BOT_API")
BOT_USERNAME = '@your_bot_name'

# define file paths in the /app directory
USER_FILE = "/app/pickle_data/users.pkl"
NEWS_FILE = "/app/pickle_data/news_selec.pkl"
TECHNOLOGY_FILE = "/app/pickle_data/technology_selec.pkl"
SPORT_FILE = "/app/pickle_data/sport_selec.pkl"

nest_asyncio.apply()

# add user function
def add_user(chat_id):
    df_users = pd.read_pickle(USER_FILE)
    if chat_id not in df_users['chat_id'].values:
        new_user_df = pd.DataFrame({'chat_id': [chat_id]})
        df_users = pd.concat([df_users, new_user_df], ignore_index=True)
        df_users.to_pickle(USER_FILE)

# list of users function
def get_all_users():
    df_users = pd.read_pickle(USER_FILE)
    return df_users['chat_id'].tolist()

# send daily message to all users function
async def send_daily_update(context: CallbackContext):
    today_date = datetime.today().date().strftime("%d.%m.%Y")
    users = get_all_users()
    
    news_today = pd.read_pickle(NEWS_FILE)
    technology_today = pd.read_pickle(TECHNOLOGY_FILE)
    sport_today = pd.read_pickle(SPORT_FILE)
    
    for chat_id in users:
        
        await context.bot.send_message(chat_id=chat_id, text=f'Новости от {today_date}')      
        for i in range(len(news_today)):
            message = news_today["Link"][i]
            await context.bot.send_message(chat_id=chat_id, text=message)
            
        await context.bot.send_message(chat_id=chat_id, text=f'Новости технологий от {today_date}')
        for i in range(len(technology_today)):
            message = technology_today["Link"][i]
            await context.bot.send_message(chat_id=chat_id, text=message)
            
        await context.bot.send_message(chat_id=chat_id, text=f'Новости спорта от {today_date}')
        for i in range(len(sport_today)):
            message = sport_today["Link"][i]
            await context.bot.send_message(chat_id=chat_id, text=message)
            
        await asyncio.sleep(1)

# start command and scheduling
async def start_command(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    add_user(chat_id)
    await update.message.reply_text('Привет! Добро пожаловать в NTS_Bot. Здесь ежедневно публикуются самые интересные новости политики, технологий и спорта')

# schedule daily job at 8:00 AM Madrid time
    madrid_timezone = pytz.timezone('Europe/Madrid')
    target_time = time(8, 0, tzinfo=madrid_timezone)
    if not context.job_queue.get_jobs_by_name('daily_update'):
        context.job_queue.run_daily(send_daily_update, target_time, name='daily_update')
    await update.message.reply_text('Следующая подборка новостей запланирована на 8:00 (GMT+1 timezone). Если хотите получить копию последней подборки новостей прямо сейчас, пожалуйста, воспользуйтесь командой /news')

# help command
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Следующая подборка новостей запланирована на 8:00 (GMT+1 timezone), пожалуйста, подождите. Если хотите получить копию последней подборки новостей прямо сейчас, пожалуйста, воспользуйтесь командой /news')

# handle unknown messages
async def handle_message(update: Update, context: CallbackContext):
    await update.message.reply_text('Извините, мы не понимаем. Если хотите узнать о работе бота, пожалуйста, воспользуйтесь командой /help')

# error handling
async def error(update: Update, context: CallbackContext):
    print(f'Error: {context.error}')

# daily selection update function
async def run_news_selection():
    print('Starting update...')
    subprocess.run(["python", "news_selection.py"])
    
# send the last daily message on user request
async def latest_news_command(update: Update, context: CallbackContext):
    today_date = datetime.today().date().strftime("%d.%m.%Y")
    chat_id = update.effective_chat.id

    try:
        news_today = pd.read_pickle(NEWS_FILE)
        technology_today = pd.read_pickle(TECHNOLOGY_FILE)
        sport_today = pd.read_pickle(SPORT_FILE)

        await context.bot.send_message(chat_id=chat_id, text=f'Новости от {today_date}')      
        for i in range(len(news_today)):
            message = news_today["Link"][i]
            await context.bot.send_message(chat_id=chat_id, text=message)
            
        await context.bot.send_message(chat_id=chat_id, text=f'Новости технологий от {today_date}')
        for i in range(len(technology_today)):
            message = technology_today["Link"][i]
            await context.bot.send_message(chat_id=chat_id, text=message)
            
        await context.bot.send_message(chat_id=chat_id, text=f'Новости спорта от {today_date}')
        for i in range(len(sport_today)):
            message = sport_today["Link"][i]
            await context.bot.send_message(chat_id=chat_id, text=message)

    except Exception as e:
        print(f'Error sending latest news: {e}')
        await context.bot.send_message(chat_id=chat_id, text='Извините, не удалось получить последние новости. Попробуйте позже')

# main function
async def main():
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()
    
    scheduler = AsyncIOScheduler(timezone='Europe/Madrid')
    scheduler.add_job(run_news_selection, 'cron', hour=7, minute=45)
    scheduler.start()
    
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('news', latest_news_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error)
    
    print('Polling...')
    await app.run_polling(poll_interval=3)

if __name__ == '__main__':
    asyncio.run(main())