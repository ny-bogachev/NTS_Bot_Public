# NTS_Bot_Public

Instructions and materials to create news-technology-sport Telegram Bot

![photo_2024-12-04 00 31 27](https://github.com/user-attachments/assets/5d41fe3b-cb96-46ec-9d7e-935a900b845e)

(ChatGPT generated picture)

## Instructions

If you want to create the same type of bot with some custom settings, take the following steps:

1. Create fly.io and OpenAI Platform accounts. The first one is needed for cloud deployment, and the second one is for the ad block part of the project
2. Create your bot using BotFather in Telegram
3. Create environmental variables (locally and in fly) OPENAI_API_KEY, TELEGRAM_API_HASH, TELEGRAM_API_ID, and TELEGRAM_NTS_BOT_API
4. Choose lists of sources (for NTS Bot they are stored in news.txt, sport.txt, and technology.txt)
5. Run collecting_old_posts.py and calculate_medians.py (first steps folder) to get reference values for your sources of information
6. Add my_bot_session.session file to the fly deployment folder (.session file generation is part of collecting_old_posts.py)
7. Upload the fly deployment folder to fly.io
8. Deploy your app
9. Enjoy!

If you have any questions, feel free to contact me on Telegram - @nybogachev
