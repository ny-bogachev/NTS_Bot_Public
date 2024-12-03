# import libraries
import pandas as pd
from datetime import datetime, timedelta, timezone

# read historical dfs
df_news = pd.read_pickle('df_news_2024-10-27_23-06-09.pkl')
df_technology = pd.read_pickle('df_technology_2024-10-27_23-25-20.pkl')
df_sport = pd.read_pickle('df_sport_2024-10-27_23-26-46.pkl')

# drop empty posts
df_news_cleaned = df_news.dropna(subset=['Views']).copy()
df_technology_cleaned = df_technology.dropna(subset=['Views']).copy()
df_sport_cleaned = df_sport.dropna(subset=['Views']).copy()

df_news_cleaned.reset_index(drop=True, inplace=True)
df_technology_cleaned.reset_index(drop=True, inplace=True)
df_sport_cleaned.reset_index(drop=True, inplace=True)

# change data type for views column
df_news_cleaned['Views'] = df_news_cleaned['Views'].astype(int)
df_technology_cleaned['Views'] = df_technology_cleaned['Views'].astype(int)
df_sport_cleaned['Views'] = df_sport_cleaned['Views'].astype(int)

# localize posts' time
df_news_cleaned['Date'] = pd.to_datetime(df_news_cleaned['Date']).dt.tz_localize(None)
df_technology_cleaned['Date'] = pd.to_datetime(df_technology_cleaned['Date']).dt.tz_localize(None)
df_sport_cleaned['Date'] = pd.to_datetime(df_sport_cleaned['Date']).dt.tz_localize(None)

df_news_cleaned['Date'] = df_news_cleaned['Date'] + pd.Timedelta(hours=3)
df_technology_cleaned['Date'] = df_technology_cleaned['Date'] + pd.Timedelta(hours=3)
df_sport_cleaned['Date'] = df_sport_cleaned['Date'] + pd.Timedelta(hours=3)

df_news_cleaned_hist = df_news_cleaned[df_news_cleaned['Date'] <= pd.Timestamp('2024-10-24 23:59:59')].copy()
df_technology_cleaned_hist = df_technology_cleaned[df_technology_cleaned['Date'] <= pd.Timestamp('2024-10-24 23:59:59')].copy()
df_sport_cleaned_hist = df_sport_cleaned[df_sport_cleaned['Date'] <= pd.Timestamp('2024-10-24 23:59:59')].copy()

df_news_cleaned_hist.reset_index(drop=True, inplace=True)
df_technology_cleaned_hist.reset_index(drop=True, inplace=True)
df_sport_cleaned_hist.reset_index(drop=True, inplace=True)

# add day of week and channel name columns
df_news_cleaned_hist['Day_of_Week'] = df_news_cleaned_hist['Date'].dt.day_name()
df_technology_cleaned_hist['Day_of_Week'] = df_technology_cleaned_hist['Date'].dt.day_name()
df_sport_cleaned_hist['Day_of_Week'] = df_sport_cleaned_hist['Date'].dt.day_name()

df_news_cleaned_hist['Channel_Name'] = df_news_cleaned_hist['Channel'].str[13:]
df_technology_cleaned_hist['Channel_Name'] = df_technology_cleaned_hist['Channel'].str[13:]
df_sport_cleaned_hist['Channel_Name'] = df_sport_cleaned_hist['Channel'].str[13:]

# calculate views coef per channel per day of week
def calculate_median_views_per_channel_day(df):
    return df.groupby(['Channel_Name', 'Day_of_Week'])['Views'].transform('median')

df_news_cleaned_hist['Median_Views_Per_Day_of_Week'] = calculate_median_views_per_channel_day(df_news_cleaned_hist)
df_news_cleaned_hist['Views_Coef'] = df_news_cleaned_hist['Views'] / df_news_cleaned_hist['Median_Views_Per_Day_of_Week']

df_technology_cleaned_hist['Median_Views_Per_Day_of_Week'] = calculate_median_views_per_channel_day(df_technology_cleaned_hist)
df_technology_cleaned_hist['Views_Coef'] = df_technology_cleaned_hist['Views'] / df_technology_cleaned_hist['Median_Views_Per_Day_of_Week']

df_sport_cleaned_hist['Median_Views_Per_Day_of_Week'] = calculate_median_views_per_channel_day(df_sport_cleaned_hist)
df_sport_cleaned_hist['Views_Coef'] = df_sport_cleaned_hist['Views'] / df_sport_cleaned_hist['Median_Views_Per_Day_of_Week']

# save results
df_news_hist_med = df_news_cleaned_hist[['Channel','Channel_Name','Day_of_Week','Median_Views_Per_Day_of_Week']].copy()
df_technology_hist_med = df_technology_cleaned_hist[['Channel','Channel_Name','Day_of_Week','Median_Views_Per_Day_of_Week']].copy()
df_sport_hist_med = df_sport_cleaned_hist[['Channel','Channel_Name','Day_of_Week','Median_Views_Per_Day_of_Week']].copy()

df_news_hist_med = df_news_hist_med.drop_duplicates()
df_technology_hist_med = df_technology_hist_med.drop_duplicates()
df_sport_hist_med = df_sport_hist_med.drop_duplicates()

df_news_hist_med.reset_index(drop=True, inplace=True)
df_technology_hist_med.reset_index(drop=True, inplace=True)
df_sport_hist_med.reset_index(drop=True, inplace=True)

df_news_hist_med.to_pickle('df_news_hist_med.pkl')
df_technology_hist_med.to_pickle('df_technology_hist_med.pkl')
df_sport_hist_med.to_pickle('df_sport_hist_med.pkl')