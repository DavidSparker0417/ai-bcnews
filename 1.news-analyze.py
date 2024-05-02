import requests
from dotenv import load_dotenv
import os
from newsapi import NewsApiClient

# load .env
load_dotenv()
secret=os.getenv('apikey')

# fetch articles from https://newsapi.org
newsapi = NewsApiClient(api_key=secret)
bitcoin_headlines = newsapi.get_everything(
  q='bitcoin',
  language='en',
  page=5,
  sort_by='relevancy')
ethereum_headlines = newsapi.get_everything(
  q='ethereum',
  language='en',
  page=5,
  sort_by='relevancy')


# create the Sentiment Scores DataFrame
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
bitcoin_sentiments = []
for article in bitcoin_headlines['articles']:
  try:
    text = article['content']
    date = article['publishedAt'][:10]
    sentiment = analyzer.polarity_scores(text)
    compound = sentiment['compound']
    pos = sentiment['pos']
    neu = sentiment['neu']
    neg = sentiment['neg']
    bitcoin_sentiments.append({
      'text': text,
      'date': date,
      'compound': compound,
      'positive': pos,
      'negative': neg,
      'neutral': neu
    })
  except AttributeError:
    pass