import streamlit as st
import numpy as np
import pandas as pd
from streamlit.elements import layouts
import preprocessing
import tweepy
from tweepy import OAuthHandler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide", )
st.title("Welcome To Twitter Sentiment Analyzer")
#st.text("Analysis is done by Logistic Regression Classifier which gave accuracy = 80.8975%.")
st.sidebar.title("Twitter Sentiment Analyzer")
with st.sidebar:
    with st.form(key="form1"):
        #consumerKey =  st.text_input('Enter Consumer_Key')
        #consumer_secret =  st.text_input('Enter Consumer_Secret')
        #accessToken =  st.text_input('Enter Access_Token')
        #accessTokenSecret =  st.text_input('Enter Access_Token_Secret')
        keyword = st.text_input("Enter Keyword to Search")
        number_of_tweet = st.number_input("Enter Number of Tweets", min_value=1, step=1)
        submitted1 = st.form_submit_button('Submit')
        
def Api_Handling(consumer_key, consumer_secret, access_token, acces_token_secret):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, acces_token_secret)
    api = tweepy.API(auth)
    return api

def get_tweets(api, query, count):
    tweets = []
    fetched_tweets = api.search_tweets(q=query, count=count, lang='en')
    # parsing tweets one by one
    for tweet in fetched_tweets:
        parsed_tweet = {}
        # saving text of tweet
        parsed_tweet['tweet'] = tweet.text

        # appending parsed tweet to tweets list
        if tweet.retweet_count > 0:
            # if tweet has retweets, ensure that it is appended only once
            if parsed_tweet not in tweets:
                tweets.append(parsed_tweet)
        else:
            tweets.append(parsed_tweet)
    # return parsed tweets
    return tweets

 
consumerKey = st.secrets["consumer_Key"]
consumerSecret = st.secrets["consumer_Secret"]
accessToken = st.secrets["access_Token"]
accessTokenSecret = st.secrets["access_Token_Secret"]



api = Api_Handling(consumerKey, consumerSecret, accessToken, accessTokenSecret)
if submitted1:
    tweets = get_tweets(api, keyword, number_of_tweet)
    tweets_df = pd.DataFrame(tweets)
    st.header("Downloaded Tweets")
    st.write(tweets_df)

    tweets_df['tweet'] = np.vectorize(preprocessing.process_tweet)(tweets_df['tweet'])

    tweets_df['tweet'] = preprocessing.lematize(tweets_df)


    load_vectoriser = pd.read_pickle(r"vectoriser")
    tweets_df_vector = load_vectoriser.transform(tweets_df['tweet'])

    load_model = pd.read_pickle(r"classifier")
    predicted = load_model.predict(tweets_df_vector)

    tweets_df['sentiment'] = predicted
    tweets_df.loc[tweets_df['sentiment'] == 0, 'sentiment'] = 'negative'
    tweets_df.loc[tweets_df['sentiment'] == 1, 'sentiment'] = 'positive'

    st.header("Tweets with sentiment")
    st.write(tweets_df)

    st.header("Wordcloud of tweets")
    #plt.figure(figsize = (20,20))
    wc = WordCloud(width = 1600 , height = 800,
               collocations=False).generate(" ".join(tweets_df['tweet']))
    fig = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)

    positive_tweets = tweets_df[tweets_df['sentiment']=='positive']['tweet']
    negative_tweets = tweets_df[tweets_df['sentiment']=='negative']['tweet']

    st.header("Wordcloud of positive tweets")
    wc = WordCloud(width = 1600 , height = 800,
               collocations=False).generate(" ".join(positive_tweets))
    fig = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)

    st.header("Wordcloud of negative tweets")
    wc = WordCloud(width = 1600 , height = 800,
               collocations=False).generate(" ".join(negative_tweets))
    fig = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)


    st.header("Countplot of Tweets")
    fig = plt.figure()
    sns.countplot(x ='sentiment', data = tweets_df, palette ='coolwarm')
    st.pyplot(fig)

