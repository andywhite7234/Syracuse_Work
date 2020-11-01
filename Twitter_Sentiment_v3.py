# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:26:00 2020

@author: andy_white
"""


from datetime import date, timedelta
import datetime
from time import gmtime, strftime
import time
import pandas as pd
import numpy as np
# To download a package:
#pip install GetOldTweets3
import GetOldTweets3 as got
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.model_selection import train_test_split
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
#Update your name and Asset_class, this will be important when saving files:
your_name = 'Andy_White'
asset_class = 'CMBS'
#also if you are in abs auto update something like abs_auto
#Finally, if you have provided twitter accounts in the excel doc and don't care how I built the functions
# jump to line 144 the line starts with "def tweets_by_practice(group,date_start,date_end): "

#Keyword list, this is specific to you asset class:
#thanks to bill for finding this - if you '"return of the mack"' will return only tweets that have
# "return of the mack" in order
import tweepy
import webbrowser
import time
import pandas as pd
import datetime
from time import gmtime, strftime
import time
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly
consumer_key = 'qsuSHdhBUlKPZxYzVgQyzLMgD'
consumer_secret = 'iDW40Fo84F95hBtw9tXwkmY7W7See1IWfjJw3jpLXUM4zk8HSh'

callback_uri = 'oob'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret,callback_uri)
#this is like "logging" into the website.

redirect_url = auth.get_authorization_url()
print(redirect_url)
#this takes you to twitters authorization of the the application i made. And i recieved a pin: 7395396
webbrowser.open(redirect_url)
#user input that allows us to add a pin:
user_pin_input = input("What's the Pin Value? ")

auth.get_access_token(user_pin_input)
print(auth.access_token,auth.access_token_secret)
#print confirms that access token is the same - so just a check

api = tweepy.API(auth)
print(api.me().screen_name)


'''
THE MAIN FUCTION!!!!!! - SEARCH BY KEYWORD. THE 'keyword' VARIABLE NEEDS TO BE UPDATED WITH YOUR KEYWORDS 
IF NOT THE FUNCTION BELOW WILL NOT RETURN TWEETS. 
'''
#i got rid of CLOs as a keword, it was pulling in too many garbage tweets
keywords = ['"credit card securitization"','"Lease securitization"','"auto loan securitization"','"Student loan securitization"',
           '"Asset-backed securitization"','"asset-backed securitization"','"subprime auto"','"auto loan ABS"','"student loan ABS"','"car loan ABS"','"aircraft lease securitization"',
           '"aircraft lease ABS"','"collateralized loan obligations"','"leveraged loans"','"leveraged finance"',
           '"residential mortgage backed securities"','"rmbs"','"residential mortgage-backed securities"','"prime jumbo"',
           '"prime mortgages"','"commercial mortgage-backed"','"commercial mortgage backed"','"industrial real estate"','"RevPAR"','CMBX',
           '"multifamily rents"','"retail rents"','"office rents"','"retail vacancy"','"retial vacancies"','"office vacancy"',
           '"office vacancies"']

#####Lets give it a shot here:
api = tweepy.API(auth)
trends = api.trends_place(1)

def tweepy_keyword_search(keyword):
    tweets_list = []
    for item in keywords:
        #this will feed each keyword into the tweepy api - and then filter out retweets (so as not to bias sentiment)
        search_hashtag = tweepy.Cursor(api.search, q=str(item+' -filter:retweets'), tweet_mode = "extended",lang="en").items(100)
        for tweet in search_hashtag:
            if 'retweeted_status' in tweet._json: 
                full_text = tweet._json['retweeted_status']['full_text']
            else:
                full_text = tweet.full_text
                            
            tweets_list.append([tweet.user.screen_name,
                                tweet.id,
                                tweet.retweet_count, # What you are interested of
                                tweet.favorite_count, # Maybe it is helpfull too
                                full_text,
                                tweet.created_at,
                                tweet.entities
                               ])
    tweets_df = pd.DataFrame(tweets_list, columns = ["screen_name", "tweet_id",
                                                  "no rt", 
                                                  "no replies",
                                                  "text",
                                                  "created_at", 
                                                  "entities"])
    return tweets_df

def save_tweets(DF,version):
    file_path = "C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/data_dump/"
    time_stamp = strftime("%Y-%m-%d_%H", gmtime())
    #a fail safe in case python shuts down on you when running naive bayes:
    DF.to_excel(file_path+'tweets_'+time_stamp+'_'+version+'.xlsx',index=False, encoding='utf-8')
    return str(file_path+'_'+version+'.xlsx')


tweets_df = tweepy_keyword_search(keywords)
save_tweets(tweets_df,'')

#now comes the NLP/Text Mining aspect, lets start with an example
test_sent = "It's crazy. This is the U.S.A. and I spent $12.99 on a 1,000 pound dog that won't play fetch!!! It's a dog-eat-dog world #fun"
import nltk
import re
from nltk.collocations import *
from nltk.corpus import stopwords
bigram_measures = nltk.collocations.BigramAssocMeasures()
nltk.word_tokenize(test_sent)
#nltk.download('nps_chat')
#nltk.corpus.nps_chat.tagged_words()[:50]

#surprisingly this tokenizes pretty well, 'U.S.A.' tokenizes together especially for polarity, one change I would make is tokenize '!!!' 
#together and work with the hashtags better

#I leave regex here so you can see how it works on a test sentence
#the following is regex expression code. I will comment what it's doing, but know that regex is it's own language
tweet_pattern = r''' (?x)    #set flag to allow verbose regexps
    (?:https?://|www)\S+     #gets simple URLS  
    | (?::-\)|;-\))          # small list of emoticons
    | &(?:amp|lt|gt|quot);   #XML or HTML entity
    | \#\w+                  #isolates hashtags
    | @\w+                   # isolates mentions like @wu_tang
    | \d+:\d+                # isolates timelike pattern
    | \d+\.d+                #numbers with decimals
    | (?:\d+,)+?\d{3}(?=(?:[^,]|$)) #numbers with a comma
    | \$?\d+(?:\.\d+)?%?         #dollars with numbers and percentages
    | (?:[A-z]\.)+           #simple abbreviations like U.S.A.
    | (?:--+)                #multiple dashes
    | [A-Z|a-z][a-z]*'[a-z]  #deals with contracionts like it's won't
    | \w+(?:-\w+)*           #words with internal hyphens or apostrophes
    | ['\".?!,:;/]+          # special characters
'''


nltk.regexp_tokenize(test_sent, tweet_pattern)

'''
Since we will be generally looking at news accounts, it's unlikely that they will say things like "I HATE ALL THINGS"
or speak in caps, for this reason i will lowercase all strings. so we can get a better idea of word fequency given the lower sample size
also i will be breaking up our unigrams and bigrams into a stopped and non-stopped list.
A stopword is something like 'the' 'a' or something that is very common in the english language. We will see if this improves accuracy
all this slicing and dicing will likely take some computational power
'''

def preprocess_unigram(sentence):
    sentence=str(sentence)  #takes sentence and converts to string (if necessary)
    sentence = sentence.lower() #lowercases - caps are likely important
    sentence = re.sub(r'http\S+', '',sentence)                     
    #filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]  #going to remove stopwords
    tweet_pattern = r''' (?x)    #set flag to allow verbose regexps
    (?:https?://|www)\S+     #gets simple URLS  
    | (?::-\)|;-\))          # small list of emoticons
    | &(?:amp|lt|gt|quot);   #XML or HTML entity
    | \#\w+                  #isolates hashtags
    | @\w+                   # isolates mentions like @wu_tang
    | \d+:\d+                # isolates timelike pattern
    | \d+\.d+                #numbers with decimals
    | (?:\d+,)+?\d{3}(?=(?:[^,]|$)) #numbers with a comma
    | \$?\d+(?:\.\d+)?%?         #dollars with numbers and percentages
    | (?:[A-z]\.)+           #simple abbreviations like U.S.A.
    | (?:--+)                #multiple dashes
    | [A-Z|a-z][a-z]*'[a-z]  #deals with contracionts like it's won't
    | \w+(?:-\w+)*           #words with internal hyphens or apostrophes
    | ['\".?!,:;/]+          # special characters
      '''
    return nltk.regexp_tokenize(sentence,tweet_pattern)
preprocess_unigram(test_sent)
def preprocess_stopped(sentence):
    sentence=str(sentence)  #takes sentence and converts to string (if necessary)
    sentence = sentence.lower() #lowercases - caps are likely important
    sentence = re.sub(r'http\S+', '',sentence)   
    tweet_pattern = r''' (?x)    #set flag to allow verbose regexps
    (?:https?://|www)\S+     #gets simple URLS  
    | (?::-\)|;-\))          # small list of emoticons
    | &(?:amp|lt|gt|quot);   #XML or HTML entity
    | \#\w+                  #isolates hashtags
    | @\w+                   # isolates mentions like @wu_tang
    | \d+:\d+                # isolates timelike pattern
    | \d+\.d+                #numbers with decimals
    | (?:\d+,)+?\d{3}(?=(?:[^,]|$)) #numbers with a comma
    | \$?\d+(?:\.\d+)?%?         #dollars with numbers and percentages
    | (?:[A-z]\.)+           #simple abbreviations like U.S.A.
    | (?:--+)                #multiple dashes
    | [A-Z|a-z][a-z]*'[a-z]  #deals with contracionts like it's won't
    | \w+(?:-\w+)*           #words with internal hyphens or apostrophes
    | ['\".?!,:;/]+          # special characters
      '''
    tokens = nltk.regexp_tokenize(sentence,tweet_pattern)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]  #going to remove stopwords    
    return filtered_words
preprocess_stopped(test_sent)

def tweet_unigram_features(tweets, word_features):
	tweet_words = set(tweets)
	features = {}
	for word in word_features:
		features['V_{}'.format(word)] = (word in tweet_words) # v for vocabulary and 
	return features

def bigram_document_features(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
    return features

le = LabelEncoder()

def add_labels(df,lab=''):
    tmp = copy.deepcopy(df)
    if lab == "polarity":
        tmp[lab]= MyTwitterLabel
    else:
        print('Use either Lie LABEL or Senti LABEL')
        return()
    return(tmp)

#in order to get word counts, we will be passing a dummy function to the tokenizer, since we have already tokenized
def dummy_fun(doc):
    return doc
#get the count vectorizer and the inverse frequency vectorizer. the first emphasizes common words, the secon emphasizes limited use

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

cv = CountVectorizer(
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
    )  


#now see what happens in our test sentence:
preprocess_stopped(test_sent)

#mil_tweets = pd.read_csv("more_tweets.csv",encoding="latin-1")
#list(mil_tweets.columns.values)
# read in the test doc:
fp = "C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/CMBS_Andy_White_v4.xlsx"
tweets_class = pd.read_excel(fp)
#need to see if we can apply the pattern to a df

def tokenize_prep(df):
    '''
    Need to tokenize, so the "df['unigram']=df['text'].map(lambda s:preprocess_unigram(s))" is basically applying our
    unigram preprocess function from above to tokenize, and the preprocess_stopped then removes stop words
    '''
    df.sample(frac=1)   #randomize rows in the dataframe
    df.loc[df['Sentiment_Analysis_1']==-1,'polarity']='neg'
    df.loc[df['Sentiment_Analysis_1']==0,'polarity']='nuet'
    df.loc[df['Sentiment_Analysis_1']==1,'polarity']='pos'
    df['unigram']=df['text'].map(lambda s:preprocess_unigram(s))  #this tokenizes by unigram
    df['unigram_stop']=df['text'].map(lambda s:preprocess_stopped(s)) #tokenizes by unigram with stop words removed
    df['bigram']=df['unigram'].map(lambda s:list(nltk.bigrams(s))) #create bigrams from our unigram tokens
    df['bigram_stopped']=df['unigram_stop'].map(lambda s:list(nltk.bigrams(s))) #creates bigrams from our unigram with stop words removed
    url = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"
    more_tweets = pd.read_csv(url,index_col=0)
    pos_tweets = more_tweets[more_tweets['Sentiment']=="positive"]
    pos_tweets = pos_tweets.sample(n=400,random_state=1)
    pos_tweets.loc[pos_tweets['Sentiment']=="positive",'polarity']='pos'
    pos_tweets['unigram']=pos_tweets['TweetText'].map(lambda s:preprocess_unigram(s))  #this tokenizes by unigram
    pos_tweets['unigram_stop']=pos_tweets['TweetText'].map(lambda s:preprocess_stopped(s)) #tokenizes by unigram with stop words removed
    pos_tweets['bigram']=pos_tweets['unigram'].map(lambda s:list(nltk.bigrams(s))) #create bigrams from our unigram tokens
    pos_tweets['bigram_stopped']=pos_tweets['unigram_stop'].map(lambda s:list(nltk.bigrams(s)))
    
    MyTwitterLabel = df['polarity'].append(pos_tweets['polarity']) #double brakets get into list format
    MyTwitterLabel = MyTwitterLabel.reset_index()
    MyTwitterLabel = MyTwitterLabel.drop('index',1)
    unigram = df['unigram'].append(pos_tweets['unigram'])
    unigram = unigram.reset_index()
    unigram = unigram.drop('index',1)
    unigram_stop = df['unigram_stop'].append(pos_tweets['unigram_stop'])
    unigram_stop = unigram_stop.reset_index()
    unigram_stop = unigram_stop.drop('index',1)
    bigram = df['bigram'].append(pos_tweets['bigram'])
    bigram = bigram.reset_index()
    bigram = bigram.drop('index',1)
    bigram_stopped =df['bigram_stopped'].append(pos_tweets['bigram_stopped'])
    bigram_stopped = bigram_stopped.reset_index()
    bigram_stopped = bigram_stopped.drop('index',1)
    return pd.DataFrame({'unigram':unigram['unigram'], 'unigram_stop': unigram_stop['unigram_stop'],
                         'bigram':bigram['bigram'],'bigram_stopped':bigram_stopped['bigram_stopped']}),pos_tweets,MyTwitterLabel 

dataset,pos_tweets,MyTwitterLabel = tokenize_prep(tweets_class)
dataset['unigram']
dataset['unigram_stop']
dataset['bigram']
MyTwitterLabel = tweets_class['polarity'].append(pos_tweets['polarity']) #double brakets get into list format
MyTwitterLabel = MyTwitterLabel.reset_index()
MyTwitterLabel = MyTwitterLabel.drop('index',1)

def vectTFIDF(sample_size):
    word_type_list = ['unigram','unigram_stop', 'bigram', 'bigram_stopped']
    for word in word_type_list:
        all_words= dataset[word].tolist()
        count_vec_orig = cv.fit(all_words)
        tfidf_vector1_orig=tfidf.fit(all_words)
        count_vec_fit =cv.transform(all_words)
        tfidf_vector1_fit = tfidf.transform(all_words)
        #count_vec_fit.shape
        MyColumnNames=count_vec_orig.get_feature_names()
        vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames) #creates a DF
        vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity') #adds labels from above function
        #prepare the x value
        X = vectTFIDFOrigDF.values
        y=vectTFIDFOrigDF_labels['polarity'].values
        y = le.fit_transform(y)   #this encodes as a label so naive bayes function can find label and predict

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=sample_size, random_state=1)
        mnb =MultinomialNB()
        mnb.fit(X_train,y_train)
        mnb_pred = mnb.predict(X_test)
        mat = confusion_matrix(y_test, mnb_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)#,
           # xticklabels=train.target_names, yticklabels=train.target_names)
        plt.xlabel('true label')
        plt.ylabel('predicted label') 
        plt.show()
        
        print(word,' mnb accuracy score is: ',accuracy_score(y_true=y_test, y_pred=mnb_pred))
        
        bnb = BernoulliNB()
        bnb.fit(X_train,y_train)
        bnb_pred = bnb.predict(X_test)
        accuracy_score(y_true=y_test, y_pred=bnb_pred)
 
        mat = confusion_matrix(y_test, bnb_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)#,
           # xticklabels=train.target_names, yticklabels=train.target_names)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
        print(word,' bnb accuracy score is: ',accuracy_score(y_true=y_test, y_pred=bnb_pred))
    return(vectTFIDFOrigDF,vectTFIDFOrigDF_labels)

vectTFIDFOrigDF,vectTFIDFOrigDF_labels = vectTFIDF(.10)


#ok looks like the bnb naive base on stopped words has the highest accuracy and it looks like it has the best distribution
# of different tweets, whereas the MNB unigram classified too many as negative. 
###now fit the model:
model = all_words= dataset['unigram_stop'].tolist()
model_tfidf_vector1_orig=tfidf.fit(model)
count_vec_orig = cv.fit(model)
#tfidf_vector1_orig=tfidf.fit(all_words)
#count_vec_fit =cv.transform(all_words)
MyColumnNames_model=count_vec_orig.get_feature_names()
tfidf_vector1_fit = tfidf.transform(model)

vectTFIDFOrigDF_model = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames_model) #creates a DF
vectTFIDFOrigDF_labels_model = add_labels(vectTFIDFOrigDF_model, 'polarity') #adds labels from above function
        #prepare the x value
X = vectTFIDFOrigDF_model.values
y=vectTFIDFOrigDF_labels_model['polarity'].values
y = le.fit_transform(y)   #this encodes as a label so naive bayes function can find label and predict

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.10, random_state=1)

bnb = BernoulliNB()
bnb.fit(X_train,y_train)
bnb_pred = bnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=bnb_pred)

###we have our model. Ok now onto the main event, applying tweets to sentiment model

#keyword = ['"credit card securitization"','"Lease securitization"','"auto loan securitization"','"Student loan securitization"',
#           '"Asset-backed securitization"','"asset-backed securitization"','"subprime auto"','"auto loan ABS"','"student loan ABS"','"car loan ABS"','"aircraft lease securitization"',
#           '"aircraft lease ABS"','"collateralized loan obligations"','"leveraged loans"','"CLOs"','"leveraged finance"']

#keyword = ['"residential mortgage backed securities"','"rmbs"','"residential mortgage-backed securities"','"prime jumbo"',
#           '"prime mortgages"']

#keyword = ['"commercial mortgage-backed"','"commercial mortgage backed"','"industrial real estate"','"RevPAR"','CMBX',
#           '"multifamily rents"','"retail rents"','"office rents"','"retail vacancy"','"retial vacancies"','"office vacancy"',
#           '"office vacancies"']
# GOT Isn't working anymore:
#sf1 = tweet_by_keyword("06/01/2018","06/01/2019",True) #this uses the first keyword list, 
#sf2 = tweet_by_keyword("06/01/2018","06/01/2019",True) #this uses the second keyword list
#sf3 = tweet_by_keyword("06/01/2018","06/01/2019",True) #this uses the third keyword list

#ok this equation will take pulled tweets and clean for the model:
#tweets_combined = sf1.append(sf2)
#tweets_combined =tweets_combined.append(sf3)
file_path = "C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentimentsf_tweets_all2018_SF_v2.xlsx"
sf_tweets = pd.read_excel(file_path)

def sentiment_model(tweets,version):
    #remember we need to use unigram_stopped to show run through the sentiment model
    tweets['unigram_stop']=tweets['text'].map(lambda s:preprocess_stopped(s))
    tweets['date']=tweets['date'].dt.tz_localize(None)
    #file_path = "C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/"
    #save down a version of the file to disk. As a backup
    #sf_tweets.to_excel(file_path+'sf_tweets_all'+version+'.xlsx',index=False, encoding='utf-8')
    #begin the preprocess for the NB model. get into a list and then use a Term Frequency IDF model to 'count' words
    all_words= tweets['unigram_stop'].tolist()
    count_vec_orig = cv.fit(all_words)
    tfidf_vector1_orig=tfidf.fit(all_words)
    tfidf_vector1_fit = tfidf.transform(all_words)
    MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names
    #next get into a datafrem
    vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
    vectTFIDFOrigDF = pd.concat([vectTFIDFOrigDF_model,vectTFIDFOrigDF], axis = 0,sort=False) 
    #this effectively "fits" our data into the model, 
    vectTFIDFOrigDF = vectTFIDFOrigDF.iloc[len(vectTFIDFOrigDF_model):,0:len(vectTFIDFOrigDF_model.columns)]
    #replace all na values with zeros:
    vectTFIDFOrigDF = vectTFIDFOrigDF.fillna(0)
    return vectTFIDFOrigDF, tweets


#file_path = "C:/Users/andy_white/OneDrive - S^&P Global/Desktop/Projects/Python"
#sf_tweets = pd.read_excel(file_path+'sf_tweets_all.xlsx')
sf_tweets['unigram_stop']=sf_tweets['text'].map(lambda s:preprocess_stopped(s))

TFIDF_18, sf_tweets = sentiment_model(sf_tweets,"2018_SF_v2")

#combine the dataframes:
#sf_tweets = sf_tweets.append(tweets_combined)
vectTFIDFOrigDF = vectTFIDFOrigDF.append(TFIDF_18)
#vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity')
sf_tweets_test = TFIDF_18.values
bnb_june18 = bnb.predict(sf_tweets_test)

########################### Had to update - getoldtweets stopped working, so i read in the unigram split file, but still need to vetorize


#add back into the dataframe:
sf_tweets['pred_polarity'] = bnb_june18
###need to convert to the proper scale -1 is negative tweet, 0 is nuetral, etc
sf_tweets.loc[sf_tweets['pred_polarity']==0,'pred_polarity']=-1
sf_tweets.loc[sf_tweets['pred_polarity']==1,'pred_polarity']=0
sf_tweets.loc[sf_tweets['pred_polarity']==2,'pred_polarity']=1

#save this down so not to have to run the code over and over
sf_tweets.to_excel('C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/tweets_senti_10.30.xlsx',index=False, encoding='utf-8')

# 'SM' or bimonthly looks promising. Also adding the agg to have two columns, one with mean, one with count
freq_wanted ='15D'
new_df = sf_tweets.groupby(pd.Grouper(key='date',freq = freq_wanted))['pred_polarity'].agg(['mean','count','std'])
counts_p = sf_tweets.groupby(pd.Grouper(key='date',freq = freq_wanted))['pred_polarity'].value_counts().unstack(-1, fill_value=0)
new_df['neg']=counts_p.iloc[:,0]
new_df['neut']=counts_p.iloc[:,1]
new_df['pos']=counts_p.iloc[:,2]

#lets try to proportion of negative tweets:
new_df['neg_portion']=counts_p.iloc[:,0]/(counts_p.iloc[:,0]+counts_p.iloc[:,1]+counts_p.iloc[:,2])
new_df
#now to add a confidence interval:
z_score = 1.95996   # corresponds to 95% confi interval
new_df['margin_error']= z_score*((new_df['neg_portion']*(1-new_df['neg_portion']))/
                                 ((new_df['neg']+new_df['neut']+new_df['pos'])))**.5
new_df['high']=new_df['neg_portion']+new_df['margin_error']
new_df['low']=new_df['neg_portion']-new_df['margin_error']

senti_graph = go.Figure([go.Scatter(
    name='Sentiment',
    x=new_df['neg_portion'],
    y=new_df.index,
    line=dict(color='rgb(0,100,80)'),
    mode='lines'
    ),
    go.Scatter(
        name='Upper Bound',
        x=new_df['high'],
        y=new_df.index, # upper, then lower reversed
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
        ),
    go.Scatter(
        name='Lower Bound',
        x=new_df['low'],
        y=new_df.index, # upper, then lower reversed
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
        )])

senti_graph.update_layout(
    yaxis_title='Sentiment',
    title='Sentiment With Margin Of Error',
    hovermode="x"
)
senti_graph.write_html('sentiment_confi')

new_df_count = sf_tweets.groupby(pd.Grouper(key='date',freq = 'MS'))['pred_polarity'].count()
new_df.plot()
#new_df = pd.DataFrame({'date':sf_tweets['date'],})
#accuracy_score(y_true=y_test, y_pred=bnb_pred)
oas_info = pd.read_excel("C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/OASpreadGraphExport_MBS.xlsx",
                         header=8)
oas_info = oas_info[oas_info['Option Adjusted Spread'].notna()]
oas_info['Effective date ']=pd.to_datetime(oas_info['Effective date '])
oas_info.set_index('Effective date ',inplace=True)
oas_info['Option Adjusted Spread'] = oas_info['Option Adjusted Spread'].astype(int)
oas_info = oas_info[oas_info.index > "2018-06-01"]

djia = pd.read_excel("C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/DJIA_Last10.xlsx",
                         header=6,usecols="A:B")
djia.head()
djia = djia[djia['Dow Jones Industrial Average'].notna()]
djia['Effective date ']=pd.to_datetime(djia['Effective date '])
djia.set_index('Effective date ',inplace=True)
djia['Dow Jones Industrial Average'] = djia['Dow Jones Industrial Average'].astype(int)
djia = djia[djia.index > "2018-06-01"]

new_df = new_df[new_df.index > "2018-06-01"]

sp500 = pd.read_excel("C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/SP500.xls",
                         usecols="A:B")
sp500 = sp500[sp500['S&P 500'].notna()]
sp500['Effective date ']=pd.to_datetime(sp500['Effective date '])
sp500.set_index('Effective date ',inplace=True)
sp500['S&P 500'] = sp500['S&P 500'].astype(int)
sp500 = sp500[sp500.index > "2018-06-01"]

clo_spreads = pd.read_excel("C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/Market_Spreads_9.28.xlsx",
                            sheet_name='U.S. CLO (WF)')
clo_spreads = clo_spreads.iloc[0:194,:]
clo_spreads.index = clo_spreads.iloc[:,0]
clo_spreads = clo_spreads[clo_spreads.index > "2018-06-01"]

abs_spreads = pd.read_excel("C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentiment/Market_Spreads_9.28.xlsx",
                            sheet_name='Subprime Auto & More ABS (WF)')
abs_spreads = abs_spreads.iloc[0:194,:]

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=new_df.index, y=new_df[0:], name="Sentiment"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=clo_spreads.index, y=clo_spreads['CLO 2.0 BSL Primary BB'], name="BB Spread"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(   #xaxis=dict(rangeslider=dict(visible=True),
                     #        type="linear"),
    title_text="Sentiment vs. S&P 500"
)


# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="Overall Sentiment Score", secondary_y=False)
fig.update_yaxes(title_text="CLO BB Spreads", secondary_y=True)

fig.show()

fig.write_html('1_month_cloBB_spread.html')

#####Now onto the topic modeling
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

number_topics = 1
number_words = 15

# Create and fit the LDA model
def topic_model(data):
    number_topics = 1
    number_words = 10
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(data)
    return lda

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)

lda.fit(count_data)

sf_tweets['text'] = sf_tweets['text'].astype(str)
new_df2 = pd.DataFrame(sf_tweets.groupby(pd.Grouper(key='date',freq = '15D'))['text'].apply(' '.join))
new_df2['text'] = new_df2.iloc[:,0]

count_data = cv.fit_transform(new_df2['text'])


new_df2['lda']=new_df2['text'].map(lambda s:topic_model(s))

#stack the series onto each other, in order to fit into the plotly chart
oas_info = pd.concat([oas_info['Option Adjusted Spread'],new_df[0:]],axis=0)
oas_info =pd.DataFrame({'info':oas_info})
oas_info.loc[oas_info['info']>2,'dtype']='OAS'
oas_info.loc[oas_info['info']<2,'dtype']='Senti'


import plotly.express as px


fig = px.line(oas_info, x=oas_info.index, y="info",color='dtype')
fig.show()




##############################################################################################
########################################stop here #############################################
################################################################################################
keyword = ['"credit card securitization"','"Lease securitization"','"auto loan securitization"','"Student loan securitization"',
           '"Asset-backed securitization"','"asset-backed securitization"','"subprime auto"','"auto loan ABS"','"student loan ABS"','"car loan ABS"','"aircraft lease securitization"',
           '"aircraft lease ABS"','"collateralized loan obligations"','"leveraged loans"','"CLOs"','"leveraged finance"']
keyword = ['"residential mortgage backed securities"','"rmbs"','"residential mortgage-backed securities"','"prime jumbo"',
           '"prime mortgages"']
#notice that you can set a query search or a username search - you can modify however you want. I searched any metion of CMBS in the last 15 days
tweetCriteria = (got.manager.
                 TweetCriteria().
                #setUsername("TreppWire")
                 setQuerySearch('"auto loan securitization"')
                 .setTopTweets(True)
                 .setSince(str(date.today() - timedelta(days = 100)))
                 .setUntil(str(date.today()))
                 .setLang('en')
                 .setMaxTweets(0)) # 0 retrieves all

tweet = got.manager.TweetManager.getTweets(tweetCriteria)
inspect.getsource(got.manager)
tweet_texts = [x.text for x in tweet]
#how long are the tweets
print(len(tweet_texts))
#quick look at the tweets pulled:
tweet_texts
'''

id (str)
permalink (str)
username (str)
to (str)
text (str)
date (datetime) in UTC
retweets (int)
favorites (int)
mentions (str)

TweetManager: A manager class to help getting tweets in Tweet's model.
    getTweets (TwitterCriteria): Return the list of tweets retrieved by using an instance of TwitterCriteria.

TwitterCriteria: A collection of search parameters to be used together with TweetManager.
    setUsername (str or iterable): An optional specific username(s) from a twitter account (with or without "@").
    setSince (str. "yyyy-mm-dd"): A lower bound date (UTC) to restrict search.
    setUntil (str. "yyyy-mm-dd"): An upper bound date (not included) to restrict search.
    setQuerySearch (str): A query text to be matched.
    setTopTweets (bool): If True only the Top Tweets will be retrieved.
    setNear(str): A reference location area from where tweets were generated.
    setWithin (str): A distance radius from "near" location (e.g. 15mi).
    setMaxTweets (int): The maximum number of tweets to be retrieved. If this number is unsetted or lower than 1 all possible tweets will be retrieved.

'''

########################### the rest of this is to look at the other categories that twitter function can pull
tweet_texts = [x.text for x in tweet]
tweet_date = [x.date for x in tweet]
tweet_hashtag = [x.hashtags for x in tweet]
tweet_retweet = [x.retweets for x in tweet]
tweet_permalink = [x.permalink for x in tweet]
tweet_user = [x.username for x in tweet]
tweet_favs = [x.favorites for x in tweet]
tweet_mentions = [x.mentions for x in tweet]
tweet_id = [x.id for x in tweet]
tweet_to = [x.to for x in tweet]
    


##### Ok now we have a lot of info - will create a function to input to dataframe


"""Functiondoes the following:
    -takes the tweet form and converts to a dataframe
    -Each dataframe has the specifications: text, date, hashtag, retweets, http link, username, favorites, metions, twitter_id, if tweet was @ someone
    - Adds 4 blank columns that will need to be filled out in order to run the naive bayes model
        -Analyst 1, 2, 3 assessment of polarity,
        -A majority tie breaker
        - if all three folks disagree the tiebreaker will go to Nuetral
    - There will be a timestamp down to the second, so each run will be documented
"""


def tweet_to_df(tweet,group):
    #tweet needs to be of GetOldTweets3.models.Tweet.Tweet class to work
    tweet_meta = ['text','date','hashtags','retweets','permalink','username','favorites','mentions','id','to']

    tweet_texts = [x.text for x in tweet]
    df = pd.DataFrame({'text':tweet_texts})
    #the following input columns into the pandas dataframe for later analysis
    df['Sentiment_Analysis_1']=''
    df['Sentiment_Analysis_2']=''
    df['Sentiment_Analysis_3']=''
    df['Majority']=''
    df['date'] = [x.date for x in tweet]
    df['hastag'] = [x.hashtags for x in tweet]
    df['retweet'] = [x.retweets for x in tweet]
    df['permalink'] = [x.permalink for x in tweet]
    df['user'] = [x.username for x in tweet]
    df['favs'] = [x.favorites for x in tweet]
    df['mentions'] = [x.mentions for x in tweet]
    df['id'] = [x.id for x in tweet]
    df['to'] = [x.to for x in tweet]

    file_path = "G:/Twitter_Sentiment/Tweet_data_dump/"
    #file_path='C:/Users/andy_white/Desktop/Projects/Python/twitter_sentiment/data_dump/tweets_'        
    time_stamp = strftime("%Y-%m-%d_%H.%M.%S", gmtime())
    # df.ExcelWriter(open(file_path+time_stamp+'.xlsx','w'),options={'remove_timezone': True})
    #write the dataframe to a CSV to analyze
    df.to_csv(open(file_path+group+time_stamp+'.csv','w',encoding='utf-8'),index=False, encoding='utf-8',line_terminator='\n')
    
    return df

cmbs_tweet_df=tweet_to_df(tweet,'commercial real estate')
# old code related to getoldtweets3 libarry:
#updated this to tweets to keyword search, also top_tweets parameter requires a true false
def tweet_by_keyword(date_start,date_end,top_tweets):
    
    date_start = datetime.datetime.strptime(date_start,"%m/%d/%Y")  #converts user input into a date object
    date_end = datetime.datetime.strptime(date_end,"%m/%d/%Y") #converts user input into a date object
    #assigning variables to blank, so that they can be appended to lists
    delta = timedelta(days=30)
    tweet_texts=[]
    tweet_date = []
    tweet_hashtag = []
    tweet_retweet=[]
    tweet_permalink = []
    tweet_user=[]
    tweet_favs=[]
    tweet_mentions=[]
    tweet_id=[]
    tweet_to=[]
    '''
    The following loop will run through all the twitter user handles provided and then get those tweets
    It also gets the date, hashtag, etc and converts to a dataframe
    '''
    while date_end >=date_start:
        new_start = date_end
        new_start-=delta
        new_end = date_end.strftime("%Y-%m-%d")
        new_start = new_start.strftime("%Y-%m-%d")
        
        for row in keyword:
            tweetCriteria = (got.manager.
                     TweetCriteria().
                    
                    setQuerySearch(str(row))    
                    .setTopTweets(top_tweets) 
                    .setSince(str(new_start))
                     .setUntil(str(new_end))
                     .setLang('en')
                     .setMaxTweets(0))
            tweet = got.manager.TweetManager.getTweets(tweetCriteria)
            tweet_texts.extend([x.text for x in tweet])
            tweet_hashtag.extend([x.hashtags for x in tweet])
            tweet_date.extend([x.date for x in tweet])
            tweet_retweet.extend([x.retweets for x in tweet])
            tweet_permalink.extend([x.permalink for x in tweet])
            tweet_user.extend([x.username for x in tweet])
            tweet_favs.extend([x.favorites for x in tweet])
            tweet_mentions.extend([x.mentions for x in tweet])
            tweet_id.extend([x.id for x in tweet])
            tweet_to.extend([x.to for x in tweet])
        date_end -= delta
    df = pd.DataFrame({'Sentiment_Analysis_1':'','Garbage? 1=Yes,0=No':'',#'Sentiment_Analysis_2':'','Sentiment_Analysis_3':'','Majority':'',
                       'text':tweet_texts,'hashtag':tweet_hashtag,'date':tweet_date,'retweet':tweet_retweet,'permalink':tweet_permalink,
                       'user':tweet_user,'favorites':tweet_favs,'mentions':tweet_mentions,'tweet_id':tweet_id,'tweet_to':tweet_to})
#    df['date']=df['date'].dt.tz_localize(None) #need this to save files to excel. important so you can edit in docshare
    return df


#test it out, remember True/False at the end needs to be updated, I recommend filtering for top tweets
test=tweet_by_keyword("06/01/2019","10/08/2020",True)
#the following list column names, the first couple rows in the df and then the first row
list(test.columns.values)
test.head()
test.iloc[1]
len(test)
##pulling this number of tweets took 8 minutes 45 seconds (yikes!!!), but returned 8668 tweets! So about 1000 tweets = 1 minute
rmbs_tweets= tweet_by_keyword("06/01/2019","07/15/2020",True)
#cmbs_v2 = tweet_by_keyword("06/01/2019","06/20/2020",False)
#clo = tweets_by_practice("CLO","06/01/2019","06/16/2020")
#abs_auto = tweets_by_practice("ABS-Auto","06/01/2019","06/16/2020")
cmbs.head()
cmbs.iloc[1]
len(cmbs)
#len(abs_auto)

#input the dataframe that returns the tweets, and input the number of tweets you want to sample for sentiment analysis. 
#at this juncture i recommend 200. but feel free to do more if you're feeling wild
def save_tweets(DF,tweet_sample_num,version):
    file_path = "G:/Twitter_Sentiment/Tweet_data_dump/"
    time_stamp = strftime("%Y-%m-%d_%H.%M.%S", gmtime())
    #a fail safe in case python shuts down on you when running naive bayes:
    DF.to_excel(file_path+asset_class+'_'+your_name+time_stamp+'.xlsx',index=False, encoding='utf-8')
    file_path2 = "G:/Twitter_Sentiment/Sentiment_scores/"
    #random sample of the dataframe to select n number of rows you specify
    DF.sample(n=tweet_sample_num).to_excel(file_path2+asset_class+'_'+your_name+version+'.xlsx',index=False, encoding='utf-8')
    return str(file_path2+asset_class+'_'+your_name+version+'.xlsx')

#note the return above will return a string of the file path so need to assign function to variable
file_path = save_tweets(test,50,"_test")
#if you want to resample and add more, be sure to update the version

#ok now for the real thing
#abs_autofile_path = save_tweets(abs_auto,200,"v1")
#clo= save_tweets(clo,200,"v1")
file_path = save_tweets(rmbs_tweets,500,'_rmbs_1000_v1')
#there will be a few errors related to the length of the URL. I don't care about those errors

#that worked, now we need to incorporate the join and selection
#file_path = save_tweets(test,50)

#### if you haven't assigned tweets into the excel doc saved down, the following code will not work

## first need to assign pos, nuetral, negativ in the xlsx file

#####Now on to the NLP/ML tasks!!!
''' Now go into the xlsx file and input sentiment scores:
1 = positve
0 = nuetral
-1 = negative
'''

#ok lets try to read in file now because we sampled without replacement, we don't need to remove the rows or anything
#funky from or cmbs object:
#may need to import matplotlib
import matplotlib.pyplot as plt
#had to update the file path to the new file
cmbs_sample = pd.read_excel("G:/Twitter_Sentiment/Sentiment_scores/CMBS_Andy_White_v4.xlsx")
cmbs_sample['Sentiment_Analysis_1'].value_counts().plot.bar()
#list(cmbs_sample.columns.values)
#cmbs_sample['Garbage? 1=Yes,0=No'].sum(skipna=True) #want to confirmt that the garbage tweets will be dropped. We have 536 tweets classified as garbage
cmbs_sample = cmbs_sample[cmbs_sample['Garbage? 1=Yes,0=No']!=1] #confirmed that dropped the tweets taht may confuse our sentiment classifier
#Quick test on our scoring, 

#y=vectTFIDFOrigDF_labels['polarity'].values
#y = le.fit_transform(y)   #this encodes as a label so naive bayes function can find label and predict
file_path = "C:/Users/andy_white/OneDrive - S&P Global/Desktop/Projects/Python/twitter_sentimentsf_tweets_all2018_SF_v2.xlsx"
sf_tweets = pd.read_excel(file_path)

def sentiment_model_with_unigram(tweets):
    tweets['date']=tweets['date'].dt.tz_localize(None)    
    all_words= tweets['unigram_stop'].tolist()
    count_vec_orig = cv.fit(all_words)
    tfidf_vector1_orig=tfidf.fit(all_words)
    tfidf_vector1_fit = tfidf.transform(all_words)
    MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names
    #next get into a datafrem
    vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
    vectTFIDFOrigDF = pd.concat([vectTFIDFOrigDF_model,vectTFIDFOrigDF], axis = 0,sort=False) 
    #this effectively "fits" our data into the model, 
    vectTFIDFOrigDF = vectTFIDFOrigDF.iloc[len(vectTFIDFOrigDF_model):,0:len(vectTFIDFOrigDF_model.columns)]
    #replace all na values with zeros:
    vectTFIDFOrigDF = vectTFIDFOrigDF.fillna(0)
    return vectTFIDFOrigDF, tweets

TFIDF_18, tweets_combined = sentiment_model_with_unigram(sf_tweets)

#combine the dataframes:
sf_tweets1 = sf_tweets.append(tweets_combined)
vectTFIDFOrigDF = vectTFIDFOrigDF.append(TFIDF_18)
#vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity')
sf_tweets_test = vectTFIDFOrigDF.values
sf_tweets_test = TFIDF_18.values
#run through the modelto apply sentiment:
bnb_june19 = bnb.predict(sf_tweets_test)
###ok now just need to unigram stop the test part
sf_tweets['unigram_stop']=sf_tweets['text'].map(lambda s:preprocess_stopped(s))
file_path = "C:/Users/andy_white/Desktop/Projects/Python/twitter_sentiment"
#need to localize the timezone in order to save to excel
sf_tweets['date']=sf_tweets['date'].dt.tz_localize(None)

#this is effectively a save point for all the tweets pulled
sf_tweets.to_excel(file_path+'sf_tweets_all.xlsx',index=False, encoding='utf-8')
#read back in
sf_tweets = pd.read_excel(file_path+'sf_tweets_all.xlsx')

all_words= sf_tweets['unigram_stop'].tolist()
count_vec_orig = cv.fit(all_words)
tfidf_vector1_orig=tfidf.fit(all_words)
#tfidf.vocabulary_

#count_vec_fit =cv.transform(all_words)
tfidf_vector1_fit = tfidf.transform(all_words)
count_vec_fit.shape
MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names

vectCountOrigDF=pd.DataFrame(count_vec_fit.toarray(), columns = MyColumnNames)
#now that we have the df of all words we need to "fit" it back into our model. the model will not be able to handle any exceptiosn

#sf_tweets_test= vectCountOrigDF.values

print(vectCountOrigDF.head(10))
#vectCountOrigDF_labels = add_labels(vectCountOrigDF, 'polarity')
print(vectCountOrigDF_labels.head(10))

vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
#test = ""
#lst = vectTFIDFOrigDF_model.columns.intersection(sorted(lst),key = lst.index)]
vectTFIDFOrigDF = pd.concat([vectTFIDFOrigDF_model,vectTFIDFOrigDF], axis = 0,sort=False) 
vectTFIDFOrigDF = vectTFIDFOrigDF.iloc[len(vectTFIDFOrigDF_model):,0:len(vectTFIDFOrigDF_model.columns)]
#replace all na values with zeros:
vectTFIDFOrigDF = vectTFIDFOrigDF.fillna(0)

print(vectTFIDFOrigDF.head(10))



test_df = pd.DataFrame({'A':[1,2,3,4],'B':[2,3,4,5],'C':[3,4,5,6]})
test_df.values

test_df = pd.DataFrame({'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9],
                   'D':[1,3,5],
                   'E':[5,3,6],
                   'F':[7,4,3]})

lst = ['B','R','A']

print(test_df.columns.intersection(lst))
#Index(['A', 'B'], dtype='object')
data = test_df[test_df.columns.intersection(sorted(lst),key = lst.index)]
data = test_df[sorted(set(test_df) & set(lst), key = lst.index)]
data
df1 = pd.DataFrame({'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9],
                   'D':[1,3,5],
                   'E':[5,3,6],
                   'F':[7,4,3]})

df2 = pd.DataFrame({'B':[1,2,3],
                   'D':[4,5,6],
                   'C':[7,8,9],
                   'F':[1,3,5],
                   'G':[5,3,6],
                   'x':[7,4,3]})

pd.concat([df1,df2], axis = 0,keys=list(df1.columns))

word_type_list = ['unigram','unigram_stop', 'bigram', 'bigram_stopped']
for word in word_type_list:  
    all_words = unigram[word]
    print(all_words)

count_vec_orig = cv.fit(all_words)
tfidf_vector1_orig=tfidf.fit(all_words)
tfidf.vocabulary_
#need to transform so that it knows there are 200 separate documents.
count_vec_fit =cv.transform(all_words)
tfidf_vector1_fit = tfidf.transform(all_words)
count_vec_fit.shape
MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names

vectCountOrigDF=pd.DataFrame(count_vec_fit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDF_labels = add_labels(vectCountOrigDF, 'polarity')
print(vectCountOrigDF_labels.head(10))

vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity')

X = vectTFIDFOrigDF.values

import random
####Now to run the naive bayes algo on our featuresets to get an accuracy score
def NB_class_all (x1,x2,x3,x4,x5,test_size):
    num = random.sample(range(0,len(unigram)),int(round(test_size*len(unigram),0)))
    train_set, test_set = x1[:num], x1[num:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    train_set2, test_set2 = x2[:num], x2[num:]
    classifier2 = nltk.NaiveBayesClassifier.train(train_set2)
    train_set3, test_set3 = x3[:num], x3[num:]
    classifier3 = nltk.NaiveBayesClassifier.train(train_set3)
    train_set4, test_set4 = x4[:num], x4[num:]
    classifier4 = nltk.NaiveBayesClassifier.train(train_set4)
    train_set5, test_set5 = x5[:num], x5[num:]
    classifier5 = nltk.NaiveBayesClassifier.train(train_set5)
    print('Accuracy of unigram '+str(nltk.classify.accuracy(classifier, test_set))+'%')
    print('Accuracy of stopped unigram '+str(nltk.classify.accuracy(classifier2, test_set2))+'%')
    print('Accuracy of bigram '+str(nltk.classify.accuracy(classifier3, test_set3))+'%')
    print('Accuracy of stopped bigram '+str(nltk.classify.accuracy(classifier4, test_set4))+'%')
    print('Accuracy of POS tags '+str(nltk.classify.accuracy(classifier5, test_set5))+'%')
    return classifier, classifier2, classifier3,classifier4,classifier5

### ok  - two lines of code to run your naive bayes classifier 4 different ways! not much to it huh? also notice that I'm assinging a number of variables to a few of the datasets
# the word_features, and word_feature_stop are for later when we apply our model to the dataset
unigram,unigram_stop,bigram,bigram_stop,POS_tags,word_features,word_feature_stop,all_words_list,all_words_stop_list = get_all_WF_unigram(cmbs_sample,2000,2000,400,300)

type(unigram)
#do not change these names ('uni','uni_stop','bi','bistop') - they are used in the next function
uni, uni_stop,  bi,   bi_stop, pos = NB_class_all(unigram,unigram_stop,bigram,bigram_stop,POS_tags,.2)
(unigram[i] for i in num)

bi.show_most_informative_features(30)

uni.classify(unigram[2][0])
uni.classify(unigram[4][0])
uni.classify(unigram[100][0])
cmbs_label = cmbs_sample['polarity']
cmbs_sample2 =cmbs_sample
#ok that works now to do it on the rest of the tweets
featuresets = [(document_features(d,word_features), c) for (d,c) in all_words1]
senti_list = []
for i in range(len(featuresets)):
    senti_list.append(uni.classify(featuresets[i][0]))
df['senti_score']=senti_list
df.loc[df['senti_score']=='neg','polarity']=-1
df.loc[df['senti_score']=='nuet','polarity']=0
df.loc[df['senti_score']=='pos','polarity']=1
#now that we have the classifier, with an Ok accuracy, lets apply the model to our universe, with the highest 
#we will have one last function to take our universe and convert like above

#Ok I want to run some tests to see where the classifer is sucking:


#to apply the stop words to the the text body takes 4 to 5 minutes, based on 8668 tweets. lots of time to remove stop words
#cmbs['unigram_stop']=cmbs['text'].map(lambda s:preprocess_stopped(s))

#was originally going to combine into one large function, but the run-time is far too long with preprocessor. 
#run it once and then tweak the "the_classifier" function with whatever specifications you want
def classifier_preprocess(df):
    df['unigram'] = df['text'].map(lambda s:preprocess_unigram(s))
    all_words = df['unigram'].tolist()
    df['unigram_stop'] = df['text'].map(lambda s:preprocess_stopped(s))
    all_words_stop = df['unigram_stop'].tolist()
    all_words_list = [item for sublist in all_words for item in sublist]
    all_words_stop_list = [item for sublist in all_words_stop for item in sublist]

    all_words1 =[]
    for sent in all_words:
        all_words1.append((sent,''))     
    all_words_stop1 =[]
    for sent in all_words_stop:
        all_words_stop1.append((sent,''))         
    return all_words1, all_words_stop1, all_words_list,all_words_stop_list
    '''
    the following takes the user specified classifier task, unigram, unigram_stop, etc, and then classifies the entire
    universe provided based on the classifier strategy built above. 
    '''
all_words1, all_words_stop1, all_words_list,all_words_stop_list = classifier_preprocess(cmbs_sample2)

#finally  the classifier step, classifies our universe with user specified 
def the_classifier(classifier_input,df):
    if str(classifier_input) == 'unigram':
        featuresets = [(tweet_unigram_features(d,word_features), c) for (d,c) in all_words1] #word features were defined above in our sample
        senti_list = []  #the only way I could find to take the featuresets, and then apply our classifier line to line, to ultimately return a score
        for i in range(len(featuresets)):
            senti_list.append(uni.classify(featuresets[i][0]))
        df['senti_score']=senti_list
        df.loc[df['senti_score']=='neg','polarity']=-1   #the numbers allow for averages, and sums per week. 
        df.loc[df['senti_score']=='nuet','polarity']=0
        df.loc[df['senti_score']=='pos','polarity']=1
        return df
    elif str(classifier_input) == 'unigram_stopped':
        #same as above, but user specifies which featureset as a parameter in the equation above
        featuresets = [(tweet_unigram_features(d,word_features_stop), c) for (d,c) in all_words_stop1]
        
        senti_list = []
        for i in range(len(featuresets)):
            senti_list.append(uni_stop.classify(featuresets[i][0]))
        df['senti_score']=senti_list
        df.loc[df['senti_score']=='neg','polarity']=-1
        df.loc[df['senti_score']=='nuet','polarity']=0
        df.loc[df['senti_score']=='pos','polarity']=1
        return df
    elif str(classifier_input) == 'bigram':
        finder = BigramCollocationFinder.from_words(all_words_list)
        bigram_features = finder.nbest(bigram_measures.chi_sq, bigram_chi_square)
        featuresets = [(bigram_document_features(d,word_features,bigram_features), c) for (d,c) in all_words1]
        senti_list = []
        for i in range(len(featuresets)):
            senti_list.append(bi.classify(featuresets[i][0]))
        df['senti_score']=senti_list
        df.loc[df['senti_score']=='neg','polarity']=-1
        df.loc[df['senti_score']=='nuet','polarity']=0
        df.loc[df['senti_score']=='pos','polarity']=1
        return df
    elif str(classifier_input)=='bigram_stop':
        finder_stop = BigramCollocationFinder.from_words(all_words_stop_list)
        bigram_features_stop = finder_stop.nbest(bigram_measures.chi_sq, bigram_stop_chi_square) #get top 250 bigrams
        featuresets = [(bigram_document_features(d,word_features,bigram_features_stop), c) for (d,c) in all_words_stop1]
        senti_list = []
        for i in range(len(featuresets)):
            senti_list.append(bi_stop.classify(featuresets[i][0]))
        
        df['senti_score']=senti_list
        df.loc[df['senti_score']=='neg','polarity']=-1
        df.loc[df['senti_score']=='nuet','polarity']=0
        df.loc[df['senti_score']=='pos','polarity']=1
        return df
    elif str(classifier_input)=='POS':
        featuresets = [(POS_features(d,word_features), c) for (d, c) in all_words1]
        senti_list = []
        for i in range(len(featuresets)):
            senti_list.append(bi_stop.classify(featuresets[i][0]))
        
        df['senti_score']=senti_list
        df.loc[df['senti_score']=='neg','polarity']=-1
        df.loc[df['senti_score']=='nuet','polarity']=0
        df.loc[df['senti_score']=='pos','polarity']=1
        return df
    else:
        print('you need to select unigram, unigram_stop, bigram, bigram_stop, or POS. And no this is not a POS function!!!')
    return df  
        
#str(uni)
#iteration slows this down - but should take about 20 seconds per 10,000 rows of data
'''
Select the following for the first parameter:
    'unigram'
    'unigram_stopped'
    'bigram'
    'bigram_stopped'
    'POS'
'''
test1= the_classifier('unigram',cmbs_sample)
test1['polarity']
cmbs_sample['Sentiment_Analysis_1']
from sklearn.metrics import confusion_matrix
accuracy_score(y_true=cmbs_sample['Sentiment_Analysis_1'], y_pred=test1['polarity'])
confusion_matrix(test1['polarity'],cmbs_sample['Sentiment_Analysis_1'])

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
target=vectTFIDFStandDFLie['Speech LABEL'].values
mat = confusion_matrix(y_test, mnb_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)#,
           # xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
list(test1.columns.values)
#test['predicted_senti']=df['featuresets'].map(lambda s:uni.classify(s))
#this converts to get a weekly sentiment score:
new_df = test1.groupby(pd.Grouper(key='date',freq = 'W-SUN'))['polarity'].mean()

new_df.plot()




#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
##### the rest of this code is how I built up these functions. It's a little messy, but it helps you understand what each
#### of these functions are really doing. feel free to execute through

def POS_features(document,word_features):
	document_words = set(document)   
	tagged_words = nltk.pos_tag(document) #take words and run POS tagger
	features = {}
	for word in word_features:
		features['contains({})'.format(word)] = (word in document_words)
	numNoun = 0
	numVerb = 0
	numAdj = 0
	numAdverb = 0
	for (word, tag) in tagged_words:
		if tag.startswith('N'): numNoun += 1
		if tag.startswith('V'): numVerb += 1
		if tag.startswith('J'): numAdj += 1
		if tag.startswith('R'): numAdverb += 1
	features['nouns'] = numNoun
	features['verbs'] = numVerb
	features['adjectives'] = numAdj
	features['adverbs'] = numAdverb
	return features
def get_all_WF_unigram(df,num_of_word_items,num_of_word_items2,bigram_chi_square,bigram_stop_chi_square):
    '''
    Need to tokenize, so the "df['unigram']=df['text'].map(lambda s:preprocess_unigram(s))" is basically applying our
    unigram preprocess function from above to tokenize, and the preprocess_stopped then removes stop words
    '''
    df.sample(frac=1)   #randomize rows in the dataframe
    df.loc[df['Sentiment_Analysis_1']==-1,'polarity']='neg'
    df.loc[df['Sentiment_Analysis_1']==0,'polarity']='nuet'
    df.loc[df['Sentiment_Analysis_1']==1,'polarity']='pos'
    df['unigram']=df['text'].map(lambda s:preprocess_unigram(s))  #this tokenizes by unigram
    df['unigram_stop']=df['text'].map(lambda s:preprocess_stopped(s)) #tokenizes by unigram with stop words removed
    df['bigram']=df['unigram'].map(lambda s:list(nltk.bigrams(s))) #create bigrams from our unigram tokens
    df['bigram_stopped']=df['unigram_stop'].map(lambda s:list(nltk.bigrams(s))) #creates bigrams from our unigram with stop words removed
    polarity = df['polarity'].tolist()   #convers column to list
    all_words = df['unigram'].tolist()   #converts unigram to list
    all_words_list = [item for sublist in all_words for item in sublist]   #gets a list of all the words from all the tweets
    x=0   #for the iteration below - but takes eatch tokenized list and assigns the polarity measure to it
    all_words1 =[]
    for sent in all_words:
        all_words1.append((sent,polarity[x]))
        x=x+1
    all_words_freq = nltk.FreqDist(all_words_list)   #creates a frequency distribution from all words in all tweets
    word_items = all_words_freq.most_common(num_of_word_items)   #take the top user specified words as 
    word_features = [word for (word, freq) in word_items]   #assigns word_features  - featuresets are assigned below
    all_words_stop = df['unigram_stop'].tolist()   #convert to list - but this makes a list in a list
    all_words_stop_list = [item for sublist in all_words_stop for item in sublist] #iterate through so it's just one long list
    x=0  #same as unigrams but with unigrams with stop words removed
    all_words_stop1 =[]
    for sent in all_words_stop:
        all_words_stop1.append((sent,polarity[x]))
        x=x+1
    all_words_stop_freq = nltk.FreqDist(all_words_stop_list) #get the frequency distribution
    word_items_stop = all_words_stop_freq.most_common(num_of_word_items2)   #take the top 1500 words as the real test will be much larger
    word_features_stop = [word for (word, freq) in word_items_stop]
    featuresets = [(tweet_unigram_features(d,word_features), c) for (d,c) in all_words1]
    featuresets_stop = [(tweet_unigram_features(d,word_features_stop), c) for (d,c) in all_words_stop1]
    '''
    now on to the bigram piece:
    '''
    #bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_words_list)
    finder_stop = BigramCollocationFinder.from_words(all_words_stop_list)
    bigram_features = finder.nbest(bigram_measures.chi_sq, bigram_chi_square)
    bigram_features_stop = finder_stop.nbest(bigram_measures.chi_sq, bigram_stop_chi_square) #get top 250 bigrams
    
    bigram_featuresets = [(bigram_document_features(d,word_features,bigram_features), c) for (d,c) in all_words1]
    bigram_featuresets_stop = [(bigram_document_features(d,word_features_stop,bigram_features_stop), c) for (d,c) in all_words_stop1]
    POS_featuresets = [(POS_features(d,word_features), c) for (d, c) in all_words1]
    print('You choose '+str(num_of_word_items)+' words out of a possible '+str(len(all_words_freq))+ ' or '+str(num_of_word_items/len(all_words_freq))+'%')
    print('You choose '+str(num_of_word_items2)+' words out of a possible '+str(len(all_words_stop_freq))+ ' or '+str(num_of_word_items2/len(all_words_stop_freq))+'%')
    print('You choose '+str(bigram_chi_square)+' bigrams out of a possible '+str(len(bigram_featuresets[0][0]))+ ' or '+str(bigram_chi_square/len(bigram_featuresets[0][0]))+'%')
    print('You choose '+str(bigram_stop_chi_square)+' bigrams out of a possible '+str(len(bigram_featuresets_stop[0][0]))+ ' or '+str(bigram_stop_chi_square/len(bigram_featuresets_stop[0][0]))+'%')
    return featuresets,featuresets_stop,bigram_featuresets,bigram_featuresets_stop,POS_featuresets,word_features,word_features_stop,all_words_list,all_words_stop_list


#practice pulling from multiple accounts at once from a specified worksheet:        
twt_df = pd.read_excel("G:/Twitter_Sentiment/SF_twitter_accs.xlsx")
twt_df = twt_df['CMBS'].dropna()
test=twt_df.iloc[:2]  #pull first two rows of the twt_df

'''
It's likely that when I try to feed the tweets into getoldtweets function, that it needs to be in the form of a list and not a
pandas df, I'm going to convert our DF into a list and then try to run a loop function that will loop through each twitter account
Note that I will be using the test variable which only has two rows of data - this was a quicker option instead of waiting it to loop through
12 accounts
'''
test = test.tolist()
for row in test:
    print(row)
tweet_texts=[]
for row in test:
    tweetCriteria = (got.manager.
                 TweetCriteria().
                setUsername(str(row))
                # setQuerySearch('CMBS')    
                 .setSince(str(date.today() - timedelta(days = 10)))
                 .setUntil(str(date.today()))
                 .setLang('en')
                 .setMaxTweets(0))
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    tweet_texts.extend([x.text for x in tweet])  #append was used to append list, found that extend can append lists together




#I want to be able to allow the user to modify the date by selecting a date in the "mm/dd/yyyy", 
#so that a user can provide the beginning and ending date

date_time_start = '06/05/2020'
date_time_end = "06/12/2020"
date_time_obj_start = datetime.datetime.strptime(date_time_start,"%m/%d/%Y")
date_time_obj_start.date()
date_time_obj_end = datetime.datetime.strptime(date_time_end,"%m/%d/%Y")
date_time_obj_end-date_time_obj_start

date_time_obj
date.today()
#bringing everything together:

len(all_words_stop)
all_words_stop1[:10]
all_words_stop_freq = nltk.FreqDist(all_words_stop_list) #get the frequency distribution
len(all_words_stop_freq)
word_items_stop = all_words_stop_freq.most_common(125)   #take the top 1500 words as the real test will be much larger
word_features_stop = [word for (word, freq) in word_items_stop]

####here's how I built the functions above:
test = cmbs_sample
test.loc[test['Sentiment_Analysis_1']==-1,'polarity']='neg'
test.loc[test['Sentiment_Analysis_1']==0,'polarity']='nuet'
test.loc[test['Sentiment_Analysis_1']==1,'polarity']='pos'
#the following code breacks up our tweets into unigrams and bigrams, and also gets a list of all words in unigram and bigram format
test['unigram']=test['text'].map(lambda s:preprocess_unigram(s))
test['unigram_stop']=test['text'].map(lambda s:preprocess_stopped(s))
test['bigram']=test['unigram'].map(lambda s:list(nltk.bigrams(s)))
test['bigram_stopped']=test['unigram_stop'].map(lambda s:list(nltk.bigrams(s)))

#test['all_words']
#create the all words list:
polarity = test['polarity'].tolist()

all_words = test['unigram'].tolist()
all_words_list = [item for sublist in all_words for item in sublist]
len(all_words)

#preprocess for the document features and tagging
# all_words = [(all_words,polarity)]
#all_words = [(sent, cat) for sent in all_words for cat in polarity]
#all_words1 = all_words[:200]
x=0
all_words1 =[]
for sent in all_words:
    all_words1.append((sent,polarity[x]))
    x=x+1



def tweets_by_practice(group,date_start,date_end):
    '''
    date must be entered in mm/dd/yyyy format

    '''
    #twt_df = pd.read_excel("B:/DocShare/BC0b3e957c8aa9ce63_0b3e957c87a969f8_3/DocShare/North America/_SF_General Use-Analytical/R Project/Twitter Sentiment/SF_twitter_accs.xlsx")
    #had to change the filepath as it wasn't working for most folks:
    twt_df = pd.read_excel("G:/Twitter_Sentiment/SF_twitter_accs.xlsx")
    col = twt_df[group].dropna()
    date_start = datetime.datetime.strptime(date_start,"%m/%d/%Y")  #converts user input into a date object
    date_end = datetime.datetime.strptime(date_end,"%m/%d/%Y") #converts user input into a date object
    #assigning variables to blank, so that they can be appended to lists
    tweet_texts=[]
    tweet_date = []
    tweet_hashtag = []
    tweet_retweet=[]
    tweet_permalink = []
    tweet_user=[]
    tweet_favs=[]
    tweet_mentions=[]
    tweet_id=[]
    tweet_to=[]
    '''
    The following loop will run through all the twitter user handles provided and then get those tweets
    It also gets the date, hashtag, etc and converts to a dataframe
    '''
    for row in col:
        tweetCriteria = (got.manager.
                 TweetCriteria().
                setUsername(str(row))
                # setQuerySearch('CMBS')    
                 .setSince(str(date_start.date()))
                 .setUntil(str(date_end.date()))
                 .setLang('en')
                 .setMaxTweets(0))
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)
        tweet_texts.extend([x.text for x in tweet])
        tweet_hashtag.extend([x.hashtags for x in tweet])
        tweet_date.extend([x.date for x in tweet])
        tweet_retweet.extend([x.retweets for x in tweet])
        tweet_permalink.extend([x.permalink for x in tweet])
        tweet_user.extend([x.username for x in tweet])
        tweet_favs.extend([x.favorites for x in tweet])
        tweet_mentions.extend([x.mentions for x in tweet])
        tweet_id.extend([x.id for x in tweet])
        tweet_to.extend([x.to for x in tweet])
    df = pd.DataFrame({'Sentiment_Analysis_1':'','Sentiment_Analysis_2':'','Sentiment_Analysis_3':'','Majority':'',
                       'text':tweet_texts,'hashtag':tweet_hashtag,'date':tweet_date,'retweet':tweet_retweet,'permalink':tweet_permalink,
                       'user':tweet_user,'favorites':tweet_favs,'mentions':tweet_mentions,'tweet_id':tweet_id,'tweet_to':tweet_to})
    df['date']=df['date'].dt.tz_localize(None) #need this to save files to excel. important so you can edit in docshare
    return df


#remember, update the first parameter to the tweets you want to pull "RMBS" or "ABS". As a reminder the column names are:
twt_df = pd.read_excel("G:/Twitter_Sentiment/SF_twitter_accs.xlsx")
for col in twt_df.columns:
    print(col)
#ok this gets into the form we need to understand featuresets
all_words_freq = nltk.FreqDist(all_words_list)
len(all_words_list)
word_items = all_words_freq.most_common(500)   #take the top 1500 words as the real test will be much larger
word_features = [word for (word, freq) in word_items]
word_features[:10]

#now for the stopped words
all_words_stop = test['unigram_stop'].tolist()   #convert to list - but this makes a list in a list
len(all_words_stop)
all_words_stop_list = [item for sublist in all_words_stop for item in sublist] #iterate through so it's just one long list

x=0
all_words_stop1 =[]
for sent in all_words_stop:
    all_words_stop1.append((sent,polarity[x]))
    x=x+1

len(all_words_stop)
all_words_stop1[:10]
all_words_stop_freq = nltk.FreqDist(all_words_stop_list) #get the frequency distribution
len(all_words_stop_freq)
word_items_stop = all_words_stop_freq.most_common(125)   #take the top 1500 words as the real test will be much larger
word_features_stop = [word for (word, freq) in word_items_stop]
word_features_stop[:50]

def tweet_unigram_features(tweets, word_features):
	tweet_words = set(tweets)
	features = {}
	for word in word_features:
		features['V_{}'.format(word)] = (word in tweet_words) # v for vocabulary and 
	return features

#last variable is our tweets with unigram
featuresets = [(document_features(d,word_features), c) for (d,c) in all_words1]
featuresets = [(document_features(d,word_features_stop), c) for (d,c) in all_words_stop1]
#naive bayes
train_set, test_set = featuresets[:30], featuresets[30:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
#ok - maxing out around 60% accuracy. not great, but lets see if bigrams are more telling
#if not we will leverage a twitter sentiment db to help juice those numbers
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder_stop = BigramCollocationFinder.from_words(all_words_stop_list)
len(bigram_featuresets_stop[0][0])
bigram_features_stop = finder_stop.nbest(bigram_measures.chi_sq, 350) #get top 250 bigrams
bigram_features_stop[:50]
len(bigram_features_stop)
len(finder_stop)
bigram_featuresets_stop = [(bigram_document_features(d,word_features_stop,bigram_features_stop), c) for (d,c) in all_words_stop1]
print(bigram_featuresets[0][0])

train_set, test_set = bigram_featuresets_stop[:30], bigram_featuresets_stop[30:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(30)

###Now on to the POS tagging
sent = ['Arthur','carefully','rode','the','brown','horse','around','the','castle']
print(sent)
print(nltk.pos_tag(sent))
#NNP proper nown, VBD - verb, DT -determiner - YIKES GRAMMAR
def POS_features(document,word_features):
	document_words = set(document)   
	tagged_words = nltk.pos_tag(document) #take words and run POS tagger
	features = {}
	for word in word_features:
		features['contains({})'.format(word)] = (word in document_words)
	numNoun = 0
	numVerb = 0
	numAdj = 0
	numAdverb = 0
	for (word, tag) in tagged_words:
		if tag.startswith('N'): numNoun += 1
		if tag.startswith('V'): numVerb += 1
		if tag.startswith('J'): numAdj += 1
		if tag.startswith('R'): numAdverb += 1
	features['nouns'] = numNoun
	features['verbs'] = numVerb
	features['adjectives'] = numAdj
	features['adverbs'] = numAdverb
	return features

#Try out the POS features.
POS_featuresets = [(POS_features(d,word_features), c) for (d, c) in all_words1]
# number of features for document 0
len(POS_featuresets[0][0].keys())
POS_featuresets[:10]

print(all_words1[0])
# the pos tag features for this sentence
print('num nouns', POS_featuresets[0][0]['nouns'])
print('num verbs', POS_featuresets[0][0]['verbs'])
print('num adjectives', POS_featuresets[0][0]['adjectives'])
print('num adverbs', POS_featuresets[0][0]['adverbs'])

#Now split into training and test and rerun the classifier.
train_set, test_set = POS_featuresets[30:], POS_featuresets[:30]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


nltk.regexp_tokenize(test_sent, tweet_pattern)
#looks like that did pretty well. now lets try bigrams and POS tags 
unigram_test = nltk.regexp_tokenize(test_sent, tweet_pattern)

### I Will take two bigram approaches, a straight bigram appraoch and find if top bigrams are present
sentbigrams = list(nltk.bigrams(unigram_test))
sentbigrams

#for this part, will need to have the bigram list first and 
finder = BigramCollocationFinder.from_words(unigram_test)
finder


#lets try some text mining type stuff, with count vectorizing etc. didn't work so well for accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.model_selection import train_test_split
import copy
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

url = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"
more_tweets = pd.read_csv(url,index_col=0)
print(more_tweets.head())

pos_tweets = more_tweets[more_tweets['Sentiment']=="positive"]
pos_tweets = pos_tweets.sample(n=400,random_state=1)


pos_tweets.loc[pos_tweets['Sentiment']=="positive",'polarity']='pos'
#df.loc[df['Sentiment_Analysis_1']==0,'polarity']='nuet'
#df.loc[df['Sentiment_Analysis_1']==1,'polarity']='pos'
pos_tweets['unigram']=pos_tweets['TweetText'].map(lambda s:preprocess_unigram(s))  #this tokenizes by unigram
pos_tweets['unigram_stop']=pos_tweets['TweetText'].map(lambda s:preprocess_stopped(s)) #tokenizes by unigram with stop words removed
pos_tweets['bigram']=pos_tweets['unigram'].map(lambda s:list(nltk.bigrams(s))) #create bigrams from our unigram tokens
pos_tweets['bigram_stopped']=pos_tweets['unigram_stop'].map(lambda s:list(nltk.bigrams(s)))

MyTwitterLabel = cmbs_sample['polarity'].append(pos_tweets['polarity']) #double brakets get into list format
MyTwitterLabel = MyTwitterLabel.reset_index()
MyTwitterLabel = MyTwitterLabel.drop('index',1)

def add_labels(df,lab=''):
    tmp = copy.deepcopy(df)
    if lab == "polarity":
        tmp[lab]= MyTwitterLabel
    else:
        print('Use either Lie LABEL or Senti LABEL')
        return()
    return(tmp)

#in order to get word counts, we will be passing a dummy function to the tokenizer, since we have already tokenized
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

cv = CountVectorizer(
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
    )  


samp = cmbs_sample['unigram'].append(pos_tweets['unigram'])
samp = samp.reset_index()
samp = samp.drop('index',1)

all_words = samp['unigram'].tolist()
count_vec_orig = cv.fit(all_words)
tfidf_vector1_orig=tfidf.fit(all_words)
tfidf.vocabulary_
#need to transform so that it knows there are 200 separate documents.
count_vec_fit =cv.transform(all_words)
tfidf_vector1_fit = tfidf.transform(all_words)
count_vec_fit.shape
MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names

vectCountOrigDF=pd.DataFrame(count_vec_fit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDF_labels = add_labels(vectCountOrigDF, 'polarity')
print(vectCountOrigDF_labels.head(10))

vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity')

X = vectTFIDFOrigDF.values
#X = vectCountStandDF.values
#X = vectCountNormOrigDF.values
#X = vectTFIDFOrigDF.values
#X = vectCountOrigDF.values

y=vectTFIDFOrigDF_labels['polarity'].values
y = le.fit_transform(y)   #this encodes as a label so naive bayes function can find label and predict

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1)

clf = GaussianNB()
clf.fit(X_train,y_train)
GaussianNB(priors=None)
y_pred = clf.predict(X_test)
accuracy_score(y_true=y_test, y_pred=y_pred)

mnb =MultinomialNB()
mnb.fit(X_train,y_train)
mnb_pred = mnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=mnb_pred)

bnb = BernoulliNB()
bnb.fit(X_train,y_train)
bnb_pred = bnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=bnb_pred)

mat = confusion_matrix(y_test, mnb_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)#,
           # xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


#lets find accuracy on stopped words
all_words = cmbs_sample['unigram_stop'].tolist()
count_vec_orig = cv.fit(all_words)
tfidf_vector1_orig=tfidf.fit(all_words)
tfidf.vocabulary_
#need to transform so that it knows there are 200 separate documents.
count_vec_fit =cv.transform(all_words)
tfidf_vector1_fit = tfidf.transform(all_words)
count_vec_fit.shape
MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names

vectCountOrigDF=pd.DataFrame(count_vec_fit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDF_labels = add_labels(vectCountOrigDF, 'polarity')
print(vectCountOrigDF_labels.head(10))

vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity')

X = vectTFIDFOrigDF.values
X = vectCountOrigDF.values
#X = vectCountNormOrigDF.values
#X = vectTFIDFOrigDF.values
#X = vectCountOrigDF.values

y=vectTFIDFOrigDF_labels['polarity'].values
y = le.fit_transform(y)   #this encodes as a label so naive bayes function can find label and predict
y = vectCountOrigDF_labels['polarity'].values
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1,stratify=y)

clf = GaussianNB()
clf.fit(X_train,y_train)
GaussianNB(priors=None)
y_pred = clf.predict(X_test)
accuracy_score(y_true=y_test, y_pred=y_pred)

mnb =MultinomialNB()
mnb.fit(X_train,y_train)
mnb_pred = mnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=mnb_pred)

bnb = BernoulliNB()
bnb.fit(X_train,y_train)
bnb_pred = bnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=bnb_pred)

#bigrams and bigram stopped:

all_words = cmbs_sample['bigram'].tolist()
count_vec_orig = cv.fit(all_words)
tfidf_vector1_orig=tfidf.fit(all_words)
tfidf.vocabulary_
#need to transform so that it knows there are 200 separate documents.
count_vec_fit =cv.transform(all_words)
tfidf_vector1_fit = tfidf.transform(all_words)
count_vec_fit.shape
MyColumnNames=count_vec_orig.get_feature_names()  #this will get column names

vectCountOrigDF=pd.DataFrame(count_vec_fit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDF_labels = add_labels(vectCountOrigDF, 'polarity')
print(vectCountOrigDF_labels.head(10))

vectTFIDFOrigDF = pd.DataFrame(tfidf_vector1_fit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDF_labels = add_labels(vectTFIDFOrigDF, 'polarity')

## STANDARDIZED NORMALIZED COUNT VECTORIZER
vectCountNormStandDF = copy.deepcopy(vectCountNormOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountNormStandDF = pd.DataFrame(scaler.fit_transform(vectCountNormStandDF), columns = MyColumnNames)
print(vectCountNormStandDF.head(10))
vectCountNormStandDFLie = add_labels(vectCountNormStandDF, 'polarity')


# STANDARDIZED TFIDF VECTORIZER
vectTFIDFStandDF = copy.deepcopy(vectTFIDFOrigDF)
scaler = preprocessing.MinMaxScaler()
vectTFIDFStandDF = pd.DataFrame(scaler.fit_transform(vectTFIDFStandDF), columns = MyColumnNames)
print(vectTFIDFStandDF.head(10))
vectTFIDFStandDFLie = add_labels(vectTFIDFStandDF, 'polarity')

X = vectTFIDFOrigDF.values
X = vectCountOrigDF.values
X = vectTFIDFStandDF.values
#X = vectTFIDFOrigDF.values
#X = vectCountOrigDF.values

y=vectTFIDFOrigDF_labels['polarity'].values
y = le.fit_transform(y)   #this encodes as a label so naive bayes function can find label and predict
y = vectCountOrigDF_labels['polarity'].values
y = le.fit_transform(y)
y=vectTFIDFStandDFLie['polarity'].values
y= le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=1)


clf = GaussianNB()
clf.fit(X_train,y_train)
GaussianNB(priors=None)
y_pred = clf.predict(X_test)
accuracy_score(y_true=y_test, y_pred=y_pred)

mnb =MultinomialNB()
mnb.fit(X_train,y_train)
mnb_pred = mnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=mnb_pred)

bnb = BernoulliNB()
bnb.fit(X_train,y_train)
bnb_pred = bnb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=bnb_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#arget=vectTFIDFStandDFLie['Speech LABEL'].values
mat = confusion_matrix(y_test, mnb_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)#,
           # xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


#now for bigrams
#We use the chi-squared measure to get bigrams that are informative features.  Note that we don’t need to get the scores 
#of the bigrams, so we use the nbest function which just returns the highest scoring bigrams, using the number specified.
all_words_bigram = test['bigram'].tolist()
all_words_bigram = [item for sublist in all_words_bigram for item in sublist]
#need to use the finder function on our all words list, only concern will be end of one tweet will be bi-gramed to another
finder = BigramCollocationFinder.from_words(all_words_list)
bigram_features = finder.nbest(bigram_measures.chi_sq,500)
bigram_features[:50]

finder_stop = BigramCollocationFinder.from_words(all_words_stop_list)
bigram_features_stop = finder_stop.nbest(bigram_measures.chi_sq,500)
bigram_features_stop[:50]






#####################################################################################################
#####################################################################################################
############################## old code   #######################################################33333
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################


cmbs['date']
#### Once polarity is decided, we will feed 
cmbs_tweet_df.head()
tweet_meta=['text','date','hashtags','retweets','permalink','username','favorites','mentions','id','to']
df = pd.DataFrame({'text':tweet_texts})
    tweet_texts = []
    tweet_date = []
    tweet_hashtag = []
    tweet_retweet = []
    tweet_permalink = []
    tweet_user = []
    tweet_favs = []
    tweet_mentions = []
    tweet_id = []
    tweet_to = []
for c in tweet_meta:
    pd.DataFrame({strg:twt})


#tried to loop through and assign columns to columns in faster way but didn't work
for strg in tweet_meta:
    twt_str =str(strg) #this concatentes the string needed to run the tweet
    twt_data = [x.twt_str for x in tweet]
    if strg == 'text':           
        df=pd.DataFrame({strg:twt_data})
    else:
        print(strg)
df        


tweet_params = ['text','date','hashtags','retweets','permalink','username','favorites','mentions','id','to']
twt_input =[] 
for strg in tweet_params:
    twt_input= 'x.'+strg
    print('x.'+ strg)






len(tweet_texts)
str(date.today() - timedelta(days = 29))

cmbs.insert(loc=1,column="Sentiment_2",value='')
cmbs.insert(loc=2,column="Sentiment_3",value='')
cmbs.insert(loc=3,column="Majority",value='')


cmbs.to_csv(open(file_path+asset_class+'_'+your_name+'.csv','w',encoding='utf-8'),index=False, encoding='utf-8',line_terminator='\n')
#the main issue = symbols are then messed up and may cause issues with the accuracy

cmbs_sample = pd.DataFrame({'Majority':cmbs_sample['Majority'],'tweet_id':cmbs_sample['tweet_id']})
cmbs_sample = cmbs_sample.astype({"Majority": int, "tweet_id": object})

cmbs_sample['tweet_id']

test = cmbs
test['tweet_id']
test = test.drop(columns=['Majority'])

test = pd.merge(test,cmbs_sample,how='left',on=['tweet_id','tweet_id'])
test.describe()
#and just to check they were added:
test[test['Majority'].notnull()]
#however a workaround would be reading in twitter id and then 'majority column'
#need to test join fuction on DF: we want to join the sample with our DF based on twitter ID
prac_df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                   'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
other_prac_df = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2'],'C': ['blah','blah','blah']})

prac_df.set_index('key').join(other_prac_df.set_index('key')) #ok this joined - but we need to only select one column
prac_df
prac_df['new col']=pd.np.where(prac_df['A']=='A0','neg')
prac_df.loc[prac_df['A']=='A0','new col']='neg'
prac_df.loc[prac_df['A']=='A1','new col']='pos'
prac_df.loc[prac_df['A']=='A2','new col']='nue'
#this will only process sentences, we need it to loop through a data frame and do the same thing, also we will explore bigrams and trigrams

preprocess(test_sent)
def preprocess(df):
    sentence=str(sentence)  #takes sentence and converts to string (if necessary)
    sentence = sentence.lower() #lowercases - caps are likely important
    tweet_pattern = r''' (?x)    #set flag to allow verbose regexps
    (?:https?://|www)\S+     #gets simple URLS  
    | (?::-\)|;-\))          # small list of emoticons
    | &(?:amp|lt|gt|quot);   #XML or HTML entity
    | \#\w+                  #isolates hashtags
    | @\w+                   # isolates mentions like @wu_tang
    | \d+:\d+                # isolates timelike pattern
    | \d+\.d+                #numbers with decimals
    | (?:\d+,)+?\d{3}(?=(?:[^,]|$)) #numbers with a comma
    | \$?\d+(?:\.\d+)?%?         #dollars with numbers and percentages
    | (?:[A-z]\.)+           #simple abbreviations like U.S.A.
    | (?:--+)                #multiple dashes
    | [A-Z|a-z][a-z]*'[a-z]  #deals with contracionts like it's won't
    | \w+(?:-\w+)*           #words with internal hyphens or apostrophes
    | ['\".?!,:;/]+          # special characters
      '''
      
    return nltk.regexp_tokenize(sentence,tweet_pattern)

test_ar= [[1,2],[8,3],[15,3]]
test_ar2 = [['blah','blah','blah'],['ha','ha'],['kodiak',]]
test_ar[0:]

documents = [(sent, cat) for cat in test_ar 
    for sent in test_ar2]
documents

test_ar3 = np.column_stack((test_ar,test_ar2))
test_ar3 = test_ar.append(test_ar2)

test_ing = list(test_ar+test_ar2)
dictionary
res = {} 
for key in test_ar2: 
    for value in test_ar: 
        res[key] = value 
        test_ar.remove(value) 
        break 

w, h = 8, 5;
Matrix = [[0 for x in range(w)] for y in range(h)] 


n=0
for line in test_ar2:
    key,value = line,test_ar[n]
    n=n+1
    items[key]=value

itemDict = {test_ar:test_ar2 for item in items}
test_keys = ["Rash", "Kil", "Varsha"] 
test_values = [1, 4, 5] 
res = {} 
for key in test_keys: 
    for value in test_values: 
        res[key] = value 
        test_values.remove(value) 
        break 
    
test_ar3[0][0]

test_ar3 = test_ar + test_ar2
test_ar3


s = [['A','B','C','D'], ['B','E','F'], ['C','A','B','D'], ['D']]
{t[0]:t[1:] for t in s}
