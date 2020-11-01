# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:26:00 2020

@author: andy_white
"""
# To download a package:
pip install GetOldTweets3

import GetOldTweets3 as got
from datetime import date, timedelta
import datetime
from time import gmtime, strftime
import time
import pandas as pd
import numpy as np

#Update your name and Asset_class, this will be important when saving files:
your_name = 'Andy_White'
asset_class = 'CMBS'
#also if you are in abs auto update something like abs_auto
#Finally, if you have provided twitter accounts in the excel doc and don't care how I built the functions
# jump to line 173 the line starts with "def tweets_by_practice(group,date_start,date_end): "

#Keyword list, this is specific to you asset class:
keyword = ['commercial real estate','cmbs','cmbx','']
#need to modify to add a list of twitter accounts, then loop through and pull from all dates associated with a twitter account
#notice that you can set a query search or a username search - you can modify however you want. I searched any metion of CMBS in the last 15 days
tweetCriteria = (got.manager.
                 TweetCriteria().
                #setUsername("TreppWire")
                 setQuerySearch('commercial real estate')
                 .setTopTweets(True)
                 .setSince(str(date.today() - timedelta(days = 15)))
                 .setUntil(str(date.today()))
                 .setLang('en')
                 .setMaxTweets(0)) # 0 retrieves all

tweet = got.manager.TweetManager.getTweets(tweetCriteria)
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

tweet_texts = [x.text for x in tweet]
len(tweet_texts)
tweet_date = [x.date for x in tweet]
tweet_hashtag = [x.hashtags for x in tweet]
#tweet_geo = [x.geo for x in tweet] can't get geoo location
tweet_retweet = [x.retweets for x in tweet]
tweet_permalink = [x.permalink for x in tweet]
tweet_user = [x.username for x in tweet]
#tweet_retweets = [x.retweets for x in tweet]
tweet_favs = [x.favorites for x in tweet]
tweet_mentions = [x.mentions for x in tweet]
tweet_id = [x.id for x in tweet]
tweet_to = [x.to for x in tweet]
tweet_to[:10]      


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


#updated this to tweets to keyword search, also top_tweets parameter requires a true false
def tweet_by_keyword(group,date_start,date_end,top_tweets):
    
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
    for row in keyword:
        tweetCriteria = (got.manager.
                 TweetCriteria().
                
                setQuerySearch(str(row))    
                .setTopTweets(top_tweets) 
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
    df = pd.DataFrame({'Sentiment_Analysis_1':'','Garbage?':'',#'Sentiment_Analysis_2':'','Sentiment_Analysis_3':'','Majority':'',
                       'text':tweet_texts,'hashtag':tweet_hashtag,'date':tweet_date,'retweet':tweet_retweet,'permalink':tweet_permalink,
                       'user':tweet_user,'favorites':tweet_favs,'mentions':tweet_mentions,'tweet_id':tweet_id,'tweet_to':tweet_to})
    df['date']=df['date'].dt.tz_localize(None) #need this to save files to excel. important so you can edit in docshare
    return df


#test it out
test=tweet_by_keyword("CMBS","06/10/2020","06/20/2020",True)
list(test.columns.values)
test.head()
test.iloc[1]
len(test)
##pulling this number of tweets took 8 minutes 45 seconds (yikes!!!), but returned 8668 tweets! So about 1000 tweets = 1 minute
cmbs= tweet_by_keyword("CMBS","06/01/2019","06/20/2020",True)
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
file_path = save_tweets(cmbs,400,'_v2')


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
cmbs_sample = pd.read_excel(file_path)
cmbs_sample['Sentiment_Analysis_1'].value_counts().plot.bar()

#now comes the NLP/Text Mining aspect, lets start with an example
test_sent = "It's crazy. This is the U.S.A. and I spent $12.99 on a 1,000 pound dog that won't play fetch!!! It's a dog-eat-dog world #fun"
import nltk
import re
from nltk.collocations import *
from nltk.corpus import stopwords
bigram_measures = nltk.collocations.BigramAssocMeasures()
nltk.word_tokenize(test_sent)
nltk.download('nps_chat')
nltk.corpus.nps_chat.tagged_words()[:50]

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

#now see what happens in our test sentence:
preprocess_stopped(test_sent)

#need to see if we can apply the pattern to a df

def get_all_WF_unigram(df,num_of_word_items,num_of_word_items2,bigram_chi_square,bigram_stop_chi_square):
    '''
    Need to tokenize, so the "df['unigram']=df['text'].map(lambda s:preprocess_unigram(s))" is basically applying our
    unigram preprocess function from above to tokenize, and the preprocess_stopped then removes stop words
    '''
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
    featuresets = [(document_features(d,word_features), c) for (d,c) in all_words1]
    featuresets_stop = [(document_features(d,word_features_stop), c) for (d,c) in all_words_stop1]
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



####Now to run the naive bayes algo on our featuresets to get an accuracy score
def NB_class_all (x1,x2,x3,x4,x5,test_size):
    num = int(round(len(unigram)*test_size,0))
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
unigram,unigram_stop,bigram,bigram_stop,POS_tags,word_features,word_feature_stop,all_words_list,all_words_stop_list = get_all_WF_unigram(cmbs_sample,300,200,200,75)
#do not change these names ('uni','uni_stop','bi','bistop') - they are used in the next function

uni, uni_stop,  bi,   bi_stop, pos = NB_class_all(unigram,unigram_stop,bigram,bigram_stop,POS_tags,.20)

uni.show_most_informative_features(30)

uni.classify(unigram[0][0])
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
'''
NEED TO FINISH THIS OFF - JUST HAVE ONE WAY TO CLASSIFY
'''
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
all_words1, all_words_stop1, all_words_list,all_words_stop_list = classifier_preprocess(cmbs)

#finally  the classifier step, classifies our universe with user specified 
def the_classifier(classifier_input,df):
    if str(classifier_input) == 'unigram':
        featuresets = [(document_features(d,word_features), c) for (d,c) in all_words1] #word features were defined above in our sample
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
        featuresets = [(document_features(d,word_features_stop), c) for (d,c) in all_words_stop1]
        
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
test1= the_classifier('unigram',cmbs)
test1['polarity']
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

MyTwitterLabel = test[['polarity']] #double brakets get into list format

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



all_words = test['unigram'].tolist()
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)

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

#lets find accuracy on stopped words
all_words = test['unigram_stop'].tolist()
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)

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

all_words = test['bigram'].tolist()
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
mat = confusion_matrix(y_test, y_pred)
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
