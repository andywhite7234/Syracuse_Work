# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:24:09 2020

@author: andy_white
"""

#### First read in CSV file with old tweets, unfortuantely the only way to get tweets was manualy
### But I have about 80 tweeets from Obama leading to his presidency
### I also have about the same from trump leading to his presidency
#### and finally have about 60 leading up to bernies primary. 
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import copy
from wordcloud import WordCloud
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

old_tweets = pd.read_csv('C:/Users/andy_white/Desktop/Projects/Syracuse/IST 736/Project/Old_tweets.csv',encoding= 'unicode_escape')


def preprocess(sentence):
    sentence=str(sentence)  #takes sentence and converts to string (if necessary)
    sentence = sentence.lower() #lowercases - caps are likely important
    sentence=sentence.replace('{html}',"")  #removes any html
    cleanr = re.compile('<.*?>') 
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    rem_parenth = re.sub(r'\([^)]*\)', '', rem_num)   #this will remove parenthesis and contents within. 
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

#ok now we clean and get ready for vectorization
old_tweets['clean_text']=old_tweets['Tweet'].map(lambda s:preprocess(s))

old_tweets=old_tweets[['Person','clean_text']]



MyLength=[]
MyLength1 = []
MyLength1 = old_tweets['clean_text'].str.split()
for x in MyLength1:
    MyLength.append(len(x))

MyLength

### Obama speech

import re
import urllib
from bs4 import BeautifulSoup
import requests
import pprint
import pandas as pd

#this is the advanced search function from Obama's URL from 2012 leading up to the general election
url = "https://www.npr.org/templates/story/story.php?storyId=88478467"

html = urllib.request.urlopen(url).read()

soup = BeautifulSoup(html,'html.parser')
print(soup.title.string)

#can find any tweet data from the source code - scraping doesn't look like a possiblity
text = soup.body.find_all(id='We the people,')
text
#lets also try
page = requests.get(url,verify=False)

soup = BeautifulSoup(page.content, 'html.parser')

results = soup.find(id='storytext')  #storytext was the entire article

print(results.prettify())

job_elems = results.find_all('We the people,') #, class_='card-content')

#Almost there
results2 = results.get_text()

#just need to subset
my_string="hello python world , i'm a beginner "
print(my_string.split("world",1)[1] )

#ok now lets split at "We the people"
results2 = results2.split("has accused the country of bringing on the Sept. 11 attacks by spreading terrorism. ",1)[1]
obama1 = results2

###no central repository, so past speechs saved in excel doc:
speechs = pd.read_csv('C:/Users/andy_white/Desktop/Projects/Syracuse/IST 736/Project/speechs.csv',encoding= 'unicode_escape')
#append obamas speech
speechs = speechs.append({'Title':'We the People','Speech':obama1,'Person': 'Obama','Win':1},ignore_index=True)

speechs.iloc[1,1]
speechs.columns.values
speechs['clean_text']=speechs['Speech'].map(lambda s:preprocess(s))

## Ok lets create a new vector and try to run a topic model
no_features = 1000 #we will keep at 1000, lots of words in the speech
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')

from sklearn.decomposition import NMF, LatentDirichletAllocation
tf_vec_speech = tf_vectorizer.fit_transform(speechs['clean_text'])
tf_feature_names_speech = tf_vectorizer.get_feature_names()

no_topics = len(speechs['clean_text'])
# Run LDA for speech data
lda_speech = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
lda_z_speech = lda_speech.fit_transform(tf_vec_speech)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def display_topics(model, feature_names, no_top_words,speech_title,speech_person):
    for topic_idx, topic in enumerate(model.components_):
        print(speech_person[topic_idx].upper(),' ',speech_title[topic_idx],':',sep='')
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
no_top_words = 10
display_topics(lda_speech,tf_feature_names_speech,no_top_words,speechs['Title'],speechs['Person'])


####LDA Results:
import matplotlib.pyplot as plt
import numpy as np

word_topic = np.array(lda_speech.components_)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(tf_feature_names_speech)

fontsize_base = 200 / np.max(word_topic) # font size for word with largest share in corpus
num_topics = 23
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base*share)

#fig, axes = plt.subplots(nrows=10, ncols=23)
start = 4
end = 10
for t in range(start,end):
    plt.subplot(1, end-start,t)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t+start])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base*share)
        
plt.tight_layout()
plt.show()



#Great - lets prep for the machine learning algos:
biden_speech = speechs[speechs.Person.str.contains('Biden',case=False)]
#speechs = speechs[~speechs.Person.str.contains('Biden')]
#speechs = speechs.reset_index(drop=True)
#speech_test = speechs
#speech_test.drop(speech_test.index[[18,19,20,22]])

MyLength_speech=[]
MyLength1_speech = []
MyLength1_speech = speechs['clean_text'].str.split()
for x in MyLength1_speech:
    MyLength_speech.append(len(x))

MyLength_speech
MySpeechLabel = speechs["Win"]
#MySpeechLabel[22]=1 #for some reason the last column returning a nan
#BidenLength_speech=[]
#BidenLength1_speech = []
#BidenLength1_speech = biden_speech['clean_text'].str.split()
#for x in BidenLength1_speech:
#    BidenLength_speech.append(len(x))

#BidenLength_speech
#Biden_speech1=biden_speech[['clean_text']]
## Remove the labels from the DF
DF_noLabel_speech= speechs[["clean_text"]] 


def add_labels(df,lab=''):
    tmp = copy.deepcopy(df)
    if lab == "Speech LABEL":
        tmp[lab]= MySpeechLabel
    else:
        print('Use either Lie LABEL or Senti LABEL')
        return()
    return(tmp)
#####interchange between the training/testing data and Biden_speech
MyList=[]
for i in range(0,len(DF_noLabel_speech)):   #update here
    NextText=DF_noLabel_speech.iloc[i,0]  #update here
    MyList.append(NextText)
print(MyList[0:4])

##########################################################   VECTORIZE #######################################
#  COUNT VECTORIZER
vectCountOrig = CountVectorizer(input="content")
vectCountOrigFit = vectCountOrig.fit_transform(MyList)   #updated with biden stuff
MyColumnNames=vectCountOrig.get_feature_names()
#We can use MyColumnNames over and over again. As long as we keep using MyList
vectCountOrigDF=pd.DataFrame(vectCountOrigFit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDFspeech = add_labels(vectCountOrigDF, 'Speech LABEL')
print(vectCountOrigDFspeech.head(10))


#Normalized Count Vectorizer:
vectCountNormOrigDF = copy.deepcopy(vectCountOrigDF)
vectCountNormOrigDF["_length"] = MyLength_speech
for col in MyColumnNames:
    vectCountNormOrigDF[col]= vectCountNormOrigDF[col] / vectCountNormOrigDF._length
vectCountNormOrigDF = vectCountNormOrigDF.drop('_length', axis = 1)
print(vectCountNormOrigDF.head(10))
#Nice!
vectCountNormOrigDFLie = add_labels(vectCountNormOrigDF, 'Speech LABEL')


####################   The inverse Vectorizer!!! 
# create the vectorizer
vectTFIDFOrig = TfidfVectorizer(input = 'content')
# tokenize and build vocab
vectTFIDFOrigFit = vectTFIDFOrig.fit_transform(MyList)
vectTFIDFOrigDF = pd.DataFrame(vectTFIDFOrigFit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDFLie = add_labels(vectTFIDFOrigDF, 'Speech LABEL')


### Standardized Count Vectroizer:
vectCountStandDF = copy.deepcopy(vectCountOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountStandDF = pd.DataFrame(scaler.fit_transform(vectCountStandDF), columns = MyColumnNames)
print(vectCountStandDF.head())
vectCountStandDFLie = add_labels(vectCountStandDF, 'Speech LABEL')


## STANDARDIZED NORMALIZED COUNT VECTORIZER
vectCountNormStandDF = copy.deepcopy(vectCountNormOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountNormStandDF = pd.DataFrame(scaler.fit_transform(vectCountNormStandDF), columns = MyColumnNames)
print(vectCountNormStandDF.head(10))
vectCountNormStandDFLie = add_labels(vectCountNormStandDF, 'Speech LABEL')


# STANDARDIZED TFIDF VECTORIZER
vectTFIDFStandDF = copy.deepcopy(vectTFIDFOrigDF)
scaler = preprocessing.MinMaxScaler()
vectTFIDFStandDF = pd.DataFrame(scaler.fit_transform(vectTFIDFStandDF), columns = MyColumnNames)
print(vectTFIDFStandDF.head(10))
vectTFIDFStandDFLie = add_labels(vectTFIDFStandDF, 'Speech LABEL')

#Alright!!!
#We have 8 datasets (6 vectorize methods, 1 label set):
list_of_DF_speech = [vectCountOrigDFspeech, vectCountNormOrigDFLie, vectTFIDFOrigDFLie, vectCountStandDFLie, vectCountNormStandDFLie, vectTFIDFStandDFLie]
list_of_DF_speech_names = ['vectCountOrigDFspeech', 'vectCountNormOrigDFLie', 'vectTFIDFOrigDFLie', 'vectCountStandDFLie', 'vectCountNormStandDFLie', 'vectTFIDFStandDFLie']


train_vec = [1,2,3,5,7,18,9,11,14,16,17]
test_vec = [4,6,8,10,12,13,15]
#new vector for cluster:
TFIDF_vec_all = vectTFIDFStandDF

#subset biden:
Biden_test = vectTFIDFStandDF.iloc[18:22,].values
#drop biden
vectTFIDFStandDF=vectTFIDFStandDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
vectTFIDFStandDFLie = vectTFIDFStandDFLie.drop(vectTFIDFStandDFLie.index[[18,19,20,21]])


#vectTFIDFStand_test = vectTFIDFStandDF.iloc[test_vec,]
#vectTFIDFStand_test_labs = vectTFIDFStandDFLie.iloc[test_vec,]
#vectTFIDFStand_test_labs = vectTFIDFStand_test_labs['Speech LABEL']
#vectTFIDFStand_train =vectTFIDFStandDFLie.iloc[train_vec,]
#vectTFIDFStand_train_used = vectTFIDFStandDFLie.iloc[train_vec,]
X_train=vectTFIDFStandDF.iloc[train_vec,].values
X_test =vectTFIDFStandDF.iloc[test_vec,].values
#now lets test the rest:
X_train=vectCountNormStandDF.iloc[train_vec,].values
X_test =vectCountNormStandDF.iloc[test_vec,].values
##vectCountStandDF
X_train=vectCountStandDF.iloc[train_vec,].values
X_test =vectCountStandDF.iloc[test_vec,].values
#vectCountNormOrigDF
X_train=vectCountNormOrigDF.iloc[train_vec,].values
X_test =vectCountNormOrigDF.iloc[test_vec,].values
#vectTFIDFOrigDF
X_train=vectTFIDFOrigDF.iloc[train_vec,].values
X_test =vectTFIDFOrigDF.iloc[test_vec,].values
#vectCountOrigDF
X_train=vectCountOrigDF.iloc[train_vec,].values
X_test =vectCountOrigDF.iloc[test_vec,].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y=vectTFIDFStandDFLie['Speech LABEL'].values
y = le.fit_transform(y)
y_train = y[train_vec,]
y_test = y[test_vec,]


import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

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
target=vectTFIDFStandDFLie['Speech LABEL'].values
mat = confusion_matrix(y_test, mnb_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)#,
           # xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

biden_pred = mnb.predict(Biden_test)
print(biden_pred)
pd.Series(biden_pred)
### Now for the test:

#biden_speech2 = biden_speech
#biden_speech2=biden_speech.iloc[0:,0]
#biden_speech2=pd.DataFrame(biden_speech2)
#biden_speech2['tfidf']=pd.Series(biden_pred,index=biden_speech2.index)
#biden_speech2


#Now lets cluster:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
cluster_labs = speechs['Person']
true_k = 5 #five candidates
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#take our TFID_vec_all from above:
model.fit(TFIDF_vec_all)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectTFIDFOrig.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

X = vectTFIDFOrigFit
y = cluster_labs

from yellowbrick.text import TSNEVisualizer
tsne = TSNEVisualizer()
tsne.fit(X, y)
tsne.show()

####twitter##########################################################
######################################################################
#####################################################################
#####################################################################:
old_tweets = pd.read_csv('C:/Users/andy_white/Desktop/Projects/Syracuse/IST 736/Project/Old_tweets.csv',encoding= 'unicode_escape')


def preprocess(sentence):
    sentence=str(sentence)  #takes sentence and converts to string (if necessary)
    sentence = sentence.lower() #lowercases - caps are likely important
    sentence=sentence.replace('{html}',"")  #removes any html
    cleanr = re.compile('<.*?>') 
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    rem_parenth = re.sub(r'\([^)]*\)', '', rem_num)   #this will remove parenthesis and contents within. 
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

#ok now we clean and get ready for vectorization
old_tweets['clean_text']=old_tweets['Tweet'].map(lambda s:preprocess(s))

old_tweets=old_tweets[['Person','clean_text']]



MyLength=[]
MyLength1 = []
MyLength1 = old_tweets['clean_text'].str.split()
for x in MyLength1:
    MyLength.append(len(x))

MyLength
DF_noLabel_speech= old_tweets[["clean_text"]] 
MyTwitterLabel = old_tweets[['Person']]

def add_labels(df,lab=''):
    tmp = copy.deepcopy(df)
    if lab == "Person":
        tmp[lab]= MyTwitterLabel
    else:
        print('Use either Lie LABEL or Senti LABEL')
        return()
    return(tmp)
#####interchange between the training/testing data and Biden_speech
MyList=[]
for i in range(0,len(DF_noLabel_speech)):   #update here
    NextText=DF_noLabel_speech.iloc[i,0]  #update here
    MyList.append(NextText)
print(MyList[0:4])

#  COUNT VECTORIZER
vectCountOrig = CountVectorizer(input="content")
vectCountOrigFit = vectCountOrig.fit_transform(MyList)   #updated with biden stuff
MyColumnNames=vectCountOrig.get_feature_names()
#We can use MyColumnNames over and over again. As long as we keep using MyList
vectCountOrigDF=pd.DataFrame(vectCountOrigFit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDFspeech = add_labels(vectCountOrigDF, 'Person')
print(vectCountOrigDFspeech.head(10))


#Normalized Count Vectorizer:
vectCountNormOrigDF = copy.deepcopy(vectCountOrigDF)
vectCountNormOrigDF["_length"] = MyLength
for col in MyColumnNames:
    vectCountNormOrigDF[col]= vectCountNormOrigDF[col] / vectCountNormOrigDF._length
vectCountNormOrigDF = vectCountNormOrigDF.drop('_length', axis = 1)
print(vectCountNormOrigDF.head(10))
#Nice!
vectCountNormOrigDFLie = add_labels(vectCountNormOrigDF, 'Person')


####################   The inverse Vectorizer!!! 
# create the vectorizer
vectTFIDFOrig = TfidfVectorizer(input = 'content')
# tokenize and build vocab
vectTFIDFOrigFit = vectTFIDFOrig.fit_transform(MyList)
vectTFIDFOrigDF = pd.DataFrame(vectTFIDFOrigFit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDFLie = add_labels(vectTFIDFOrigDF, 'Person')


### Standardized Count Vectroizer:
vectCountStandDF = copy.deepcopy(vectCountOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountStandDF = pd.DataFrame(scaler.fit_transform(vectCountStandDF), columns = MyColumnNames)
print(vectCountStandDF.head())
vectCountStandDFLie = add_labels(vectCountStandDF, 'Person')


## STANDARDIZED NORMALIZED COUNT VECTORIZER
vectCountNormStandDF = copy.deepcopy(vectCountNormOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountNormStandDF = pd.DataFrame(scaler.fit_transform(vectCountNormStandDF), columns = MyColumnNames)
print(vectCountNormStandDF.head(10))
vectCountNormStandDFLie = add_labels(vectCountNormStandDF, 'Person')


# STANDARDIZED TFIDF VECTORIZER
vectTFIDFStandDF = copy.deepcopy(vectTFIDFOrigDF)
scaler = preprocessing.MinMaxScaler()
vectTFIDFStandDF = pd.DataFrame(scaler.fit_transform(vectTFIDFStandDF), columns = MyColumnNames)
print(vectTFIDFStandDF.head(10))
vectTFIDFStandDFLie = add_labels(vectTFIDFStandDF, 'Person')


#####Lets cluster some tweets:
TFIDF_vec_all = vectTFIDFStandDF   #circular - i know....
true_k = 5 #five candidates
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#take our TFID_vec_all from above:
model.fit(TFIDF_vec_all)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectTFIDFOrig.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

X = vectTFIDFOrigFit
y = MyTwitterLabel['Person'].values.tolist()

from yellowbrick.text import TSNEVisualizer
tsne = TSNEVisualizer()
tsne.fit(X, y)
tsne.show()



####Try this, this gives how similar each document is to eachother
### think correlation matrix
from sklearn.metrics.pairwise import cosine_similarity
MyTwitterLabel[0:]
vectTFIDFOrigDF.head()

vectTFIDFOrigDF.index=MyTwitterLabel
similarity_matrix = cosine_similarity(vectTFIDFOrigDF)
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df_new = similarity_df
similarity_df_new.index=MyTwitterLabel[0:]
similarity_df_new.columns = MyTwitterLabel

from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(similarity_matrix, 'ward')
pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2', 
                         'Distance', 'Cluster Size'], dtype='object')

plt.figure(figsize=(50, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z,leaf_font_size=12,labels=similarity_df_new.index)
plt.axhline(y=1.0, c='k', ls='--', lw=0.5)

from scipy.cluster.hierarchy import fcluster
max_dist = 1.0

cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([cluster_labels,MyTwitterLabel], axis=1)

#subset biden:
Biden_test = vectTFIDFStandDF.iloc[283:,].values
#drop biden
vectTFIDFStandDF=vectTFIDFStandDF.drop(vectTFIDFStandDF.index[283:])
vectTFIDFStandDFLie = vectTFIDFStandDFLie.drop(vectTFIDFStandDFLie.index[283:])
Biden_test_CNS = vectCountNormStandDF.iloc[283:,].values
Biden_test_CST = vectCountStandDF.iloc[283:,].values
Biden_test_CNO =vectCountNormOrigDF.iloc[283:,].values
Biden_test_TFIdf =vectTFIDFOrigDF.iloc[283:,].values
Biden_test_COD =vectCountOrigDF.iloc[283:,].values
#drop biden

vectCountNormStandDF = vectCountNormStandDF.drop(vectCountNormStandDF.index[283:])
vectCountStandDF = vectCountStandDF.drop(vectCountStandDF.index[283:])
vectCountNormOrigDF = vectCountNormOrigDF.drop(vectCountNormOrigDF.index[283:])
vectTFIDFOrigDF = vectTFIDFOrigDF.drop(vectTFIDFOrigDF.index[283:])
vectCountOrigDF = vectCountOrigDF.drop(vectCountOrigDF.index[283:])


X = vectTFIDFStandDF.values
X = vectCountStandDF.values
X = vectCountNormOrigDF.values
X = vectTFIDFOrigDF.values
X = vectCountOrigDF.values

y=vectTFIDFStandDFLie['Person'].values
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


biden_pred = mnb.predict(Biden_test)
print(biden_pred)
plt.hist(biden_pred)
plt.show()
###get important words:
#vectTFIDFOrig = TfidfVectorizer(input = 'content')
neg_class_prob_sorted = mnb.feature_log_prob_[0, :].argsort()
pos_class_prob_sorted = mnb.feature_log_prob_[1, :].argsort()

print(np.take(vectTFIDFOrig.get_feature_names(), neg_class_prob_sorted[:11]))
print(np.take(vectTFIDFOrig.get_feature_names(), pos_class_prob_sorted[:11]))

#### SVM on Twitter data
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train,y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)

biden_pred=SVM.predict(Biden_test_COD)
print(biden_pred)
plt.hist(biden_pred)
plt.show()

##########  ##############################
##########################################################   VECTORIZE BiGRAM #######################################
#  COUNT VECTORIZER
vectCountOrig = CountVectorizer(input="content",ngram_range=(2, 2))
vectCountOrigFit = vectCountOrig.fit_transform(MyList)   #updated with biden stuff
MyColumnNames=vectCountOrig.get_feature_names()
#We can use MyColumnNames over and over again. As long as we keep using MyList
vectCountOrigDF=pd.DataFrame(vectCountOrigFit.toarray(), columns = MyColumnNames)
print(vectCountOrigDF.head(10))
vectCountOrigDFspeech = add_labels(vectCountOrigDF, 'Speech LABEL')
print(vectCountOrigDFspeech.head(10))


#Normalized Count Vectorizer:
vectCountNormOrigDF = copy.deepcopy(vectCountOrigDF)
vectCountNormOrigDF["_length"] = MyLength_speech
for col in MyColumnNames:
    vectCountNormOrigDF[col]= vectCountNormOrigDF[col] / vectCountNormOrigDF._length
vectCountNormOrigDF = vectCountNormOrigDF.drop('_length', axis = 1)
print(vectCountNormOrigDF.head(10))
#Nice!
vectCountNormOrigDFLie = add_labels(vectCountNormOrigDF, 'Speech LABEL')


####################   The inverse Vectorizer!!! 
# create the vectorizer
vectTFIDFOrig = TfidfVectorizer(input = 'content',ngram_range=(2, 2))
# tokenize and build vocab
vectTFIDFOrigFit = vectTFIDFOrig.fit_transform(MyList)
vectTFIDFOrigDF = pd.DataFrame(vectTFIDFOrigFit.toarray(), columns = MyColumnNames)
print(vectTFIDFOrigDF.head(10))
vectTFIDFOrigDFLie = add_labels(vectTFIDFOrigDF, 'Speech LABEL')


### Standardized Count Vectroizer:
vectCountStandDF = copy.deepcopy(vectCountOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountStandDF = pd.DataFrame(scaler.fit_transform(vectCountStandDF), columns = MyColumnNames)
print(vectCountStandDF.head())
vectCountStandDFLie = add_labels(vectCountStandDF, 'Speech LABEL')


## STANDARDIZED NORMALIZED COUNT VECTORIZER
vectCountNormStandDF = copy.deepcopy(vectCountNormOrigDF)
scaler = preprocessing.MinMaxScaler()
vectCountNormStandDF = pd.DataFrame(scaler.fit_transform(vectCountNormStandDF), columns = MyColumnNames)
print(vectCountNormStandDF.head(10))
vectCountNormStandDFLie = add_labels(vectCountNormStandDF, 'Speech LABEL')


# STANDARDIZED TFIDF VECTORIZER
vectTFIDFStandDF = copy.deepcopy(vectTFIDFOrigDF)
scaler = preprocessing.MinMaxScaler()
vectTFIDFStandDF = pd.DataFrame(scaler.fit_transform(vectTFIDFStandDF), columns = MyColumnNames)
print(vectTFIDFStandDF.head(10))
vectTFIDFStandDFLie = add_labels(vectTFIDFStandDF, 'Speech LABEL')


train_vec = [1,2,3,5,7,18,9,11,14,16,17]
test_vec = [4,6,8,10,12,13,15]
#new vector for cluster:
TFIDF_vec_all = vectTFIDFStandDF

#subset biden:
Biden_test = vectTFIDFStandDF.iloc[18:22,].values
Biden_test_CNS = vectCountNormStandDF.iloc[18:22,].values
Biden_test_CST = vectCountStandDF.iloc[18:22,].values
Biden_test_CNO =vectCountNormOrigDF.iloc[18:22,].values
Biden_test_TFIdf =vectTFIDFOrigDF.iloc[18:22,].values
Biden_test_COD =vectCountOrigDF.iloc[18:22,].values
#drop biden
vectTFIDFStandDF=vectTFIDFStandDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
vectTFIDFStandDFLie = vectTFIDFStandDFLie.drop(vectTFIDFStandDFLie.index[[18,19,20,21]])
vectCountNormStandDF = vectCountNormStandDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
vectCountStandDF = vectCountStandDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
vectCountNormOrigDF = vectCountNormOrigDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
vectTFIDFOrigDF = vectTFIDFOrigDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
vectCountOrigDF = vectCountOrigDF.drop(vectTFIDFStandDF.index[[18,19,20,21]])
#vectTFIDFStand_test = vectTFIDFStandDF.iloc[test_vec,]
#vectTFIDFStand_test_labs = vectTFIDFStandDFLie.iloc[test_vec,]
#vectTFIDFStand_test_labs = vectTFIDFStand_test_labs['Speech LABEL']
#vectTFIDFStand_train =vectTFIDFStandDFLie.iloc[train_vec,]
#vectTFIDFStand_train_used = vectTFIDFStandDFLie.iloc[train_vec,]
X_train=vectTFIDFStandDF.iloc[train_vec,].values
X_test =vectTFIDFStandDF.iloc[test_vec,].values
#now lets test the rest:
X_train=vectCountNormStandDF.iloc[train_vec,].values
X_test =vectCountNormStandDF.iloc[test_vec,].values
##vectCountStandDF
X_train=vectCountStandDF.iloc[train_vec,].values
X_test =vectCountStandDF.iloc[test_vec,].values
#vectCountNormOrigDF
X_train=vectCountNormOrigDF.iloc[train_vec,].values
X_test =vectCountNormOrigDF.iloc[test_vec,].values
#vectTFIDFOrigDF
X_train=vectTFIDFOrigDF.iloc[train_vec,].values
X_test =vectTFIDFOrigDF.iloc[test_vec,].values
#vectCountOrigDF
X_train=vectCountOrigDF.iloc[train_vec,].values
X_test =vectCountOrigDF.iloc[test_vec,].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y=vectTFIDFStandDFLie['Speech LABEL'].values
y = le.fit_transform(y)
y_train = y[train_vec,]
y_test = y[test_vec,]


import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

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


biden_pred = mnb.predict(Biden_test_CNS)
print(biden_pred)

#####Old code

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

MyTextDF = speechs
# possible labels indices for both labels because we have a small dataset
Win_ind = MyTextDF[(MyTextDF["Win"] == 1)].index
Loss_ind = MyTextDF[(MyTextDF["Win"] == 0)].index
trainIndex = []
testIndex = []
trainIndexW = []
testIndexW = []
trainIndexL = []
testIndexL = []
#Get two folds
kfwin = KFold(n_splits = 5, shuffle = True)
kfwin.get_n_splits(Win_ind)
kfloss = KFold(n_splits = 5, shuffle = True)
kfloss.get_n_splits(Loss_ind)
for train_index, test_index in kfwin.split(Win_ind):
    trainIndexW.append(train_index)
    testIndexW.append(test_index)
for train_index, test_index in kfloss.split(Loss_ind):
    trainIndexL.append(train_index)
    testIndexL.append(test_index)
for i in range(9):
    trainIndex.append(trainIndexW[i] + trainIndexL[i])
    testIndex.append(testIndexW[i] + testIndexL[i])

#MULTINOMIAL NAIVE BAYES LIE
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#For each metric, we have a dictionary where the keys are the experiment dataframes and the value is a list of the cross validation metrics

cm_Lie = {}
acc_Lie = {}
prfs_Lie = {}
#Also get the most important features for each run
features_Lie = {}

for loc in range(len(list_of_DF_speech)):
    DF = list_of_DF_speech[loc]
    name = list_of_DF_speech_Names[loc]
    cm_Lie[name] = []
    acc_Lie[name] = []
    prfs_Lie[name] = []
    features_Lie[name] = []
    ind = 1
    for train_ind, test_ind in zip(trainIndex_Lie, testIndex_Lie):
        train = DF.iloc[train_ind, ]
        test = DF.iloc[test_ind, ]
        #Remove labels
        trainLabels = train["Lie LABEL"]
        testLabels = test["Lie LABEL"]
        train = train.drop(["Lie LABEL"], axis = 1)
        test = test.drop(["Lie LABEL"], axis = 1)
        #Create the modeler
        MyModelNB= MultinomialNB()
        MyModelNB.fit(train, trainLabels)
        Prediction = MyModelNB.predict(test)
        y_true = (testLabels).tolist()
        y_predict = (Prediction).tolist()
        labels =['lie', 'truth']
        cm = confusion_matrix(y_true, y_predict, labels)
        cm_Lie[name].append(cm)
        acc = accuracy_score(y_true, y_predict)
        acc_Lie[name].append(acc)
        prfs = precision_recall_fscore_support(y_true, y_predict, pos_label = 'lie', average = 'binary')
        prfs_Lie[name].append(prfs)
        features_Lie[name].append(feat_imp(train, MyModelNB))
    #Plot the confusion matrix
    # title = str('Confusion Matrix\n' + name + ' fold ' + str(ind))
    # cm_plot = plot_confusion_matrix(y_true = y_true, y_pred = y_predict, classes = labels, normalize=False, title=title)
    # outpath = str('output/Lie/confmat/' + name + '_fold_' + str(ind) + '.png')
    # plt.savefig(outpath, bbox_inches='tight')
    #
    # plt.clf()
    #
    # #Create a word cloud
    # wc = WordCloud().generate_from_frequencies(features_Lie[name][ind - 1])
    # plt.imshow(wc)
    # plt.xticks(ticks = None)
    # plt.yticks(ticks = None)
    # outpath = str('output/Lie/wordclouds/' + name + '_fold_' + str(ind) + '.png')
    # plt.savefig(outpath, bbox_inches='tight')
        ind += 1






