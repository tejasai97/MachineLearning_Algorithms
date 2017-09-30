
# coding: utf-8



# In[2]:

import nltk
#nltk.download() 


# ## Get the Data

# [UCI datasets](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)! 


# In[3]:

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))


\
# In[4]:

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')



# In[4]:

import pandas as pd



# In[5]:

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


# ## Exploratory Data Analysis

# In[6]:

messages.describe()



# In[7]:

messages.groupby('label').describe()



# In[8]:

messages['length'] = messages['message'].apply(len)
messages.head()


# ### Data Visualization

# In[9]:

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[10]:

messages['length'].plot(bins=50, kind='hist') 



# In[11]:

messages.length.describe()


# In[12]:

messages[messages['length'] == 910]['message'].iloc[0]



# In[13]:

messages.hist(column='length', by='label', bins=50,figsize=(12,4))



# ## Text Pre-processing


# In[14]:

import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)



# In[15]:

from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[16]:

nopunc.split()


# In[17]:

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[18]:

clean_mess



# In[19]:

def text_process(mess):
    
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
 
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[20]:

messages.head()


# In[21]:

# Check to make sure its working
messages['message'].head(5).apply(text_process)


# In[22]:

# Show original dataframe
messages.head()


# ### Continuing Normalization
# 

# In[23]:

from sklearn.feature_extraction.text import CountVectorizer



# In[24]:

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))



# In[25]:

message4 = messages['message'][3]
print(message4)



# In[26]:

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)



# In[27]:

print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])



# In[28]:

messages_bow = bow_transformer.transform(messages['message'])


# In[29]:

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[30]:

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)



# In[32]:

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])



# In[33]:

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)



# ## Training a model


# In[34]:

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])



# In[35]:

print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])



# ## Part 6: Model Evaluation

# In[36]:

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)



# In[37]:

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))



# ## Train Test Split

# In[38]:

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))



# ## Creating a Data Pipeline

# In[39]:

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])



# In[40]:

pipeline.fit(msg_train,label_train)


# In[41]:

predictions = pipeline.predict(msg_test)


# In[42]:

print(classification_report(predictions,label_test))
