#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline


# In[68]:


import nltk
from nltk.corpus import stopwords
stop_words = set (stopwords.words( 'english' ))
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize, word_tokenize
import string


# In[77]:


data = pd.read_csv('newnewuk.csv')

data.head()


# In[78]:


data.columns = ['abuse', 'text']


# In[80]:


data['abuse'].value_counts()


# In[82]:


stemmer = SnowballStemmer(language='english')


# In[98]:


def clean_text(text):
 text = str(text).lower()
 tokens = [stemmer.stem(word) for word in wordpunct_tokenize(text) if word not in list(stop_words) + list(string.punctuation)]
 text = ''.join(tokens)
 return text


# In[85]:


data['text'].iloc[0]


# In[86]:


clean_text(data['text'].iloc[0])


# In[87]:


data['clean_text'] = data['text'].apply(clean_text)


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['abuse'], test_size=0.33, random_state=42)


# In[89]:


vectorizer = TfidfVectorizer(ngram_range=(1,2))


# In[90]:


vector_train = vectorizer.fit_transform(X_train)
vector_test = vectorizer.transform(X_test)


# In[91]:


model = LogisticRegression()
model.fit(vector_train, y_train)


# In[92]:


accuracy_score(y_test, model.predict(vector_test))


# In[99]:


model2 = LinearSVC()
model2.fit(vector_train, y_train)


# In[94]:


accuracy_score(y_test, model2.predict(vector_test))


# In[95]:


model3 = MultinomialNB()
model3.fit(vector_train, y_train)
accuracy_score(y_test, model3.predict(vector_test))


# In[96]:


model4 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
model4.fit(vector_train, y_train)
accuracy_score(y_test, model4.predict(vector_test))

