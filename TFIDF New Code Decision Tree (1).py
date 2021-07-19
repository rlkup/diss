#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import re


# In[2]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('newnewuk.csv')
train_original=train.copy()
test = pd.read_csv('ali.csv')
test_original = test.copy()


# In[4]:


train['text_punct'] = train['text_punct'].str.replace("[^a-zA-Z#]", " ")
test['text']= test['text'].str.replace("[^a-zA-Z#]", " ")


# In[5]:


def clean_text(text):
 text = str(text).lower()
 return text
train['clean_text'] = train['text_punct'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(train['clean_text'])
bow_test = bow_vectorizer.fit_transform(test['clean_text'])
df_bow = pd.DataFrame(bow.todense())


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(train['clean_text'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())


# In[9]:


train_tfidf_matrix = tfidf_matrix[0:]

train_tfidf_matrix.todense()

train_bow = bow[0:]

train_bow.todense()


x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['abuse'],test_size=0.3,random_state=17)
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,train['abuse'],test_size=0.3,random_state=2)


# In[10]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train_bow,y_train_bow)


# In[11]:


prediction_bow = model.predict_proba(x_valid_bow)

prediction_bow


# In[12]:


from sklearn.metrics import f1_score

prediction_int = prediction_bow[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
prediction_int
log_bow = f1_score(y_valid_bow, prediction_int)

log_bow


# In[13]:



model.fit(x_train_tfidf,y_train_tfidf)
prediction_tfidf = model.predict_proba(x_valid_tfidf)

prediction_tfidf


# In[14]:


prediction_int = prediction_tfidf[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
prediction_int

log_tfidf = f1_score(y_valid_tfidf, prediction_int)

log_tfidf


# In[45]:


from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


# In[46]:


model2 = SGDClassifier()
model2.fit(x_train_bow,y_train_bow)


# In[47]:


prediction_bow = model2.predict(x_valid_bow)

prediction_bow


# In[48]:


prediction_int = prediction_bow[:]
prediction_int = prediction_int.astype(np.int)
prediction_int

sdg_bow = f1_score(y_valid_bow, prediction_int)

sdg_bow


# In[49]:



model2.fit(x_train_tfidf,y_train_tfidf)
prediction_tfidf = model2.predict(x_valid_tfidf)

prediction_tfidf


# In[50]:


prediction_int = prediction_tfidf[:]
prediction_int = prediction_int.astype(np.int)
prediction_int

sdg_tfidf = f1_score(y_valid_tfidf, prediction_int)

sdg_tfidf


# In[51]:


from sklearn.naive_bayes import MultinomialNB
model3 = MultinomialNB()
model3.fit(x_train_bow,y_train_bow)


# In[52]:


prediction_bow = model3.predict(x_valid_bow)

prediction_bow


# In[26]:


prediction_int = prediction_bow[:]
prediction_int = prediction_int.astype(np.int)
prediction_int

log_bow = f1_score(y_valid_bow, prediction_int)

log_bow


# In[27]:



model3.fit(x_train_tfidf,y_train_tfidf)
prediction_tfidf = model3.predict(x_valid_tfidf)

prediction_tfidf


# In[28]:


prediction_int = prediction_tfidf[:]
prediction_int = prediction_int.astype(np.int)
prediction_int

log_tfidf = f1_score(y_valid_tfidf, prediction_int)

log_tfidf


# In[34]:


from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(x_train_bow,y_train_bow)


# In[35]:


dct_bow = dct.predict_proba(x_valid_bow)

dct_bow


# In[36]:


dct_bow=dct_bow[:,1]>=0.3

# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)

# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)

dct_score_bow


# In[37]:


dct.fit(x_train_tfidf,y_train_tfidf)
dct_tfidf = dct.predict_proba(x_valid_tfidf)

dct_tfidf


# In[38]:


dct_tfidf=dct_tfidf[:,1]>=0.3

# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)

# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)

dct_score_tfidf


# In[53]:


Algo_1 = ['LinearSVC(Bag-of-Words)','SGDClassifier(Bag-of-Words)','DecisionTree(Bag-of-Words)']

score_1 = [log_bow,sdg_bow,dct_score_bow]

compare_1 = pd.DataFrame({'Model':Algo_1,'F1_Score':score_1},index=[i for i in range(1,4)])

compare_1.T


# In[54]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare_1)

plt.title('Bag-of-Words')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[56]:


Algo_2 = ['LinearSVC(TF-IDF)','SDGClassifier(TF-IDF)','DecisionTree(TF-IDF)']

score_2 = [log_tfidf,sdg_tfidf,dct_score_tfidf]

compare_2 = pd.DataFrame({'Model':Algo_2,'F1_Score':score_2},index=[i for i in range(1,4)])

compare_2.T


# In[57]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare_2)

plt.title('TF-IDF')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[58]:


test = pd.read_csv('ali.csv')
test_original = test.copy()
test['text']= test['text'].str.replace("[^a-zA-Z#]", " ")
test['clean_text'] = test['text'].apply(clean_text)


# In[59]:


test.head


# In[131]:


#kept getting value error- maybe because abuse was already a column? so removed it and we'll see


# In[60]:


len(test['clean_text'])


# In[61]:


train.head


# In[62]:


test_tfidf = tfidf_matrix[:22783]
test_pred = dct.predict_proba(test_tfidf)
test_pred_int = test_pred[:] >= 0.3
test_pred_int = test_pred_int.astype(np.int)


# In[63]:


test['abuse'] = test_pred_int
submission = test[['clean_text','abuse']]
submission.to_csv('result.csv', index=False)


# In[65]:


len(test['abuse'])


# In[66]:


len(test['clean_text'])


# In[67]:


## keep getting "ValueError: Length of values does not match length of index" though len both columns is same! 

