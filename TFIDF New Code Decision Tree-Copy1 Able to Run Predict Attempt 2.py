#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


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


# In[39]:


train = pd.read_csv('newnewuk.csv')
train_original=train.copy()
test = pd.read_csv('allin.csv')
test_original = test.copy()


# In[40]:


train['text_punct'] = train['text_punct'].str.replace("[^a-zA-Z#]", " ")
test['text']= test['text'].str.replace("[^a-zA-Z#]", " ")


# In[41]:


def clean_text(text):
 text = str(text).lower()
 return text
train['clean_text'] = train['text_punct'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(train['clean_text'])
bow_test = bow_vectorizer.fit_transform(test['clean_text'])
df_bow = pd.DataFrame(bow.todense())


# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(train['clean_text'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())


# In[44]:


train_tfidf_matrix = tfidf_matrix[0:]

train_tfidf_matrix.todense()

train_bow = bow[0:]

train_bow.todense()


x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['abuse'],test_size=0.3,random_state=17)
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,train['abuse'],test_size=0.3,random_state=2)


# In[45]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train_bow,y_train_bow)


# In[46]:


prediction_bow = model.predict_proba(x_valid_bow)

prediction_bow


# In[47]:


train_tfidf_matrix = tfidf_matrix[0:]

train_tfidf_matrix.todense()

train_bow = bow[0:]

train_bow.todense()


x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['abuse'],test_size=0.3,random_state=17)
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,train['abuse'],test_size=0.3,random_state=2)


# In[48]:


from sklearn.metrics import f1_score


prediction_bow = model.predict_proba(x_valid_bow)

prediction_int = prediction_bow[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
prediction_int
log_bow = f1_score(y_valid_bow, prediction_int)

log_bow



# In[49]:


prediction_bow


from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(x_train_bow,y_train_bow)


# In[50]:


dct_bow = dct.predict_proba(x_valid_bow)

dct_bow


# In[51]:


dct_bow=dct_bow[:,1]>=0.3

# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)

# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)

dct_score_bow


# In[52]:


dct.fit(x_train_tfidf,y_train_tfidf)
dct_tfidf = dct.predict_proba(x_valid_tfidf)

dct_tfidf


# In[53]:


dct_tfidf=dct_tfidf[:,1]>=0.3

# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)

# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)

dct_score_tfidf


# In[54]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix_test=tfidf.fit_transform(test['clean_text'])
df_tfidf_test = pd.DataFrame(tfidf_matrix_test.todense())


# In[55]:


tfidf_matrix_test = tfidf_matrix_test[0:]

train_tfidf_matrix.todense()

train_bow = bow[0:]

train_bow.todense()


# In[56]:


test.head


# In[57]:


test_tfidf = tfidf_matrix_test[:22783]
test_pred = dct.predict_proba(test_tfidf)
test_pred_int = test_pred[:] >= 0.3
test_pred_int = test_pred_int.astype(np.int)


# In[58]:


len(test['clean_text'])


# In[59]:


len(test['abuse'])


# In[61]:


test['abuse'] = test_pred_int
submission = test[['clean_text','abuse']]
submission.to_csv('result_allin.csv', index=False)


# In[32]:


submission['abuse'].value_counts()


# In[32]:


submission['abuse'].value_counts(normalize=True)


# In[31]:


## keep getting "ValueError: Length of values does not match length of index" though len both columns is same! 

