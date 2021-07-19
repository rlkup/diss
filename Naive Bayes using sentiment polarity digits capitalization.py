#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import collections
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import math
import nltk

np.set_printoptions(suppress=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.metrics import confusion_matrix


# In[9]:


df_input_prefiltered = pd.read_csv('newnewuk.csv')


# In[10]:


df_input_prefiltered.head()


# In[11]:


df_input_prefiltered.columns = ['abuse', 'text']
df_input = df_input_prefiltered


# In[12]:


non_df = df_input[df_input.abuse == 0]
vawip_df = df_input[df_input.abuse == 1]
df = pd.concat([non_df, vawip_df])


# In[13]:


print("Number of Non-Abuse tweets:",len(non_df),"\n")
print("Number of Abuse tweets:",len(vawip_df))


# In[14]:


df.isnull().sum()


# In[18]:


#to get number of digits in a tweet
digits_list = []

for i in range(0,len(df)):
    if(sum(c.isdigit() for c in df.text.iloc[1]) == 0):
        digits_list.append(0)
    else:
        digits_list.append(sum(c.isdigit() for c in df.text.iloc[i]))

digits_col = pd.Series(digits_list)
df['DIGITS'] = digits_col.values


# In[19]:


#to get number of capitalized words
cap_list = []

for i in range(0,len(df)):
    words = df.text.iloc[1].split()
    count = 0
    for j in range(0,len(words)):
        if(words[j].isupper()):
            count = count + 1;
    cap_list.append(count)    
    #print(cap_list)
        
cap_col = pd.Series(cap_list)
df['CAP'] = cap_col.values


# In[20]:


from nltk.corpus import stopwords
filtered_text = []

for i in range(0,len(df)):
    word_list = str(df.text.iloc[i]).split()
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    filtered_text.append(' '.join(filtered_words))
df['filtered_text'] = filtered_text
df['filtered_text_lower'] = df['filtered_text'].str.lower()


# In[21]:


df.columns


# In[22]:


vawip_text = []
vawip_df = df[df.abuse==1]

for i in range(0,len(vawip_df)):
    vawip_text.append(vawip_df.filtered_text.iloc[i])
    
non_text = []
non_df = df[df.abuse==0]

for i in range(0,len(non_df)):
    non_text.append(non_df.filtered_text.iloc[i])


# In[23]:


vawip_text = ' '.join(vawip_text).split()
freq = nltk.FreqDist(vawip_text)

print("\nBelow is the plot of 50 most commonly used words in Abuse post")
plt.figure(figsize=(15,7))
freq.most_common(50)
freq.plot(50)

non_text = ' '.join(non_text).split()
freq = nltk.FreqDist(non_text)
print("\nBelow is the plot of 50 most commonly used words in NonAbuse comment")

plt.figure(figsize=(15,7))
freq.most_common(50)
freq.plot(50)


# In[25]:


#removed usernames in Attempt6copy
#had to try new code here so I think I may have messed it up- want to make sure that I can use bigrams and trigrams! 
from itertools import combinations 
from collections import Counter

from nltk.corpus import stopwords
stoplist = stopwords.words('english') + ['though']

from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
# matrix of ngrams
ngrams = c_vec.fit_transform(df['text']).astype('U').values
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})


"""
#code original text: 
from itertools import combinations from collections import Counter

def ngram(lines,i): pair_counter = Counter() for line in lines: unique_tokens = sorted(set(line))
combos = combinations(unique_tokens, i) pair_counter += Counter(combos) return pair_counter

vawip_df = df[df.label==1] 
non_df = df[df.label==0]

lines = [] for i in range(0,len(vawip_df)): tokens = vawip_df.filtered_text_lower.iloc[i].split() lines.append(tokens)

one_gram_vawip = ngram(lines,1) 
bi_gram_vawip = ngram(lines,2) 
tri_gram_vawip = ngram(lines,3)

lines = [] for i in range(0,len(non_df)): tokens = non_df.filtered_text_lower.iloc[i].split() lines.append(tokens)

one_gram_non = ngram(lines,1) 
bi_gram_non = ngram(lines,2) 
tri_gram_non = ngram(lines,3)

one_gram_vawip.most_common(20)

bi_gram_vawip.most_common(20)

tri_gram_vawip.most_common(20)

one_gram_non.most_common(20)

bi_gram_non.most_common(20)

tri_gram_non.most_common(20)"""


# In[26]:


get_ipython().system('pip install wordcloud')

text = ''
for i in range(0,len(vawip_df)):
    text = text + str(vawip_df.filtered_text_lower.iloc[i])
    
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,9))
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[18]:


text = ''
for i in range(0,len(non_df)):
    text = text + str(non_df.filtered_text_lower.iloc[i])
    
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,9))
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[27]:


from textblob import TextBlob

polarity=[]
subjectivity=[]
sentiment_flag=[]
subjectivity_flag=[]
for i in df.filtered_text_lower:
    testimonial = TextBlob(i).sentiment
    polarity.append(testimonial.polarity)
    subjectivity.append(testimonial.subjectivity)
    
    if testimonial.polarity>0.33:
        sentiment_flag.append('positive')
    elif testimonial.polarity<-0.33:
        sentiment_flag.append('negative')
    else:
        sentiment_flag.append('neutral')
        
    if testimonial.subjectivity>0.66:
        subjectivity_flag.append('subjective')
    elif testimonial.subjectivity<0.33:
        subjectivity_flag.append('objective')
    else:
        subjectivity_flag.append('neutral')
print("The polarity score is a float within the range [-1.0, 1.0]. \nThe subjectivity is a float within the range [0.0, 1.0] \nwhere 0.0 is very objective and 1.0 is very subjective.")            
df['polarity']=pd.Series(polarity)
df['subjectivity']=pd.Series(subjectivity)
df['sentiment_flag']=pd.Series(sentiment_flag)
df['subjectivity_flag']=pd.Series(subjectivity_flag)
df[['polarity','subjectivity','sentiment_flag','subjectivity_flag']].head()


# In[20]:


plt.figure(figsize=(10,8))
res= df.groupby(['abuse','sentiment_flag'])['filtered_text_lower'].count()
res.unstack(level=0).plot(kind='bar')
plt.yabuse("Count")
plt.xabuse("Sentiment")
plt.title("Sentiment in Abuse and NonAbuse\n")
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
res=df.groupby(['abuse'])['polarity','subjectivity'].mean()
res.unstack(level=0).plot(kind='bar')
plt.ylabel("score between 0 to 1")
plt.xlabel("polarity and subjectivity")
plt.title("Polarity and Subjectivity in Abuse and NonAbuse\n")
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
res= df.groupby(['abuse','subjectivity_flag'])['filtered_text_lower'].count()
res.unstack(level=0).plot(kind='bar',color=['green', 'blue'])
large = 24

plt.rc('legend',fontsize=large)
plt.rc('axes',titlesize = 20)
plt.rc('axes',labelsize = 16)
plt.rc('xtick',labelsize = 14)
plt.rc('ytick',labelsize = 14)

plt.ylabel("Score")
plt.xlabel("subjectivity")
plt.title("Subjectivity score in Abuse and NonAbuse\n")
plt.show()


# In[ ]:


import seaborn as sns


sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(10,6)})
res=[x for x in list(df[df.abuse==0].polarity) if str(x) != 'nan']
#sns.distplot(res)
sns.kdeplot(res, shade=True,label="NonAbuse",color='blue');
res=[x for x in list(df[df.abuse==1].polarity) if str(x) != 'nan']
#sns.distplot(res)
sns.kdeplot(res, shade=True,label="Abuse",color='red');

large = 24

plt.rc('legend',fontsize=large)
plt.rc('axes',titlesize = 20)
plt.rc('axes',labelsize = 16)
plt.rc('xtick',labelsize = 20)
plt.rc('ytick',labelsize = 20)



plt.title("KDE distribution plot\nPolarity value in Ham and Spam\n")
plt.xlabel("Polarity value")
plt.ylabel("Probability")
plt.show()


# In[ ]:


#https://github.com/pooji0401/Spam-Tweets-Detection/blob/master/Twitter%20Spam%20Classification.ipynb
#left off at "Top words in positive sentiment"


# In[28]:


#doing something different here to get ngrams/bigrams/trigrams
import re
import unicodedata
import nltk
from nltk.corpus import stopwords

def basic_clean(text):
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
words = basic_clean(''.join(str(df['text'].tolist())))


# In[29]:


words [:20]


# In[30]:


(pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]


# In[31]:


bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())
trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())


# In[39]:


#back to github 


# In[32]:


text_corpus= ''
import string 

for i in df[df.abuse==1][df.sentiment_flag=='positive']['filtered_text_lower']:
    text_corpus += i
    
allWords = nltk.tokenize.word_tokenize(text_corpus)
allWordDist = nltk.FreqDist(w.lower() for w in allWords if w not in list(string.punctuation))

stopwords = nltk.corpus.stopwords.words('english')
allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords )
print("Top words in positive sentiment abuse tweets")
allWordDist.most_common(10)


# In[33]:


text_corpus= ''
for i in df[df.abuse==1][df.sentiment_flag=='negative']['filtered_text_lower']:
    text_corpus += i
    
allWords = nltk.tokenize.word_tokenize(text_corpus)
allWordDist = nltk.FreqDist(w.lower() for w in allWords if w not in list(string.punctuation))

stopwords = nltk.corpus.stopwords.words('english')
allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords )
print("Top words in negative sentiment abuse tweets")
allWordDist.most_common(10)


# In[34]:


text_corpus= ''
for i in df[df.abuse==1][df.sentiment_flag=='neutral']['filtered_text_lower']:
    text_corpus += i
    
allWords = nltk.tokenize.word_tokenize(text_corpus)
allWordDist = nltk.FreqDist(w.lower() for w in allWords if w not in list(string.punctuation))

stopwords = nltk.corpus.stopwords.words('english')
allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords )
print("Top words in neutral sentiment abuse tweets")
allWordDist.most_common(10)


# In[35]:


df.columns


# In[36]:


df[['DIGITS', 'CAP']].describe()


# In[37]:


from scipy import stats
np.set_printoptions(suppress=True)

print("t tests for continuous variables")

list_continous_variables=['DIGITS', 'CAP']

for i in list_continous_variables:
    print("\nVariable name:",i)
    test = stats.ttest_ind(df[i],df['abuse'], equal_var = False)
    #print(test)
    if test.pvalue<0.05:
        print("The variable",i,"is significant, with a t-stat of",test.statistic)
    else:
        print("The variable",i,"is not significant")


# In[38]:


plt.figure(figsize=(10,7))
sns.distplot(df[df.abuse==0].DIGITS,color="black",kde=True,label="Nonabuse");
sns.distplot(df[df.abuse==1].DIGITS,color="red",kde=True,label="Abuse");
plt.xlim(0,15)

large = 24

plt.rc('legend',fontsize=large)
plt.rc('axes',titlesize = 20)
plt.rc('axes',labelsize = 16)
plt.rc('xtick',labelsize = 20)
plt.rc('ytick',labelsize = 20)



plt.title("KDE distribution plot\nNumber of digits in Nonabuse and Abuse\n")
plt.xlabel("Number of digits")
plt.ylabel("Probability")
plt.show()

plt.show()


# In[51]:


plt.figure(figsize=(12,7))
sns.distplot(df[df.abuse==0].CAP,color="black",kde=True)
sns.distplot(df[df.abuse==1].CAP,color="red",kde=True)
plt.xlim(0,10)

large = 24

plt.rc('legend',fontsize=large)
plt.rc('axes',titlesize = 20)
plt.rc('axes',labelsize = 16)
plt.rc('xtick',labelsize = 20)
plt.rc('ytick',labelsize = 20)



plt.title("KDE distribution plot\nNumber of Capitalized words in NonAbuse and Abuse\n")
plt.xlabel("Number of Capitalized words")
plt.ylabel("Probability")
plt.show()

plt.show()

plt.show()


# In[59]:


cols = ['','DIGITS']

for i in range(1,len(cols)):
    plt.figure(figsize=(12,8))
    c = cols[i]
    print("Count plot for","'",cols[i],"' feature, based on Abuse/NonAbuse tweet label")
    sns.countplot(c,hue=df.abuse,data=df)
    plt.show()
    print("\n")


# In[60]:


cols = ['','CAP']

for i in range(1,len(cols)):
    plt.figure(figsize=(12,8))
    c = cols[i]
    print("Count plot for","'",cols[i],"' feature, based on Abuse/NonAbuse tweet label")
    sns.countplot(c,hue=df.abuse,data=df)
    plt.show()
    print("\n")


# In[39]:


def comfusion_matrix_properties(mat):


    from sklearn.metrics import confusion_matrix

    mat = confusion_matrix(y_test,ypred)
    print("Confusion Matrix:\n\n",mat,"\n")

    TP = mat[0][0]
    FP = mat[0][1]
    FN = mat[1][0]
    TN = mat[1][1]


    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("True Positive Rate",round(TPR*100,2),"%")

    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print("True Negative Rate",round(TNR*100,2),"%")

    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("Positive Predictive Value",round(PPV*100,2),"%")

    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative Predictive Value",round(NPV*100,2),"%")

    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("False Positive Rate",round(FPR*100,2),"%")

    # False negative rate
    FNR = FN/(TP+FN)
    print("False Negative Rate",round(FNR*100,2),"%")

    # False discovery rate
    FDR = FP/(TP+FP)
    print("False Discovery Rate",round(FDR*100,2),"%")


    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("Overall Accuracy",round(ACC*100,2),"%")


# In[40]:


def roc_curve_plot(input_probabilities,title):
    large = 24

    plt.rc('legend',fontsize=large)
    plt.rc('axes',titlesize = 20)
    plt.rc('axes',labelsize = 16)
    plt.rc('xtick',labelsize = 14)
    plt.rc('ytick',labelsize = 14)
    #plt.set_facecolor('xkcd:white')
    plt.figure(figsize=(12,9))
    y_pred_proba_dt = input_probabilities


    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_dt)
    auc = metrics.roc_auc_score(y_test, y_pred_proba_dt)

    dt_model_object={}
    dt_model_object['fpr']=fpr
    dt_model_object['tpr']=tpr
    dt_model_object['auc']=auc

    plt.plot(fpr,tpr,label="AUC ="+str(round(auc,2)), lw=3,color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve: \n'+title)
    plt.legend(loc="lower right")
    plt.grid("off")
    plt.show()


# In[63]:


#left off at Modelling and Evaluation


# In[41]:


NB_df = df[['filtered_text_lower','abuse']]
NB_df = NB_df.sample(frac=1)
from sklearn.feature_extraction.text import CountVectorizer

vecfinal = CountVectorizer(min_df=50,max_df=0.8,stop_words="english")
finalX = vecfinal.fit_transform(NB_df['filtered_text_lower'])


# In[42]:


split_limit = round(len(NB_df)/2)
X_train = finalX.toarray()[:split_limit]
X_test = finalX.toarray()[split_limit:]
y_train = NB_df.abuse[:split_limit]
y_test = NB_df.abuse[split_limit:]


# In[43]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score

clf_nb = BernoulliNB()
clf_nb.fit(X_train, y_train)
ypred = clf_nb.predict(X_test)
ypred_NB_model=clf_nb.predict(X_test)
accuracy = f1_score(y_test, ypred, average='weighted')
print("Accuracy for this model: ", accuracy*100,"%")


# In[44]:


print("Naive Bayes Model ")
mat = confusion_matrix(y_test,ypred)
comfusion_matrix_properties(mat)


# In[45]:


large = 24

from sklearn import metrics
plt.rc('legend',fontsize=large)
plt.rc('axes',titlesize = 20)
plt.rc('axes',labelsize = 16)
plt.rc('xtick',labelsize = 14)
plt.rc('ytick',labelsize = 14)
#plt.set_facecolor('xkcd:white')
plt.figure(figsize=(12,9))
y_pred_proba_knn = clf_nb.predict_proba(X_test)[:,1]
variable2=y_pred_proba_knn

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_knn)
auc = metrics.roc_auc_score(y_test, y_pred_proba_knn)

naive_bayes_model_object={}
naive_bayes_model_object['fpr']=fpr
naive_bayes_model_object['tpr']=tpr
naive_bayes_model_object['auc']=auc

plt.plot(fpr,tpr,label="AUC ="+str(round(auc,2)), lw=3,color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve:\nNaive Bayes Model\n')
plt.legend(loc="lower right")
plt.grid("off")
plt.show()


# In[46]:


from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


y_pred_proba_nb=clf_nb.predict_proba(X_test)[:,1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_nb)
average_precision = average_precision_score(y_test, y_pred_proba_nb)

large = 24

plt.rc('legend',fontsize=large)
plt.rc('axes',titlesize = 20)
plt.rc('axes',labelsize = 16)
plt.rc('xtick',labelsize = 14)
plt.rc('ytick',labelsize = 14)

plt.figure(figsize=(12,9))

naive_bayes_model_object['recall']=recall
naive_bayes_model_object['precision']=precision
naive_bayes_model_object['average_precision']=average_precision

plt.step(recall, precision, color='green',label="Average Precision ="+str(round(average_precision,2)),lw=3)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve:\nNaive Bayes Model')
plt.legend(loc="lower left")
plt.grid("off")
plt.show()


# In[47]:



from sklearn.naive_bayes import MultinomialNB
clf_mnb = MultinomialNB()
clf_mnb.fit(X_train, y_train)
ypred = clf_mnb.predict(X_test)
ypred_MultinomialNB_model=clf_mnb.predict(X_test)
accuracy = f1_score(y_test, ypred, average='weighted')
print("Accuracy for this model: ", accuracy*100,"%")


# In[48]:


from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test,ypred)
print("Confusion Matrix:\n\n",mat,"\n")


# In[49]:


comfusion_matrix_properties(mat)


# In[79]:


y = df.abuse
X = df[['DIGITS', 'CAP', 'subjectivity', 'polarity']]
model = X


# In[80]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(model, y, test_size = 0.5,random_state=0)

depth = []
training_score = []
testing_score = []

for i in range(1,25):
    clf_dt = DecisionTreeClassifier(criterion = "gini",random_state=0)
    clf_dt.max_depth = i
    clf_dt.fit(X_train, y_train)
    pred_train = clf_dt.predict(X_train)
    #print("Training accuracy, with",i,"depth :",accuracy_score(pred_train,y_train)*100)
    
    pred_test = clf_dt.predict(X_test)
    #print("Testing accuracy, with",i,"depth :",accuracy_score(pred_test,y_test)*100,"\n")
    
    depth.append(i)
    training_score.append(accuracy_score(pred_train,y_train))
    testing_score.append(accuracy_score(pred_test,y_test))


# In[81]:


plt.figure(figsize=(14,7))
plt.grid()

plt.plot(depth, training_score,c='black')
plt.plot(depth, testing_score,c='red')
plt.scatter(depth, training_score,c='black',marker='*',linewidth=2)
plt.scatter(depth, testing_score,c='red',marker='*',linewidth=2)


# In[82]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X = model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,random_state=0)
    
clf = DecisionTreeClassifier(random_state=0)
 
param_grid = {
              "criterion": ["gini", "entropy"],
              "max_depth": list(np.arange(1,50,1)),
              }
 
CLF = GridSearchCV(estimator = clf, param_grid=param_grid, cv = 5)
CLF.fit(X_train, y_train)


# In[83]:


CLF.best_params_


# In[84]:


clf_dt = DecisionTreeClassifier(criterion=CLF.best_params_['criterion'],max_depth=CLF.best_params_['max_depth'],random_state=0)
clf_dt.fit(X_train,y_train)
ypred = clf_dt.predict(X_test)
ypred_DT_model=clf_dt.predict(X_test)
print("Prediction Accuracy:",accuracy_score(ypred,y_test)*100,"%")


# In[85]:


comfusion_matrix_properties(mat)


# In[86]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X = model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,random_state=0)
    
neigh = KNeighborsClassifier()

k = np.arange(20)+1
param_grid = {'n_neighbors': k}
                                  
CLF = GridSearchCV(estimator = neigh, param_grid = param_grid, cv = 5)
CLF.fit(X_train, y_train)


# In[87]:


CLF.best_params_


# In[88]:


neigh = KNeighborsClassifier(n_neighbors = CLF.best_params_['n_neighbors'])
neigh.fit(X_train,y_train)
ypred = neigh.predict(X_test)
ypred_knn_model=neigh.predict(X_test)
accuracy_score(ypred,y_test)*100


# In[89]:


comfusion_matrix_properties(mat)


# In[90]:



from sklearn.ensemble import RandomForestClassifier
print("RANDOM FOREST CLASSIFIER")

rfc = RandomForestClassifier(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,random_state=0)

rfc.fit(X_train,y_train)
ypred = rfc.predict(X_test)
ypred_rfc_model=rfc.predict(X_test)
print("Prediction Accuracy:",round(accuracy_score(ypred,y_test)*100,2),"%")


mat = confusion_matrix(y_test,ypred)

print(comfusion_matrix_properties(mat))
print()

print("Feature importances:")
print(rfc.feature_importances_)


# In[91]:


for i in range(len(rfc.feature_importances_)):
    print(X_train.columns[i],rfc.feature_importances_[i])


# In[ ]:





# In[ ]:




