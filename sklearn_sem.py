
# coding: utf-8

# ## Importing modules

# In[1]:


import numpy as np
from collections import Counter


# ## Preprocessing the training data

# In[2]:


import string
def preprocess(f_name):
    f=open(f_name, 'r')
    txt1=f.read().translate(str.maketrans("\t\r", "  "))
    #txt1 = txt1.lower()
    "".join(txt1.split())
    txt=txt1.split('\n')
    sentence_corpora=[]
    sentence_labels=[]
    words=[]
    for i in range(0, 32000, 4):
        txt[i]=txt[i].lstrip('0123456789')
        txt[i]=txt[i].replace('\"','')
        txt[i]=txt[i].replace('.','')
        at=str(txt[i].strip())
        for elem in at.split(" "):
            words.append(elem.replace("<e1>","").replace("</e1>", "").replace("</e2>", "").replace("<e2>", "").lower())
        sentence_corpora.append(str(txt[i].strip().replace("<e1>","").replace("</e1>", "").replace("</e2>", "").replace("<e2>", "").lower()))
        sentence_labels.append(str(txt[i+1].strip().replace("(e1,e2)", "").replace("(e2,e1)", "")))
    return sentence_corpora,sentence_labels,words


# In[3]:


sentence_corpora,sentence_labels,words = preprocess("TRAIN_FILE.TXT")
print(type(sentence_corpora))
print(sentence_corpora[:10])
print(len(sentence_corpora))


# In[4]:


#Setting Label values for Softmax Classifier
label_dict={"Cause-Effect": 0, 
            "Instrument-Agency": 1, 
            "Product-Producer": 2, 
            "Content-Container": 3, 
            "Entity-Origin": 4, 
            "Entity-Destination": 5, 
            "Component-Whole": 6,
            "Member-Collection": 7,
            "Message-Topic": 8,
            "Other": 9}
final_labels=[]
for elem in sentence_labels:
    final_labels.append(label_dict[elem])
final_labels = np.array(final_labels)
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab,0)}
print(len(vocab_to_int))


# In[5]:


print(type(final_labels))
print(final_labels[:10])


# ## Using Scikit-Learn's Text Processing functionality

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(sentence_corpora)
X_train_counts.shape


# In[8]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[9]:


print(type(X_train_tfidf))
print(X_train_tfidf[:5])


# ## Preprocessing for the test data

# In[10]:


f=open('TEST_FILE_CLEAN.TXT', 'r')
txt1=f.read().translate(str.maketrans("\t\r", "  "))
txt1 = txt1.lower()
"".join(txt1.split())
txt=txt1.split('\n')
#print(txt[2716])
sentence_test=[]
words=[]
for i in range(0, 2716):
    txt[i]=txt[i].lstrip('0123456789')
    txt[i]=txt[i].replace('\"','')
    txt[i]=txt[i].replace('.','')
    at=str(txt[i].strip())
    for elem in at.split(" "):
        words.append(elem.replace("<e1>","").replace("</e1>", "").replace("</e2>", "").replace("<e2>", ""))
    sentence_test.append(str(txt[i].strip().replace("<e1>","").replace("</e1>", "").replace("</e2>", "").replace("<e2>", "")))
    #labels_test.append(str(txt[i+1].strip().replace("(e1,e2)", "").replace("(e2,e1)", "")))
#print(sentence_test[2715])


# In[11]:


f=open('TEST_FILE_KEY.TXT', 'r')
txt1=f.read().translate(str.maketrans("\t\r", "  "))
"".join(txt1.split())
labels_test=[]
txt=txt1.split('\n')
#print(txt[0:5])
for i in range(0, 2716):
    txt[i]=txt[i].lstrip('0123456789')
    labels_test.append(str(txt[i].strip()))
                       
#print(labels_test[0:5])

final_labels_test=[]
for elem in labels_test:
    final_labels_test.append(label_dict[elem])
final_labels_test = np.array(final_labels_test)


# ## Implementation of Multinomial Naive Bayes Classifier 

# In[12]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, final_labels)


# In[13]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', MultinomialNB(alpha=0.01)),
])
text_clf = text_clf.fit(sentence_corpora, final_labels)


# In[14]:


predicted = text_clf.predict(sentence_test)
np.mean(predicted == final_labels_test)


# ## Using GridSearchCV for obtaining optimum values

# In[15]:


from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3,1e-3),
              'vect__stop_words':('english',None)
}


# In[16]:


gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(sentence_corpora, final_labels)


# In[17]:


gs_clf.best_score_
gs_clf.best_params_


# ## Implementing SVM classifier

# In[18]:


from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf-svm', SGDClassifier(loss='squared_hinge', penalty='l2',
                                           alpha=2*1e-4, n_iter=800, random_state=100,learning_rate='constant',eta0=0.0009)),
 ])
_ = text_clf_svm.fit(sentence_corpora, final_labels)


# ## Storing predictions in output.txt

# In[19]:


f2 = open('output.txt','w')
i=8001
pred = text_clf_svm.predict(sentence_test)
for p in pred:
    for label, val in label_dict.items():
        if val == p:
            #print(label)
            f2.write(str(i))
            f2.write("\t")
            f2.write(label)
            f2.write("\n")
            i=i+1
f2.close()


# ## Calculating Accuracy

# In[20]:


def preprocess_test(f_name):
    f3 = open(f_name,'r')
    data = f3.read()
    data = data.split("\n")
    return data


# In[21]:


output = preprocess_test('output.txt')
test = preprocess_test('TEST_FILE_KEY.TXT')


# In[22]:


count = 0
for i in range(2716):
    if output[i]==test[i]:
        count = count+1    


# ## Final Accuracy

# In[23]:


print(count/2717)


# In[ ]:




