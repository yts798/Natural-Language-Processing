import json
import numpy as np
import pandas as pd
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
w2v = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', limit = 100000)
from sklearn.tree import DecisionTreeClassifier

wd_tk = TreebankWordTokenizer()
# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']


# This is a very poor model which looks for keywords and if none are found it predicts
# randomly according to the class distribution in the training set
# class KeywordModel(object):
#     def __init__(self):
#         self.counts = None

#     def fit(self, X, Y):
#         # fit the model
#         # normally you would want to use X the training data but this simple model doesn't need it
#         self.counts = np.array(np.bincount(Y), dtype=np.float32)
#         self.counts /= np.sum(self.counts)
    
#     def predict(self, Xin):
#         Y_test_pred = []
#         for x in Xin:
#             # split into words
#             xs = x.lower().split()

#             # check if for our keywords
#             if "scary" in xs or "spooky" in xs or "raven" in xs: # horror book
#                 Y_test_pred.append(0)
#             elif "science" in xs or "space" in xs: # science fiction book
#                 Y_test_pred.append(1)
#             elif "funny" in xs or "embarrassed" in xs: # humor book
#                 Y_test_pred.append(2)
#             elif "police" in xs or "murder" in xs or "crime" in xs: # crime fiction book
#                 Y_test_pred.append(3)
#             else: 
#                 Y_test_pred.append(np.random.choice(len(self.counts), p=self.counts)) # predict randomly
#         return Y_test_pred

def count_class(Y):
    tally = [0] * 4
    for i in Y:
        tally[i] += 1
    
    print("The number for each class is:", tally[0], tally[1], tally[2], tally[3])
    
#     return word_to_vector

# convert a document into a vector
def lst_to_vec(wd_lst):
    all_vec = []
    wd_len = 1
    for wd in wd_lst:
        
        
        if w2v.__contains__(wd):
            all_vec.append(w2v[wd])
#         else:
#             print(wd)
    
    
    count = len(all_vec)
    vec  = [0] * 300
    if count == 0:
        return vec
    
    for one_vec in all_vec:
        vec = np.add(vec, one_vec)
        
    vec = np.divide(vec, count)   
    return vec



def prep(X):
    all_wd = []
    for text in X:
        re = text.lower()
        re = re.translate(str.maketrans('', '', string.punctuation))
        re = wd_tk.tokenize(re)
        re = lst_to_vec(re)
        all_wd.append(list(re))
    return all_wd
# fit the model on the training data
# model = KeywordModel()
# model.fit(X, Y)

# # predict on the test data
# Y_test_pred = model.predict(Xt)

# print(X[0])
# print(Xt[0])
# print(len(X))
# print(len(docid))
# print(len(Xt))
# # 
# # print(Y[0:1000])
# count_class(Y[0:500])
# count_class(Y[0:1000])
# count_class(Y)


print("loading fasttext done")
train_frac, val_frac = 0.8, 0.2
train_end = int(train_frac*len(X))

X_train = X[0:train_end]
Y_train = Y[0:train_end]
X_val = X[train_end:]
Y_val = Y[train_end:]
docid_train = docid[0:train_end]
docid_val = docid[train_end:]

Y_test_pred = []
lr = DecisionTreeClassifier(criterion="entropy", max_depth=1000)

# solver='newton-cg'
X_train = prep(X_train)
# print(len(X_train))
# print(X_train[0])
print("embedding done")
# print(count_class(Y_train)
# a = X_train[0]
# b = X_train[1]
# print(X_train[0])
# print(Y_train[0])
# print(X_train[1])

# print(X_train[1])

# X_train = [list[i] for i in zip(X_train, docid_train)]
# print(len(X_train), len(docid_train))
# print(X_train[0])
lr = lr.fit(X_train, Y_train)
print("training done")
X_val = prep(X_val)

re = lr.predict(X_train)
print("validation done")
count_class(Y_train)
count_class(re)
print(f1_score(Y_train, re, average='macro'))
print(accuracy_score(Y_train, re))
re = lr.predict(X_val)
print(f1_score(Y_val, re, average='macro'))
print(accuracy_score(Y_val, re))
count_class(Y_val)
count_class(re)

X_train_new = X_train + X_val
Y_train_new = Y_train + Y_val
print(len(X_train_new))
print(len(X_train_new))
lr = lr.fit(X_train_new, Y_train_new)
Xt = prep(Xt)
print(len(Xt))
Y_test_pred = lr.predict(Xt)   
count_class(Y_test_pred)

# f1_score(y_true, y_pred, average='macro')

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

