import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

wd_tk = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
w2v = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', limit = 50000)
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


# fit the model on the training data

def document_to_vector(wd_lst):
    all_vec = []
    wd_len = 1
    
    for wd in wd_lst:
        if w2v.__contains__(wd):
            all_vec.append(w2v[wd])
    l = len(all_vec)  
    if l > 0:
        wd_len = l
    
#     print(l)
#     print(len(wd_lst))
    vec  = [0] * 300
    for i in range(300):
        for one_vec in all_vec:
            vec[i] += one_vec[i]
        vec[i] /= wd_len
            
    return vec

def prep(train_x):
    all_wd = []
    for doc in train_x:
        low = doc.lower()
        nop = low.translate(str.maketrans('', '', string.punctuation))
        raw_lst = wd_tk.tokenize(nop)
        wd_lst = [w for w in raw_lst if not w in stop_words]
        all_wd.append(wd_lst)
    return all_wd

# predict on the test data
# tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
# tfidfvectorizer.fit(X)
# tfx = tfidfvectorizer.transform(X)
# tfxt = tfidfvectorizer.transform(Xt)


# model = DecisionTreeClassifier(random_state=1, criterion="gini", max_depth=5)
model = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=1, max_iter = 1000)

print("--------training--------")
X_clean = prep(X)
train_input = []
count = 0
for doc in X:
    count += 1
    train_input.append(document_to_vector(doc))
            
    if count % 100 == 0:
        print(count)
            
model.fit(train_input, Y)

pd = self.LRModel.predict(train_input)
print(pd)
print(metrics.accuracy_score(Y, pd))
        
        
        
print("--------testing--------")
Xtst = prep(Xt)
test_input = []
count = 0
for doc in Xt:
    count += 1
    test_input.append(document_to_vector(doc))
            
    if count % 100 == 0:
        print(count)
Y_test_pred = model.predict(test_input)
print(Y_test_pred)    

# model.fit(tfx, Y)
# Y_test_pred = model.predict(tfxt)
# print(set(Y_test_pred))
print(Y_test_pred[:100])
# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

