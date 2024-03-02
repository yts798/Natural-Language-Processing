import json
import io
import numpy as np
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression

wd_tk = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
w2v = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', limit = 1000)

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# vec = json.load("wiki-news-300d-1M.vec")
# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']

# This is a very poor model which looks for keywords and if none are found it predicts
# randomly according to the class distribution in the training set
class KeywordModel(object):
    def __init__(self):
        self.counts = None

    def fit(self, X, Y, C):
        model = LogisticRegression(C=0.5)
        model.fit(X, Y)

    
    def predict(self, Xtst):
        Y_test_pred = model.predict(Xtst)
        return Y_test_pred

# def LoadFastText():
#     input_file = io.open("wiki-news-300d-1M.vec", 'r', encoding='utf-8', newline='\n', errors='ignore')
#     no_of_words, vector_size = map(int, input_file.readline().split())
#     word_to_vector: Dict[str, List[float]] = dict()
#     for i, line in enumerate(input_file):
#         tokens = line.rstrip().split(' ')
#         word = tokens[0]
#         vector = list(map(float, tokens[1:]))
#         assert len(vector) == vector_size
#         word_to_vector[word] = vector
#     return word_to_vector

def prep(train_x):
    all_wd = []
    for doc in train_x:
        low = doc.lower()
        nop = low.translate(str.maketrans('', '', string.punctuation))
        raw_lst = wd_tk.tokenize(nop)
        wd_lst = [w for w in raw_lst if not w in stop_words]
        all_wd.append(wd_lst)
    return all_wd

# fit the model on the training data
# model = KeywordModel()
# model.fit(X, Y)
# print(len(X))
# print(len(Y))
# print(X[0])
# print(Y[0])
# print(len(docid))
# print(len(Xt))

# preprocess
wd_lst = prep(X[:3])
print(wd_lst)
# w2v = LoadFastText()
print(w2v['we'])
# preprocess
# predict on the testdata
# Y_test_pred = model.predict(Xt)

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
# fout = open("out.csv", "w")
# fout.write("Id,Predicted\n")
# for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
#     fout.write("%d,%d\n" % (i, line))
# fout.close()

