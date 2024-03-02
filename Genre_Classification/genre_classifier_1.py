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
w2v = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', limit = 50000)
# all_wd = w2v.vocab

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


# convert a document into a vector
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

class LRModel(object):
    def __init__(self, C):
        self.LRModel  = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=C)

    def train(self, X, Y):
        print("--------training--------")
        X_clean = prep(X)
        train_input = []
        count = 0
        for doc in X:
            count += 1
            train_input.append(document_to_vector(doc))
            
            if count % 100 == 0:
                print(count)
            
        self.LRModel.fit(train_input, Y)
        

    
    def test(self, Xtst):
        print("--------testing--------")
        Xtst = prep(Xtst)
        test_input = []
        count = 0
        for doc in Xtst:
            count += 1
            test_input.append(document_to_vector(doc))
            
            if count % 100 == 0:
                print(count)
        Y_test_pred = self.LRModel.predict(test_input)
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

# preprocess
# wd_lst = prep(X[:3])
# print(wd_lst)
# w2v = LoadFastText()
# print(len(w2v['we']))
# print(w2v.__contains__('we'))
print(len(X))
model = LRModel(10)
model.train(X[:500], Y[:500])
Y_test_pred = model.test(Xt[:500])
count = [0, 0, 0, 0]
for i in Y_test_pred:
    count[i] += 1
    
print(count)
# print(Y[0:10])
# preprocess
# predict on the testdata
# Y_test_pred = model.predict(Xt)

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

