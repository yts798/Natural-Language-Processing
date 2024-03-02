import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import TreebankWordTokenizer



# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}

# global treebanktokenizer
wd_tk = TreebankWordTokenizer()
all_wd = w2v.keys()

# convert a document into a vector
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    # tokenize the input document
    wd_lst = wd_tk.tokenize(doc)
    all_vec = []
    
    # get the vector for each word
    for wd in wd_lst:
        # ignore words that is not in dict
        if wd in all_wd:
            all_vec.append(w2v[wd])
            
    wd_len = len(all_vec)
    
    # aggregate the vectors of words in the input document
#     the len of one vector is 300
    vec  = [0] * 300
    for one_vec in all_vec:
        vec = np.add(vec, one_vec)
    vec = np.divide(vec, wd_len) 
#     print(vec)
    return vec
#     vec  = [0] * 300
#     for i in range(300):
#         for one_vec in all_vec:
#             vec[i] += one_vec[i]
#         vec[i] /= wd_len
#     print(vec)
#     return vec
            

# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # convert each of the training documents into a vector
#     train_input = []
#     count = 0
#     for doc in Xtr:
#         count += 1
#         train_input.append(document_to_vector(doc))
#         if count % 1000 == 0:
#             print(count)
        
    #TODO: train the logistic regression classifier
    model = LogisticRegression(C=C, max_iter = 1000)
    model.fit(Xtr, Ytr)
    return model

# fit a linear model 
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    # convert each of the testing documents into a vector
#     test_input = []
    
#     for doc in Xtst:
#         test_input.append(document_to_vector(doc))
    
    # test the logistic regression classifier and calculate the accuracy
    re = model.predict(Xtst)

    score = accuracy_score(Ytst, re)
    return score

def save_agg_vec(X, name):
    print(len(X))
    train_input = []
    count = 0
    for doc in X:
        count += 1
        train_input.append(document_to_vector(doc))
        if count % 1000 == 0:
            print(count)
#     print(train_input)
#     print(len(train_input))
    df = pd.DataFrame(train_input)
    df.to_csv(name, index = False) 

def word_embedding(X):
    print(len(X))
    train_input = []
    count = 0
    for doc in X:
        count += 1
        train_input.append(document_to_vector(doc))
        if count % 1000 == 0:
            print(count)
    return train_input

# # save_agg_vec(X_train, "tr.csv")
# xt = pd.read_csv("tr.csv")
# print(len(X_train), len(xt))
# # save_agg_vec(X_val, "v.csv")
# xv = pd.read_csv("v.csv")
# print(len(X_val), len(xv))
# # save_agg_vec(X_test, "t.csv")
# t = pd.read_csv("t.csv")
# print(len(X_test), len(t))

# search for the best C parameter using the validation set
# c_values = [c/10 for c in range(1, 16, 1)]
# ts = []
# vs = []
# for c in c_values:
#     print("c value is:", c)
#     model = fit_model(xt, Y_train, c)
#     c += 0.1
#     train_score = test_model(model, xt, Y_train)
#     val_score = test_model(model, xv, Y_val)
# #     ts.append(train_score)
#     print(train_score)
#     print(val_score)
#     ts.append(train_score)
#     vs.append(val_score)

# print(ts)
# print(vs)

X_train = word_embedding(X_train[0:2000])
X_val = word_embedding(X_val[0:500])
X_test = word_embedding(X_test[0:500])

c = 16

# new_xt = xt + xv
# new_yt = Y_train + Y_val

# print(len(new_xt))

# print(len(new_yt))
model = fit_model(X_train, Y_train[0:2000], c)
new_test_score = test_model(model, X_test, Y_test[0:500])
print(new_test_score)

# print(xt.values.tolist())
# print(xt)
# # search for the best C parameter using the validation set
# c_values = [c/10 for c in range(1, 16, 1)]
# ts = []
# vs = []
# for c in c_values:
#     print("c value is:", c)
#     model = fit_model(X_train, Y_train, c)
#     c += 0.1
# #     train_score = test_model(model, X_train[0:500], Y_train[0:500])
#     val_score = test_model(model, X_val, Y_val)
# #     ts.append(train_score)
#     vs.append(val_score)

#     score = test_model(model, X_test[0:500], Y_test[0:500])
#     print(score)
# print(ts)
# print(vs)
# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result


    
    

# print(model.predict(document_to_vector(X_val[0])), Y_val[0])
# print(document_to_vector(X_val[0]))
# print(Y_val[0])
