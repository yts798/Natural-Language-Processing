import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from scipy.sparse import hstack
# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))


def write_output_file(relations, filename = "q3.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "q3.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)


# filter data do not contains 2 or more GPE and PERSON entities
def keep_p_n(data):
    filter_p_n = []
    for record in data:
        keep = 0
        for entities in record['entities']:
            # count number of entities
            if entities['label'] == 'GPE':
                keep += 1
            elif entities['label'] == 'PERSON':
                keep += 1
        if keep == 2:
            filter_p_n.append(record)
            
    return filter_p_n

# extract features for each records.
# extract (people, GPE) entity pair, and corresponding lemma feature, pos feature and dep feature 
def ft_extract(data, ct_name, train = 1):
    # 2 modes for train and test
    # generate ground truth label for train
    if train == 1:
        df = pd.DataFrame(columns=['e1', 'e2', 'lemma', 'pos', 'dep', 'y'])
    else:
        df = pd.DataFrame(columns=['e1', 'e2', 'lemma', 'pos', 'dep'])
        
    # extract feature for each row
    df_row = 0
    for record in data:
        for e_p in record['entities']:
            # for each entities, connect it with all GPE 
            if e_p['label'] == 'PERSON':
                # extract person
                e1 = record['tokens'][e_p['start'] :  e_p['end']]
                # find all GPE
                for e_g in record['entities']:
                    if e_g['label'] == 'GPE':
                        # extract GPE
                        e2 = record['tokens'][e_g['start'] :  e_g['end']]
                        # remove 's in GPE
                        e2_p = []
                        for t in e2:
                            if t != '\'s':
                                e2_p.append(t)

                        # extract words between them
                        if e_p['end'] < e_g['start']:
                            st = e_p['end']
                            ed = e_g['start']
                            
                        else:
                            st = e_g['end']
                            ed = e_p['start']
                        # extract corresponding lemma feature, pos feature and dep feature 
                        lm = record['lemma'][st:ed]
                        pos = record['pos'][st:ed]
                        dep = record['dep'][st:ed]
                        

                        # convert GPE list to text
                        e2_p_text = (' '.join(e2_p)).lower()
                        # check if a valid country name, rather than other GPE 
                        # open counttry names to filter only nation GPE
                        if e2_p_text in ct_name:
                            # remove 's in People
                            e1_p = []
                            for t in e1:
                                if t != '\'s':
                                    e1_p.append(t)
                            # convert Person list to text                                    
                            e1_p_text = (' '.join(e1_p)).lower()
                            # produce label from relation attribute in trainset
                            if train == 1:
                                rel = record['relation']
                                output = 0
                                y = 0 
                                if rel['relation'] == "/people/person/nationality":
                                    if rel['a'] == e1_p and rel['b'] == e2_p:
                                        y = 1
                                df.loc[df_row] = [e1_p_text, e2_p_text, lm, pos, dep, y]
                                    
                                
                            else:
                                df.loc[df_row] = [e1_p_text, e2_p_text, lm, pos, dep]
                            df_row += 1
                 
    return df

def ft_prep(x_up):
    lm_text = []
    pos_text = []
    dep_text = []
    for i in range(len(x_up)):
        lm = x_up.loc[i,'lemma']
        pos = x_up.loc[i,'pos']
        dep = x_up.loc[i,'dep']
        t1 = ' '.join(lm)
        t2 = ' '.join(pos)
        t3 = ' '.join(dep)
        lm_text.append(t1)
        pos_text.append(t2)
        dep_text.append(t3)
        
    return lm_text, pos_text, dep_text
                

# pipeline for prep, testing, optimising, and generate final results

# open counttry names to filter only nation GPE
f = open('country_names.txt', 'r')
temp = f.read().splitlines()
names = []
for line in temp:
    names.append(line.lower())
    
# we first need to filter records containing both people and GPE 
filter_p_n = keep_p_n(train_data)

df_up = ft_extract(filter_p_n, names)

vectorizer_c13 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
vectorizer_c23 = CountVectorizer(analyzer='word', ngram_range=(2, 3))
vectorizer_t13 = TfidfVectorizer(stop_words = 'english', ngram_range=(1, 3))
vectorizer_t14 = TfidfVectorizer(ngram_range=(1, 4))
vectorizer_t23 = TfidfVectorizer(ngram_range=(2, 3))
# extract input 
lm, pos, dep = ft_prep(df_up[['lemma', 'pos', 'dep']])


# extract y_true
y_label = list(df_up['y'])
   

lr = LogisticRegression(C = 0.8, solver = 'lbfgs', penalty = 'l2', class_weight = None, max_iter = 100)

# try features of lemma, pos and lemma, 
# fit
lm_tfidf = vectorizer_t13.fit_transform(lm)
pos_tfidf = vectorizer_t13.fit_transform(pos)
dep_tfidf = vectorizer_t13.fit_transform(dep)
two = hstack((lm_tfidf, pos_tfidf))
three = hstack((two, dep_tfidf))

two, y_label = shuffle(two, y_label)
# three, y_label = shuffle(three, y_label)

# 5 fold cross_validation is used to select appropriate vectorizer and logit hyper parameter
scores = cross_validate(lr, two, y_label, scoring='accuracy', cv=5)
# print("cv_score", np.mean(scores['test_score']))

# generate results for eval_output
lr = lr.fit(two, y_label)
predicted = lr.predict(two)
# print("whole train score", accuracy_score(y_label, predicted))
relations = []

# used to check for eval_output, achieve f1 score above 0.47 by train set
# for i in range(len(predicted)):

#     if predicted[i] == 1:
#         relations.append(tuple([df_up.loc[i]['e1'], df_up.loc[i]['e2']]))
        

# write for whole train
# write_output_file(relations)
# do the same prep for test set
filter_p_n_test = keep_p_n(test_data)
# print(len(filter_p_n))
# print(len(relations))
# print(len(filter_p_n_test))
test_up = ft_extract(filter_p_n_test, names, 0)
lm_test, pos_test, dep_test = ft_prep(test_up[['lemma', 'pos', 'dep']])


# fit tfidf vectorizer with all train test corpurs
vectorizer_t13.fit(lm + lm_test)

lm_tfidf_test = vectorizer_t13.transform(lm_test)
lm_tfidf = vectorizer_t13.transform(lm)

vectorizer_t13.fit(pos + pos_test)

pos_tfidf_test = vectorizer_t13.transform(pos_test)
pos_tfidf = vectorizer_t13.transform(pos)


pos_tfidf_test = vectorizer_t13.transform(pos_test)

two = hstack((lm_tfidf, pos_tfidf))
two_test = hstack((lm_tfidf_test, pos_tfidf_test))


lr = LogisticRegression(C = 0.8, solver = 'lbfgs', penalty = 'l2', class_weight = None, max_iter = 100)
# fit and predict
lr = lr.fit(two, y_label)
predicted_test = lr.predict(two_test)

relations = []
for i in range(len(predicted_test)):

    if predicted_test[i] == 1:
        relations.append(tuple([test_up.loc[i]['e1'], test_up.loc[i]['e2']]))

# print(len(relations))
# write for whole test
write_output_file(relations)