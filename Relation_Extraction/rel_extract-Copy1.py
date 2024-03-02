import json
import pandas as pd
import numpy as np
#import spacy
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from scipy.sparse import hstack
# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

def print_example(data, index):
    """Prints a single example from the dataset. Provided only
    as a way of showing how to access various fields in the
    training and testing data.

    Args:
        data (list(dict)): A list of dictionaries containing the examples 
        index (int): The index of the example to print out.
    """
    # NOTE: You may Delete this function if you wish, it is only provided as 
    #   an example of how to access the data.
    
    # print the sentence (as a list of tokens)
    print("Tokens:")
    print(data[index]["tokens"])

    # print the entities (position in the sentence and type of entity)
    print("Entities:")
    for entity in data[index]["entities"]:
        print("%d %d %s" % (entity["start"], entity["end"], entity["label"]))
    
    # print the relation in the sentence if this is the training data
    if "relation" in data[index]:
        print("Relation:")
        relation = data[index]["relation"]
        print("%d:%s %s %d:%s" % (relation["a_start"], relation["a"],
            relation["relation"], relation["b_start"], relation["b"]))
    else:
        print("Test examples do not have ground truth relations.")

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

# print a single training example
# print("Training example:")
# print_example(train_data, 0)
# print_example(train_data, 1)

# print("---------------")
# print("Testing example:")
# print a single testing example
# the testing example does not have a ground
# truth relation
# print_example(test_data, 2)

def keep_p_n(data):
    filter_p_n = []
    for record in data:
        keep = 0
        for entities in record['entities']:
            if entities['label'] == 'GPE':
                keep += 1
            elif entities['label'] == 'PERSON':
                keep += 1
        if keep == 2:
            filter_p_n.append(record)
            
    return filter_p_n
 


# add relation
def rel_add(model, data):
    rel = []
    for one in data:
        e1 = one[0]
        e2 = one[1]
        if model.predict(one):
            relation.append(tuple(e1, e2))
        
    return rel


# extract features between people and GPE
def ft_extract(data, ct_name, train = 1):
    if train == 1:
        df = pd.DataFrame(columns=['e1', 'e2', 'lemma', 'pos', 'dep', 'y'])
    else:
        df = pd.DataFrame(columns=['e1', 'e2', 'lemma', 'pos', 'dep'])
    df_row = 0
    for record in data:
        for e_p in record['entities']:
            if e_p['label'] == 'PERSON':
                e1 = record['tokens'][e_p['start'] :  e_p['end']]
                for e_g in record['entities']:
                    if e_g['label'] == 'GPE':
                        e2 = record['tokens'][e_g['start'] :  e_g['end']]
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
                        # extract lm list
                        lm = record['lemma'][st:ed]
                        pos = record['pos'][st:ed]
                        dep = record['dep'][st:ed]
                        
                        # check if a valid country name, rather than other GPE
                        # convert list to text
                        e2_p_text = (' '.join(e2_p)).lower()
                        if e2_p_text in ct_name:
                            # clean names, removed 's
                            e1_p = []
                            for t in e1:
                                if t != '\'s':
                                    e1_p.append(t)
                            e1_p_text = (' '.join(e1_p)).lower()
                            if train == 1:
                                rel = record['relation']
                                output = 0
                                y = 0 
                                if rel['relation'] == "/people/person/nationality":
                                    if rel['a'] == e1_p and rel['b'] == e2_p:
#                                         print("find one!")
                                        y = 1
                                df.loc[df_row] = [e1_p_text, e2_p_text, lm, pos, dep, y]
                                    
                                
                            else:
                                df.loc[df_row] = [e1_p_text, e2_p_text, lm, pos, dep]
                            df_row += 1
#                         else:
#                             print(e2_p_text)
                 
    return df
                
# def label_rel(data):
#     y = []
#     for i in data:
#         if check_nationality(pair, re)


# print(test_data[0])

#TODO: build a training/validation/testing pipeline for relation extraction
#       then write the list of relations extracted from the *test set* to "q3.csv"
#       using the write_output_file function.

# print(train_data[0])
# i=2
# print(train_data[i]['tokens'])
# print(train_data[i]['entities'])
# print(train_data[i]['relation'])
# for i in range(20):
#     print(train_data[i]['entities'])
# relations = []

# read in the ground truth training data
# rel = set()
# for d in train_data:
#     # ignore relations that are not nationality
#     if d["relation"]["relation"] != "/people/person/nationality":
#         continue
    
#     # get the person and the gpe joined as a string and in lower case
#     person = ' '.join(d["relation"]["a"]).lower()
#     gpe = ' '.join(d["relation"]["b"]).lower()
#     rel.add((person, gpe))
    
# print(rel)

# def check_nationality(rel):
#     if rel["relation"] == "/people/person/nationality":
#         if(pair[0] == rel['a'] and pair[1] == rel['b']):
#             return True
#     return False


# we first need to filter records containing both people and GPE 
f = open('country_names.txt', 'r')
temp = f.read().splitlines()
names = []
for line in temp:
    names.append(line.lower())
    

filter_p_n = keep_p_n(train_data)
# print(len(test_data))
# print(len(filter_p_n))

# print(filter_p_n[0]['entities'])
# print(filter_p_n[0]['relation'])
# for i in range(5):
# print(filter_p_n[0:3])
# un-preprocessed dataframe
df_up = ft_extract(filter_p_n, names)


print(df_up)

# print(filter_p_n[0:20])
# for i in filter_p_n[0:50]:
#     print(i['relation'])
# print(df_up.loc[5]['lemma'])
# print(df_up.loc[5]['pos'])
# print(df_up.loc[5]['dep'])
# my_name = []

     
        
# print(count)

le = preprocessing.LabelEncoder()

vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 3))
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
# extract input 
x_up = df_up[['lemma', 'pos', 'dep']]
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
  
   


# print(len(pos_text[0]))
# print(dep_text.shape)
# print(x_up['lemma_text'])
# print(x_up)
y_label = list(df_up['y'])
# print(y_label)




# lm_text = []
# for lm_list in x_up['lemma']:
   
#     text = ' '.join(lm_list)
#     lm_text.append(text)
        
# tfidf = TfidfVectorizer()
# lm_tfidf = tfidf.fit_transform(lm_text)


# print(lm_tfidf)

lm_tfidf = vectorizer.fit_transform(lm_text)
pos_tfidf = vectorizer.fit_transform(pos_text)
dep_tfidf = vectorizer.fit_transform(dep_text)
lr = LogisticRegression()
# print(lm_tfidf.shape)
# print(y_label)
# lr.fit(lm_tfidf, y_label)
print(pos_tfidf.shape)
# two = np.hstack([lm_tfidf, lm_tfidf])
two = hstack((lm_tfidf, pos_tfidf))
three = hstack((two, dep_tfidf))
print(three.shape)
# three, y_label = shuffle(three, y_label)
two, y_label = shuffle(two, y_label)
scores = cross_validate(lr, two, y_label, scoring='accuracy', cv=5)
print(scores)

lr = lr.fit(two, y_label)
predicted = lr.predict(two)
print(accuracy_score(y_label, predicted))
count = 0
relations = []
for i in range(len(predicted)):

    if predicted[i] == 1:
        relations.append(tuple([df_up.loc[i]['e1'], df_up.loc[i]['e2'],]))
        count+=1
        
print(count)
# print(lm_tfidf[0])
# pos_le = le.fit_transform(df_up['pos'])
# print(pos_le)



# print(set(my_name))
# print(filter_p_n[0:20])
# rel = {'a': ['Hokusai'], 'b': ['Japan'], 'a_start': 13, 'b_start': 23, 'relation': '/people/person/nationality'}
# p1 =[['Hokusai'], ['Japan']]
# print(check_nationality(p1, rel))
# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
# TODO: remove this and write out the relations you extracted (obviously don't hard code them)
# relations = [
#     ('Hokusai', 'Japan'), 
#     ('Hans Christian Andersen', 'Denmark')
#     ]
write_output_file(relations)

