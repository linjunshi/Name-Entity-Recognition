#!/venv/bin/python3

import pickle
import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt


occupationList = ['manager', 'director', 'administrator', 'president', 'officer', 'examiner', 'controller', 'coordinator', 'consultant', 'engineer', 'directors', 'programmer', 'operator', 'supervisor', 'engineer', 'technician', 'analyst', 'statistician',
'subcontractor', 'editor', 'producer', 'author', 'columnist', 'correspondent', 'planner', 'announcer', 'proofreader', 'publicist', 'reporter', 'translator', 'typesetter', 'writer', 'specialist', 'journalist', 'actor', 'actress', 'instructor',
'designer', 'assistant', 'clerk', 'accountant', 'choreographer', 'comedian', 'dancer', 'artist', 'model', 'musician', 'clerk', 'auditor', 'underwriter', 'bookkeeper', 'consultant', 'servicer', 'planner', 'counselor', 'worker',
'advocate', 'teacher', 'counselor', 'professor', 'counselor', 'general', 'teacher', 'educator', 'Nanny', 'Principal', 'psychologist', 'professor', 'Coach', 'archivist', 'tutor', 'receptionist', 'stenographer', 'expediter', 'typist',
'fighter', 'police', 'sergeant', 'guard', 'general', 'ombudsman', 'paralegal', 'agent', 'secretary', 'reporter', 'attorney', 'cardiologist', 'hygienist', 'chiropractor', 'dentist', 'dietician', 'nurse', 'nutritionist', 'orthodontist',
'pharmacist', 'psychiatrist', 'therapist', 'surgeon', 'veterinarian', 'therapist', 'pediatrician', 'pathologist', 'cashier', 'representative', 'caterer', 'chef', 'cosmetologist', 'inspector', 'hairstylist', 'attendant', 'Stewardess', 'CAO', 'CBO',
'CCO', 'CDO', 'CEO', 'CFO', 'CHO', 'CIO', 'CKO', 'CMO', 'CNO', 'COO', 'CPO', 'CQO', 'CSO', 'CTO', 'CVO', 'minister']


# df[(df['tag']=='PDT') & (df['title']=='O')]
# df.groupby(['word', 'tag']).agg({'sum': [np.sum]})
def get_pd (training_data):
    a = []
    for sentence in training_data:
        for nth_word in range(len(sentence)-1):
            # start of the sentence
            # print("length = %s, nth_word = %s\n\n" % (len(sentence), nth_word))
            if nth_word == 0:
                a.append(['^','^','O']+sentence[nth_word]+sentence[nth_word+1])
            elif nth_word == len(sentence):
                a.append(sentence[nth_word-1]+sentence[nth_word]+['$','$','O'])
            else:
                a.append(sentence[nth_word-1]+sentence[nth_word]+sentence[nth_word+1])

    df = pd.DataFrame(a, columns = ['lword', 'ltag', 'ltitle',
                                    'word', 'tag', 'title',
                                    'rword', 'rtag', 'rtitle'])
    # print(df)
    return df


# (title, false)
# get stats of all words in
def to_binary (token):
    return [token[2] == 'TITLE', token[2] != 'TITLE']


# yes, no
def get_statistics(training_data):
    # get stats, statistics['...'] =
    statistics = {}
    for sentence in training_data:
        # print("%s\n" % sentence)
        for token in sentence:
            if token[1] in statistics:
                statistics[token[1]] = [ x+y for x,y in zip (statistics[token[1]], to_binary(token)) ]
            else:
                statistics[token[1]] = to_binary(token)
    # print (statistics)
    return statistics



def test_lregression (X, Y, coefficient_c=14.0, folds=10):
    test_accu = 0
    train_accu = 0
    folds = 10
    kf = KFold(n=X.shape[0], n_folds=folds, shuffle=True, random_state=42)
    for train, test in kf:
        X_train = X[train]
        X_test = X[test]
        Y_train = Y[train]
        Y_test = Y[test]
        model = LogisticRegression(penalty='l2', C=coefficient_c)
        model.fit(X_train, Y_train)
        # train_error = model.score(X_train, Y_train)
        # test_error = model.score(X_test, Y_test)
        # print("train_accu = %s, test_accu = %s" % (train_error, test_error))
        # a = accuracy_score(Y_test, model.predict(X_test))
        train_accu += f1_score(Y_train, model.predict(X_train))
        test_accu += f1_score(Y_test, model.predict(X_test))

    test_accu /= folds
    train_accu /= folds
    print("C = %s, test_accu = %s, train_accu = %s" % (coefficient_c, test_accu, train_accu))
    return train_accu, test_accu




def get_best_result(X, Y):
    temp = []
    train_error = []
    test_error = []
    coefficient_c = np.arange(1.0, 25.0)
    for c in coefficient_c:
        train, test = test_lregression(X, Y, c)
        train_error.append(train)
        test_error.append((c, test))
    test_error = sorted(test_error, key=lambda x:x[1], reverse=True)
    best = test_error[0][0]
    return best


        # temp.append([coefficient_c/10, train_error, test_error])
    # temp_arr = np.array(temp)
    # plot_train, = plt.plot(coefficient_c, train_error, 'r--', label='training_accuracy')
    # plot_test, = plt.plot(coefficient_c, test_error, 'b--', label='test_accuracy')
    # plt.legend(handles=[plot_train, plot_test])
    # plt.show()


# def distance_to_verb (index, sentence):
#     verb1_found = False
#     for x1 in range(0, index):
#         if sentence[x1][1].find('VB') != -1:
#             break
#
#     verb2_found = False
#     for x2 in range(index, len(sentence)):
#         if sentence[x2][1].find('VB') != -1:
#             break
#
#     dis1 = float("inf")
#     dis2 = float("inf")
#     dis1 = index - x1 if verb1_found else float("inf")
#     dis2 = x2 - index if verb2_found else float("inf")
#
#     return dis1 if dis1 < dis2 else dis2



if __name__ == '__main__':
    data_file = sys.argv[1]
    classifier_path = sys.argv[2]

    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)

    feature_list = []
    Y = []
    for sentence in training_data:
        for nth_word in range(len(sentence)):
            is_occupation = (sentence[nth_word][0].upper() in
                                (x.upper() for x in occupationList))
            temp_dict = {'upper':sentence[nth_word][0][0].isupper(), 'is_occupation':is_occupation}
            if nth_word == 0:
                d = {'ltag':'O', 'mtag':sentence[nth_word][1], 'rtag':sentence[nth_word+1][1],
                    'lword':'^', 'mword':sentence[nth_word][0], 'rword':sentence[nth_word+1][0]}
            elif nth_word == len(sentence)-1:
                d = {'ltag':sentence[nth_word-1][1], 'mtag':sentence[nth_word][1], 'rtag':'O',
                    'lword':sentence[nth_word-1][0], 'mword':sentence[nth_word][0], 'rword':'$'}
            else:
                d = {'ltag':sentence[nth_word-1][1], 'mtag':sentence[nth_word][1], 'rtag':sentence[nth_word+1][1],
                    'lword':sentence[nth_word-1][0], 'mword':sentence[nth_word][0], 'rword':sentence[nth_word+1][0]}
            d.update(temp_dict)
            # print(d)
            feature_list.append(d)
            Y.append(sentence[nth_word][2])
            # start of the sentence
            # print("length = %s, nth_word = %s\n\n" % (len(sentence), nth_word))
    vec = DictVectorizer()
    X = vec.fit_transform(feature_list)
    # preprocessing, make it to be 1 or 0
    lb = preprocessing.LabelBinarizer()
    Y = lb.fit_transform(Y)
    Y = Y.ravel()
    Y = np.array(Y)
    best = get_best_result(X, Y)
    model = LogisticRegression(penalty='l2', C=best)
    model.fit(X, Y)

    with open(classifier_path, 'wb') as f:
        pickle.dump(model, f)
    with open('vec.dat', 'wb') as f:
        pickle.dump(vec, f)

    for sentences in training_data:
        a = [[[word[0], word[1]] for word in sentences]]
    with open('test.dat', 'wb') as f:
        pickle.dump(a, f)











# preprocessing, make it to be 1 or 0
# temp = []
# for index in range(len(Y)):
#     temp.append(int(Y[index] == 'TITLE'))
# print(temp)

# for nth_word in range(len(sentence)-1):
#     d = {}
# # for word in sentence:
#     word = sentence[nth_word]
#     print("length = %s, nth_word = %s\n\n" % (len(sentence), nth_word))
#     if nth_word == 0:
#         d['^' + word[nth_word][0] + word[nth_word+1][0]] = 1
#     elif nth_word == len(sentence):
#         d[sentence[nth_word-1][0] + sentence[nth_word][0] + '$'] = 1
#     else:
#         d[sentence[nth_word-1][0] + sentence[nth_word][0] + word[nth_word+1][0]] = 1
#     feature_list.append(d)
#     # start of the sentence
#     # print("length = %s, nth_word = %s\n\n" % (len(sentence), nth_word))
