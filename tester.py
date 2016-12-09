
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


occupationList = ['manager', 'director', 'administrator', 'president', 'officer', 'examiner', 'controller', 'coordinator', 'consultant', 'engineer', 'directors', 'programmer', 'operator', 'supervisor', 'engineer', 'technician', 'analyst', 'statistician',
'subcontractor', 'editor', 'producer', 'author', 'columnist', 'correspondent', 'planner', 'announcer', 'proofreader', 'publicist', 'reporter', 'translator', 'typesetter', 'writer', 'specialist', 'journalist', 'actor', 'actress', 'instructor',
'designer', 'assistant', 'clerk', 'accountant', 'choreographer', 'comedian', 'dancer', 'artist', 'model', 'musician', 'clerk', 'auditor', 'underwriter', 'bookkeeper', 'consultant', 'servicer', 'planner', 'counselor', 'worker',
'advocate', 'teacher', 'counselor', 'professor', 'counselor', 'general', 'teacher', 'educator', 'Nanny', 'Principal', 'psychologist', 'professor', 'Coach', 'archivist', 'tutor', 'receptionist', 'stenographer', 'expediter', 'typist',
'fighter', 'police', 'sergeant', 'guard', 'general', 'ombudsman', 'paralegal', 'agent', 'secretary', 'reporter', 'attorney', 'cardiologist', 'hygienist', 'chiropractor', 'dentist', 'dietician', 'nurse', 'nutritionist', 'orthodontist',
'pharmacist', 'psychiatrist', 'therapist', 'surgeon', 'veterinarian', 'therapist', 'pediatrician', 'pathologist', 'cashier', 'representative', 'caterer', 'chef', 'cosmetologist', 'inspector', 'hairstylist', 'attendant', 'Stewardess', 'CAO', 'CBO',
'CCO', 'CDO', 'CEO', 'CFO', 'CHO', 'CIO', 'CKO', 'CMO', 'CNO', 'COO', 'CPO', 'CQO', 'CSO', 'CTO', 'CVO', 'minister']


if __name__ == '__main__':
    data_file = sys.argv[1]
    model_file = sys.argv[2]
    path_to_results = sys.argv[3]

    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open('vec.dat', 'rb') as f:
        vec = pickle.load(f)

    print(training_data)

    feature_list = []
    # Y = []
    for sentence in training_data:
        for nth_word in range(len(sentence)):
            is_occupation = (sentence[nth_word][0].upper()
                            in
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
            feature_list.append(d)

    X = vec.transform(feature_list)
    predict = model.predict(X)
    index = 0
    a = []
    for sentence in training_data:
        b = []
        for word in sentence:
            b.append([word[0], 'TITLE' if predict[index] == 1 else 'O'])
            index += 1
        a.append(b)

    with open(path_to_results, 'wb') as f:
        pickle.dump(a, f)

    # print(a)
