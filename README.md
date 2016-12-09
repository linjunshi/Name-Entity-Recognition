# Name-Entity-Recognition

A program that recognize if an input word is a title. It takes in a large training data set which every word has a class label indicating if this word is or is part of a title. Logistic regression in sklearn package is used.

To get training data (where training data is the path of the provided training dataset file):
with open(training_data, ’rb’) as f:
    training_set = pickle.load(f)

