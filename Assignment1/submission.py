import json
import collections
import argparse
import random

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    words = ex.get('sentence1') + ex.get('sentence2')
    unique_words = set(words)
    bow = {}
    for word in unique_words:
        bow[word] = words.count(word)
    return bow
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    words1 = ex.get('sentence1')
    words2 = ex.get('sentence2')
    #lower_words = [word.lower() for word in words]
    unique_words = set(words1+words2)
    bow = {}
    for word in unique_words:
        bow[word] = words1.count(word)+words2.count(word)*2
    return bow
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    features = []
    for data in train_data:
        features += data.get('sentence1') + data.get('sentence2')
    weight = {}
    for f in features:
        weight[f] = 0.0
    for epoch in range(num_epochs):
        for data in train_data:
            X = feature_extractor(data)
            y = data.get('gold_label')
            loss = (y - predict(weight,X))
            for x in X:
                gradient = X.get(x)*loss
                weight[x] = weight[x] + learning_rate*gradient
    return weight
    # END_YOUR_CODE
