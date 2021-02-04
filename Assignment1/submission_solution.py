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
    return dict(collections.Counter(ex['sentence1'] + ex['sentence2']))
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    s1, s2 = ex['sentence1'], ex['sentence2']
    feat = {'{}_1_{}_2'.format(w1, w2) : 1 for w1 in s1 for w2 in s2}
    overlap = len([w in s1 for w in s2]) / float(len(s2))
    feat['overlap'] = overlap
    return feat
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
    train_examples = [(feature_extractor(ex), ex['gold_label']) for ex in train_data]
    valid_examples = [(feature_extractor(ex), ex['gold_label']) for ex in valid_data]
    weights = {}
    for n in range(num_epochs):
        random.shuffle(train_examples)
        for feat, label in train_examples:
            scale = (label - predict(weights, feat)) * learning_rate
            increment(weights, feat, scale)
        predictor = lambda x : 1 if dot(weights, x) > 0 else 0
        train_err = evaluate_predictor(train_examples, predictor)
        valid_err = evaluate_predictor(valid_examples, predictor)
        print('epoch={} train_err={} valid_err={}'.format(n, train_err, valid_err))
    return weights
    # END_YOUR_CODE