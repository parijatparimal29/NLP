import argparse
import math
import collections
import numpy as np

def train_and_test(train_file, dev_file, test_file):
    f = open(train_file, "r")
    g = open(dev_file,"r")
    dev = g.read()
    ex = f.read() + dev
    emission_score = dict(collections.Counter(ex.split()))
    input_seq = ex.split()
    transition_score = {}

    for i in range(1,len(input_seq)):
        curr_tag = (input_seq[i-1].split('/'))[1]+'/'+(input_seq[i].split('/'))[1]
        if(curr_tag in transition_score):
            transition_score[curr_tag] = transition_score[curr_tag] + 1
        else:
            transition_score[curr_tag] = 1
    tag_dict = {}
    word_dict = {}
    for x in emission_score:
        if((x.split('/'))[1] in tag_dict):
            tag_dict[(x.split('/'))[1]] += emission_score[x]
        else:
            tag_dict[(x.split('/'))[1]] = emission_score[x]
    for x in emission_score:
        if((x.split('/'))[0] in word_dict):
            word_dict[(x.split('/'))[0]] += emission_score[x]
        else:
            word_dict[(x.split('/'))[0]] = emission_score[x]

    lambda1 = 0.01
    for x in word_dict:
        for y in tag_dict:
            curr = x+'/'+y
            if(curr in emission_score):
                emission_score[curr] += lambda1
            else:
                emission_score[curr] = lambda1
            tag_dict[y] += lambda1
            word_dict[x] += lambda1
            
    for x in emission_score:
        emission_score[x] = emission_score[x]/tag_dict[(x.split('/'))[1]]

    tag_tag = {}
    for x in transition_score:
        if((x.split('/'))[0] in tag_tag):
            tag_tag[(x.split('/'))[0]] += transition_score[x]
        else:
            tag_tag[(x.split('/'))[0]] = transition_score[x]

    lambda2 = 0.99
    for x in tag_tag:
        for y in tag_tag:
            curr = x+'/'+y
            if(curr in transition_score):
                transition_score[curr] += lambda2
            else:
                transition_score[curr] = lambda2
            tag_tag[y] += lambda2

    for x in transition_score:
        transition_score[x] /= tag_tag[(x.split('/'))[0]]
    
    vocabulary = []
    for key in tag_tag.keys():
        k = key
        vocabulary.append(k)
    test(test_file, vocabulary, emission_score, transition_score)

def test(test_file, vocabulary, emission_score, transition_score):
    f = open(test_file, "r")
    ex2 = f.read()
    ex = ex2.split()
    pi = np.zeros((len(vocabulary),len(ex)+1))
    for i in range(0,len(vocabulary)):
        combo = (ex[0].split('/'))[0]+'/'+vocabulary[i]
        combo2 = '###/'+vocabulary[i]
        emit = 0
        if(combo in emission_score):
            emit = np.log(emission_score[combo])
        trans = np.log(transition_score[combo2])
        pi[i,0] = emit + trans
    for i in range(1,len(ex)):
        for j in range(0,len(vocabulary)):
            combo = (ex[i].split('/'))[0]+'/'+vocabulary[j]
            emit = 1
            if(combo in emission_score):
                emit =  emission_score[combo]
            max_score = -float("inf")
            for k in range(0,len(vocabulary)):
                combo = vocabulary[k]+'/'+vocabulary[j]
                score = np.log(emit) + np.log(transition_score[combo]) + pi[k,i-1]
                max_score = max(score, max_score)
            pi[j,i] = max_score
            
    x = np.argmax(pi,0)
    """
    #Accuracy check
    error = 0
    for i in range(0,len(result)):
        if(result[i] - x[i]!=0.):
            error+=1
    accuracy = (1 - (error/len(result)))*100
    print(accuracy)
    """
    out = ""
    for i in range(0,len(ex)):
        out += ex[i].split('/')[0]+'/'+vocabulary[x[i]]+'\n'
    p = open("output.txt","w")
    p.write(out)
    p.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='training dataset name', default='identity')
    parser.add_argument('--dev', help='test dataset name', default='identity')
    parser.add_argument('--test', help='test dataset name', default='identity')
    args = parser.parse_args()
    train_and_test(args.train, args.dev, args.test)
   