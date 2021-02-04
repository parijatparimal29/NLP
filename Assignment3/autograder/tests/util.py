import os

def calculate_accuracy(gold,predicted):
    readline = lambda line: line.strip().split('/')
    words = []
    groundtruth_tags = []
    predicted_tags = []
    with open(gold,"r") as fgold, open(predicted,"r") as fpred:
        for g, p in zip(fgold, fpred):
            gw, gt = readline(g)
            pw, pt = readline(p)
            if gw == '###':
                continue
            words.append(gw)
            predicted_tags.append(pt)
            groundtruth_tags.append(gt)
    acc = float(sum([pt == gt for gt, pt in zip(groundtruth_tags, predicted_tags)]))/ len(predicted_tags)
    return acc
