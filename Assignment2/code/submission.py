from util import *
from model import *
import numpy as onp
import itertools
from mxnet import np, gluon

def hamming_loss(gold_seqs, pred_seqs):
    """Return the average hamming loss of pred_seqs.
    Useful function: np.equal(x1, x2)
    Parameters:
        gold_seqs : (batch_size, seq_len)
        pred_seqs : (batch_size, seq_len)
    Return:
        loss : float
    """
    return 1. - np.equal(gold_seqs, pred_seqs).sum() / gold_seqs.shape[0] / gold_seqs.shape[1]

def greedy_decode(scores):
    """Decode sequence for independent classification models.
    Parameters:
        scores : (batch_size, seq_len, vocab_size)
    Returns:
        labels : (batch_size, seq_len)
    """
    return np.max(scores, axis=-1), np.argmax(scores, axis=-1)

def score_sequence(seqs, unigram_scores, bigram_scores):
    """Compute score of the sequence:
        \sum_{t} unigram_score(s[t]) + bigram_score(s[t-1], s[t])
    Parameters:
        seqs : (batch_size, seq_len)
        unigram_scores : (batch_size, seq_len, num_labels)
            score of unigrams s[t]
        bigram_scores : (batch_size, seq_len, num_labels, num_labels)
            score of bigrams s[t], s[t-1].
            Note: coordinate 2 corresponds to s[t] and coordinate 3 correspond to s[t-1]
    Returns:
        scores : (batch_size,)
    """
    batch_size, seq_len = seqs.shape
    prev_scores = unigram_scores[np.arange(batch_size), 0, seqs[:, 0]]
    for i in range(1, seqs.shape[1]):
        prev_scores = prev_scores + unigram_scores[np.arange(batch_size), i, seqs[:, i]] + bigram_scores[np.arange(batch_size), i, seqs[:, i], seqs[:, i - 1]]
    return prev_scores

def bruteforce_decode(unigram_scores, bigram_scores):
    batch_size, seq_len, num_labels = unigram_scores.shape
    seq_scores = None  # (num_seqs, batch_size)
    seqs = None  # (num_seqs, seq_len)
    for seq in itertools.product(range(num_labels), repeat=seq_len):
        seq = np.array(seq)
        score = score_sequence(np.broadcast_to(seq, (batch_size, seq_len)),
                    unigram_scores, bigram_scores)
        seq_scores = score if seq_scores is None else np.vstack((seq_scores, score))
        seqs = seq if seqs is None else np.vstack((seqs, seq))
    return np.max(seq_scores, axis=0), seqs[np.argmax(seq_scores, axis=0)]

def viterbi_decode(scores):
    """Implement Viterbi decoding.
    Your result should match what returned by bruteforce_decode.
    Parameters:
        scores : (unigram_scores, bigram_scores)
            unigram_scores : (batch_size, seq_len, num_labels)
                batch_id, time_step, curr_symbol
                score of current symbol
            bigram_scores : (batch_size, seq_len, num_labels, num_labels)
                batch_id, time_step, curr_symbol, prev_symbol
                score of previous symbol followed by current symbol
    Returns:
        scores : (batch_size,)
        labels : (batch_size, seq_len,)
    """
    unigram_scores, bigram_scores = scores
    # BEGIN_YOUR_CODE
    batch_size, seq_len, num_labels = unigram_scores.shape
    final_score = np.zeros((batch_size))
    final_seq = None
    for batch in range(0, batch_size):
        pi = np.zeros((seq_len,num_labels))
        tag = np.zeros((seq_len,num_labels))
        for j in range(1,seq_len):
            for t in range(0,num_labels):
                scores = None
                for i in range(0,num_labels):
                    prev = 0
                    if(j==1):
                        prev = (unigram_scores[batch,0,i])
                    score = (bigram_scores[batch,j,t,i]) + (unigram_scores[batch,j,t]) + pi[j-1,i] + prev
                    scores = score if scores is None else np.vstack((scores, score))
                pi[j,t] =  np.max(scores.T)
                tag[j-1,t] = np.argmax(scores.T)
        a = np.max(pi[-1].T)
        tag[-1,-1] = np.argmax(pi[-1].T)
        chseq = np.zeros((seq_len))
        chosen = tag[-1,-1]
        chseq[seq_len-1] = chosen
        for k in range(1,seq_len):
            chseq[seq_len-1-k] = tag[seq_len-1-k,chosen]
            chosen = tag[seq_len-1-k,chosen]
        chseq = np.expand_dims(chseq,0)
        final_score[batch] = a
        final_seq = chseq if final_seq is None else np.vstack((final_seq, chseq))
    return final_score, final_seq
    # END_YOUR_CODE

def bruteforce_normalizer(unigram_scores, bigram_scores):
    batch_size, seq_len, num_labels = unigram_scores.shape
    seq_scores = None  # (num_seqs, batch_size)
    for seq in itertools.product(range(num_labels), repeat=seq_len):
        seq = np.array(seq)
        score = score_sequence(np.broadcast_to(seq, (batch_size, seq_len)),
                    unigram_scores, bigram_scores)
        seq_scores = score if seq_scores is None else np.vstack((seq_scores, score))
    a = logsumexp(seq_scores.T)
    return a

def compute_normalizer(unigram_scores, bigram_scores):
    """Compute the normalizer (partition function) in CRF's loss function.
    Your result should match what returned by bruteforce_normalizer.
    Parameters:
        scores : (unigram_scores, bigram_scores)
            unigram_scores : (batch_size, seq_len, num_labels)
                batch_id, time_step, curr_symbol
                score of current symbol
            bigram_scores : (batch_size, seq_len, num_labels, num_labels)
                batch_id, time_step, curr_symbol, prev_symbol
                score of previous symbol followed by current symbol
    Returns:
        normalizer : (batch_size,)
    """
    # BEGIN_YOUR_CODE
    batch_size, seq_len, num_labels = unigram_scores.shape
    final_score = np.zeros((batch_size))
    for batch in range(0,batch_size):
        pi = np.zeros((seq_len,num_labels))
        for j in range(1,seq_len):
            for t in range(0,num_labels):
                scores = None
                for i in range(0,num_labels):
                    prev = 0
                    if(j==1):
                        prev = (unigram_scores[batch,0,i])
                    score = (bigram_scores[batch,j,t,i]) + (unigram_scores[batch,j,t]) + pi[j-1,i] + prev
                    scores = score if scores is None else np.vstack((scores, score))
                max_score = np.max(scores)
                scores = scores - max_score
                lse_val = max_score + logsumexp(scores.T)
                lse_val = lse_val.asnumpy()
                pi[j,t] =  lse_val
        max_pi = np.max(pi[-1])
        pi_sub = pi[-1] - max_pi
        a = max_pi + logsumexp(pi_sub.T)
        final_score[batch] = a
    return final_score
    # END_YOUR_CODE

def crf_loss(scores, y):
    """Compute the loss for the CRF model.
    You can use score_sequence and compute_normalizer.
    Parameters:
        scores : (unigram_scores, bigram_scores)
        y : (batch_size, seq_len)
            gold sequence
    """
    unigram_scores, bigram_scores = scores
    gold_seq_score = score_sequence(y, unigram_scores, bigram_scores)
    normalizer = compute_normalizer(unigram_scores, bigram_scores)
    loss = normalizer - gold_seq_score
    return loss


