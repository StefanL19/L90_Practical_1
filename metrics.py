import numpy as np
import math
import scipy
import mpmath

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def acc(results, ground_truth):
    errors = 0
    for i, res in enumerate(results):
        if res != ground_truth[i]:
            errors += 1

    all_items = len(results)
    correct_preds = all_items - errors

    return correct_preds / all_items

def sign_test(preds_system_A, preds_system_B, ground_truth, q=0.5):
    ties = 0
    plus = 0
    minus = 0

    for i, gt in enumerate(ground_truth):
        if preds_system_A[i] == preds_system_B[i]:
            ties += 1
            continue

        else:
            if preds_system_A[i] == gt:
                plus += 1
                continue

            else:
                minus += 1
                continue

    all_samples = len(ground_truth)
    k = np.ceil(ties/2) + min(plus, minus)

    N = 2 * np.ceil(ties/2) + plus + minus

    print("The k is: ", k)
    print("The N is: ", N)
    res = 0
    for idx in range(0, int(k)):
        res += (nCr(N, idx)*pow(q, idx)*pow((1-q), (N-idx)))

    return (2*res)

def sign_test_precision(preds_system_A, preds_system_B, ground_truth, q=0.5):
    ties = 0
    plus = 0
    minus = 0

    for i, gt in enumerate(ground_truth):
        if preds_system_A[i] == preds_system_B[i]:
            ties += 1
            continue

        else:
            if preds_system_A[i] == gt:
                plus += 1
                continue

            else:
                minus += 1
                continue

    all_samples = len(ground_truth)
    k = np.ceil(ties/2) + min(plus, minus)

    N = 2 * np.ceil(ties/2) + plus + minus

    print("The k is: ", k)
    print("The N is: ", N)

    res = 0
    for idx in range(0, int(k)):
        res += (mpmath.binomial(N, idx)*mpmath.power(q,idx)*mpmath.power((1-q), (N-idx)))
        #res += (nCr(N, idx)*pow(q, idx)*pow((1-q), (N-idx)))

    return (2*res)

def vocabulary_stats(vocab_path):
    count_bigrams = 0
    count_unigrams = 0

    with open(vocab_path, 'r') as f:
        vocabulary = f.read().splitlines()

    for word in vocabulary:
        if len(word.split(" ")) == 2:
            count_bigrams += 1
        else :
            count_unigrams += 1

    return count_unigrams, count_bigrams
    # print("All unigrams in the vocabulary: ", count_unigrams)
    # print("All bigrams in the vocabulary: ", count_bigrams)




