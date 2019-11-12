import data_preprocessing
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import multiprocessing 
from functools import partial

def save_nb_classifier(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, out_directory):
    with open(out_directory+'vocab.txt', 'w') as f:
        for item in vocabulary:
            f.write("%s\n" % item)

    with open(out_directory+'vocab_pos_freq.txt', 'w') as f:
        for item in vocab_pos_freq:
            f.write("%s\n" % item)

    with open(out_directory+'vocab_neg_freq.txt', 'w') as f:
        for item in vocab_neg_freq:
            f.write("%s\n" % item)

    with open(out_directory+'prior_pos.txt', 'w') as f:
         f.write(str(prior_pos))

    with open(out_directory+'prior_neg.txt', 'w') as f:
         f.write(str(prior_neg))

    print("Model training finished, the model was saved in directory: ", out_directory)

def load_nb_model(out_path):

    with open(out_directory+'vocab.txt', 'r') as f:
        vocabulary = f.read().splitlines()

    with open(out_directory+'vocab_pos_freq.txt', 'r') as f:
        vocab_pos_freq = []
        pos_freq = f.read().splitlines()

        for item in pos_freq:
            vocab_pos_freq.append(float(item))

    with open(out_directory+'vocab_neg_freq.txt', 'r') as f:
        vocab_neg_freq = []
        neg_freq = f.read().splitlines()

        for item in neg_freq:
            vocab_neg_freq.append(float(item))

    with open(out_directory+'prior_pos.txt', 'r') as f:
        prior_pos = float(f.read().splitlines()[0])

    with open(out_directory+'prior_neg.txt', 'r') as f:
        prior_neg = float(f.read().splitlines()[0])

    return (vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq)

def count_word_occurences(len_vocab, q, vocabulary, doc):
    """
        Generates BoW embeddings for a document based on a vocabulary
    """

    vocab_freq = [0]*len_vocab

    # For each word in the document
    for w in doc:

        # Check if it is in the vocabulary
        for i, word in enumerate(vocabulary):

            # If it is in the vocabulary
            if word == w:

                # Increment the number of its occurences in the positive corpora by 1
                vocab_freq[i] += 1

    q.append(vocab_freq)
    
    print(len(q))



def train_multinomial_NB(train_files_pos, train_files_neg, use_unigrams, use_bigrams, laplace_smoothing=False):

    prior_pos = len(train_files_pos)/(len(train_files_pos) + len(train_files_neg))
    prior_neg = len(train_files_neg)/(len(train_files_pos)+len(train_files_neg))

    train_files = train_files_pos + train_files_neg

    if use_unigrams and use_bigrams:
        print("We are using both unigrams and bigrams")
        # Generating the embeddings by using unigrams and bigrams
        vocab, docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, train_files)

    elif use_bigrams and not use_unigrams:
        print("We are using bigrams but not unigrams")
        vocab, docs_tokenized = data_preprocessing.generate_embeddings_bigrams(train_files)

    elif use_unigrams and not use_bigrams:
        # We are using only unigrams
        print("We are using unigrams but not bigrams")
        vocab, docs_tokenized = data_preprocessing.generate_embeddings_unigrams(train_files)

    print("The total length of the vocabulary is: ", len(vocab))

    pos_docs_tokens = docs_tokenized[:len(train_files_pos)]
    neg_docs_tokens = docs_tokenized[len(train_files_neg):]

    vocab_length = len(vocab)

    m = multiprocessing.Manager()
    pos_list = m.list()
    neg_list = m.list()

    print("Started iterating positive documents")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 40) as pool:
        pool.map(partial(count_word_occurences, vocab_length, pos_list, vocab), pos_docs_tokens)

    print("Started iterating negative documents")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 40) as pool:
        pool.map(partial(count_word_occurences, vocab_length, neg_list, vocab), neg_docs_tokens)
    
    vocab_pos_freq = np.array([0]*len(vocab))
    vocab_neg_freq = np.array([0]*len(vocab))

    for el in pos_list:
        vocab_pos_freq += np.array(el)

    for el in neg_list:
        vocab_neg_freq += np.array(el)

    count_all_pos = np.sum(vocab_pos_freq)
    count_all_neg = np.sum(vocab_neg_freq)


    if laplace_smoothing:
        print("Laplace Smoothing Applied")
        vocab_pos_freq = [(x+1) / (count_all_pos+1) for x in vocab_pos_freq]
        vocab_neg_freq = [(x+1) / (count_all_neg+1) for x in vocab_neg_freq]

    else:
        print("No Laplace Smoothing")
        vocab_pos_freq = [x / count_all_pos for x in vocab_pos_freq]
        vocab_neg_freq = [x / count_all_neg for x in vocab_neg_freq]

    return vocab, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq

def apply_multinomial_NB(vocab, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, gt, all_predictions, min_grams, max_grams, tokens):

    bag = [0] * len(vocab)

    if min_grams == 1 and max_grams == 1:
        #Apply unigrams to the tokens
        augmented_tokens, _ = data_preprocessing.generate_n_grams(1,1,tokens)

    elif min_grams == 1 and max_grams == 2:
        #Apply unigrams and bigrams to the tokens
        augmented_tokens, _ = data_preprocessing.generate_n_grams(1, 2, tokens)

    elif min_grams == 2 and max_grams == 2:
        #Apply bigrams to the tokens 
        augmented_tokens, _ = data_preprocessing.generate_n_grams(2,2,tokens)

    unknown_words = []
    for w in augmented_tokens:
        is_unknown = True
        for i, word in enumerate(vocab):
            if word == w:
                bag[i] = 1
                is_unknown = False

        if is_unknown:
            unknown_words.append(w)

    # print("All words in the document are: ", len(augmented_tokens))
    # print("The unknown words in the document were: ", len(unknown_words))
    # print("-------------------------------------------------------------")

    score_pos = np.log(prior_pos)
    score_neg = np.log(prior_neg)

    bag = np.array(bag)
    vocab_pos_freq = np.array(vocab_pos_freq)
    vocab_neg_freq = np.array(vocab_neg_freq)

    rel_scores_pos = np.multiply(bag, vocab_pos_freq)
    rel_scores_neg = np.multiply(bag, vocab_neg_freq)

    features_pos = []
    features_neg = []
    for i, v in enumerate(bag):
        if v == 1:
            features_pos.append(rel_scores_pos[i])
            features_neg.append(rel_scores_neg[i])

    final_pos = score_pos + np.sum(np.log(features_pos))
    final_neg = score_neg + np.sum(np.log(features_neg))

    if final_pos > final_neg:
        all_predictions.append([gt, 1])

    else:
        all_predictions.append([gt, 0])

    print(len(all_predictions))




