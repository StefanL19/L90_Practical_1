import classifiers
from os import listdir, mkdir
import data_loading
from tqdm import tqdm
import metrics
import multiprocessing 
from functools import partial
import numpy as np

LAPLACE_SMOOTHING=True
STOPWORDS = ["\n"]
TRAIN_POS_PATH = "data/data-tagged/POS/"
TRAIN_NEG_PATH = "data/data-tagged/NEG/"
USE_UNIGRAMS = False
USE_BIGRAMS = True

TRAIN_NEW = True
OUT_PATH = "data/trained_models/no_fold_unigram_false_bigram_true_laplace_true/"

import os
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# #Parameter that will determine which files are we going to use for testing
# TEST_CATEGORY  = 9

# Step 1 Load the data
pos_train, pos_test, neg_train, neg_test = data_loading.load_data_simple_train_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS)

if TRAIN_NEW:
	# Step 2 Train Naive Bayes Classifier on the training data
	vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(pos_train, neg_train, USE_UNIGRAMS, USE_BIGRAMS, LAPLACE_SMOOTHING)

else:
	vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.load_nb_model(OUT_PATH)


top_10_idx = np.argsort(vocab_pos_freq)[-100:]
top_10_values = [vocabulary[i] for i in top_10_idx]
print("Top positive words: ", top_10_values)
print("-------------------------------------------")

top_10_idx = np.argsort(vocab_neg_freq)[-100:]
top_10_values = [vocabulary[i] for i in top_10_idx]
print("Top negative words: ", top_10_values)
print("-------------------------------------------")

# Generate the predictions by using a saved model
m = multiprocessing.Manager()
preds = m.list()
with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 1) as pool:
    pool.map(partial(classifiers.apply_multinomial_NB, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 1, preds, 2, 2), pos_test)

with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 1) as pool:
    pool.map(partial(classifiers.apply_multinomial_NB, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 0, preds, 2, 2), neg_test)

all_gt = np.array(preds)[:, 0]
all_preds = np.array(preds)[:, 1]

overall_accuracy = metrics.acc(all_preds, all_gt)
print("The overall accuracy of the model is: ", overall_accuracy)

if TRAIN_NEW:
	classifiers.save_nb_classifier(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, OUT_PATH)


