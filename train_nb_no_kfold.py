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
USE_UNIGRAMS = True
USE_BIGRAMS = False

# #Parameter that will determine which files are we going to use for testing
# TEST_CATEGORY  = 9

# Step 1 Load the data
pos_train, pos_test, neg_train, neg_test = data_loading.load_data_simple_train_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS)

# Step 2 Train Naive Bayes Classifier on the training data
vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(pos_train, neg_train, USE_UNIGRAMS, USE_BIGRAMS, LAPLACE_SMOOTHING)


# Generate the predictions by using a saved model
m = multiprocessing.Manager()
preds = m.list()
with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 40) as pool:
    pool.map(partial(classifiers.apply_multinomial_NB, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 1, preds, 1, 1), pos_test)

with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 40) as pool:
    pool.map(partial(classifiers.apply_multinomial_NB, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 0, preds, 1, 1), neg_test)

all_gt = np.array(preds)[:, 0]
all_preds = np.array(preds)[:, 1]

overall_accuracy = metrics.acc(all_preds, all_gt)
print("The overall accuracy of the model is: ", overall_accuracy)


