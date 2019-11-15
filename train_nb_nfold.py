import classifiers
from os import listdir, mkdir
import data_loading
from tqdm import tqdm
import metrics
import multiprocessing 
from functools import partial
import numpy as np

feature_combinations = [[True, True, True], 
                        [True, True, False], 
                        [True, False, True],
                        [True, False, False],
                        [False, True, True],
                        [False, True, False]]

for feature_combination in feature_combinations:
    LAPLACE_SMOOTHING=feature_combination[2]
    USE_UNIGRAMS = feature_combinationp[0]
    USE_BIGRAMS = feature_combination[1]

    # Add initial list of stopwords 
    STOPWORDS = []
    with open('stopwords.txt', 'r') as f:
        STOPWORDS = f.read().splitlines()

    # Add the empty line token to the list of stopwords
    STOPWORDS.append("\n")
    print(STOPWORDS)

    TRAIN_POS_PATH = "data/data-tagged/POS/"
    TRAIN_NEG_PATH = "data/data-tagged/NEG/"


    #TRAIN_NEW = True
    OUT_PATH = "data/trained_models/10_fold_no_test/unigram_"+ str(USE_UNIGRAMS).lower() + "_bigram_"+ str(USE_BIGRAMS).lower() +"_laplace_"+ str(LAPLACE_SMOOTHING).lower() +"/val_fold_"

    for TEST_CATEGORY in range(0, 10):
        print("Started iterating for category " + str(TEST_CATEGORY))
        print("-----------------------------------------------")
        OUT_PATH_n = OUT_PATH + str(TEST_CATEGORY) + "/"

        import os
        if not os.path.exists(OUT_PATH_n):
            os.makedirs(OUT_PATH_n)



    #Parameter that will determine which files are we going to use for testing
    #TEST_CATEGORY    = 0


        # Step 1 Load the data
        pos_train, pos_test, neg_train, neg_test = data_loading.load_data_kfold_10_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS, TEST_CATEGORY)

        if TRAIN_NEW:
            print("Training a new model")
            print("-----------------------------------")
         # Step 2 Train Naive Bayes Classifier on the training data
            vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(pos_train, neg_train, USE_UNIGRAMS, USE_BIGRAMS, LAPLACE_SMOOTHING)

        else:
            vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.load_nb_model(OUT_PATH_n)


        top_10_idx = np.argsort(vocab_pos_freq)[-100:]
        top_10_values = [vocabulary[i] for i in top_10_idx]
        print("Top positive words: ", top_10_values)
        print("-------------------------------------------")

        top_10_idx = np.argsort(vocab_neg_freq)[-100:]
        top_10_values = [vocabulary[i] for i in top_10_idx]
        print("Top negative words: ", top_10_values)
        print("-------------------------------------------")

    # Generate the predictions by using a saved model
    #m = multiprocessing.Manager()
    #preds = m.list()
    #with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 40) as pool:
    #        pool.map(partial(classifiers.apply_multinomial_NB, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 1, preds, 1, 2), pos_test)

    #with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 40) as pool:
    #        pool.map(partial(classifiers.apply_multinomial_NB, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 0, preds, 1, 2), neg_test)

    #all_gt = np.array(preds)[:, 0]
    #all_preds = np.array(preds)[:, 1]

    #overall_accuracy = metrics.acc(all_preds, all_gt)
    #print("The overall accuracy of the model is: ", overall_accuracy)

        if TRAIN_NEW:
                classifiers.save_nb_classifier(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, OUT_PATH_n)
