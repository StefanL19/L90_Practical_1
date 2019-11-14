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

for fold in range(0,10):
	print("Iterating for fold "+str(fold))
	print("---------------------------")
	OUT_PATH = "data/trained_models/10_fold_no_test/unigram_true_bigram_true_laplace_true_val_fold_"+str(fold)+"/"

	vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.load_nb_model(OUT_PATH)

	#pos_train, pos_test, neg_train, neg_test = data_loading.load_data_simple_train_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS)
	pos_train, pos_test, neg_train, neg_test = data_loading.load_data_kfold_10_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS, fold)

	predictions = []

	print("Started doing positive predictions")
	for entry in tqdm(pos_test):
		pred_pos = classifiers.apply_multinomial_NB_slow(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 1, 2, entry)
		predictions.append(pred_pos)

	print("Started doing negative predictions")
	for entry in tqdm(neg_test):
		pred_neg = classifiers.apply_multinomial_NB_slow(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, 1, 2, entry)
		predictions.append(pred_neg)




	save_path = "data/trained_models/predictions/cross_fold/"+str(fold)+"/unigram_true_bigram_true_laplace_true.txt"
	np.savetxt(save_path,np.array(predictions))

