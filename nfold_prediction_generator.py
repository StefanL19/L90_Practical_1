import classifiers
from os import listdir, mkdir
import data_loading
from tqdm import tqdm
import metrics
import multiprocessing 
from functools import partial
import numpy as np
from threading import Thread

def predict(unigrams, bigrams, laplace_smoothing, stopwords):
	OUT_PATH = "data/trained_models/10_fold_no_test/unigram_"+str(unigrams).lower()+"_bigram_"+str(bigrams).lower()+"_laplace_"+str(laplace_smoothing).lower()+"_stopwords_"+str(stopwords).lower()+"/val_fold_"

	print("Started Generating Predictions for Model: ", OUT_PATH)
	print("-----------------------------------------------------")

	stop_words = []

	if stopwords == True:
		with open('stopwords.txt', 'r') as f:
			stop_words = f.readlines()
	
	STOPWORDS = stop_words + ["\n"]

	n_grams_begin = 1
	n_grams_end = 2

	if unigrams == False:
		n_grams_begin = 2
	if bigrams == False:
		n_grams_end = 1


	TRAIN_POS_PATH = "data/data-tagged/POS/"
	TRAIN_NEG_PATH = "data/data-tagged/NEG/"

	for fold in range(0,10):
		print("Iterating for fold "+str(fold))
		print("---------------------------")
		OUT_PATH = OUT_PATH+str(fold)+"/"

		vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.load_nb_model(OUT_PATH)

		#pos_train, pos_test, neg_train, neg_test = data_loading.load_data_simple_train_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS)
		pos_train, pos_test, neg_train, neg_test = data_loading.load_data_kfold_10_test(TRAIN_POS_PATH, TRAIN_NEG_PATH, STOPWORDS, fold)

		predictions = []

		print("Started doing positive predictions")
		for entry in tqdm(pos_test):
			pred_pos = classifiers.apply_multinomial_NB_slow(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, n_grams_begin, n_grams_end, entry)
			predictions.append(pred_pos)

		print("Started doing negative predictions")
		for entry in tqdm(neg_test):
			pred_neg = classifiers.apply_multinomial_NB_slow(vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq, n_grams_begin, n_grams_end, entry)
			predictions.append(pred_neg)




		import os
		if not os.path.exists("data/trained_models/predictions/cross_fold/"+str(fold)+"/"):
			os.makedirs("data/trained_models/predictions/cross_fold/"+str(fold)+"/")

		save_path = "data/trained_models/predictions/cross_fold/"+str(fold)+"/unigram_"+str(unigrams).lower()+"_bigram_"+str(bigrams).lower()+"_laplace_"+str(laplace_smoothing).lower()+"_stopwords_"+str(stopwords).lower()+".txt"
		np.savetxt(save_path,np.array(predictions))

if __name__ == '__main__':
    Thread(target = predict, args=(True, True, True, True,)).start()
    Thread(target = predict, args=(True, True, True, False,)).start()
    Thread(target = predict, args=(True, True, False, True,)).start()
    Thread(target = predict, args=(True, True, False, False,)).start()
    Thread(target = predict, args=(True, False, True, True,)).start()
    Thread(target = predict, args=(True, False, True, False,)).start()
    Thread(target = predict, args=(True, False, False, True,)).start()
    Thread(target = predict, args=(True, False, False, False,)).start()
    Thread(target = predict, args=(False, True, True, True,)).start()
    Thread(target = predict, args=(False, True, True, False,)).start()
    Thread(target = predict, args=(False, True, False, True,)).start()
    Thread(target = predict, args=(False, True, False, False,)).start()


