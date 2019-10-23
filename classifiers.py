import data_preprocessing
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import multiprocessing 
from functools import partial

def count_word_occurences(len_vocab, q, vocabulary, doc):
	vocab_freq = [0]*len_vocab

	# For each word in the document
	for w in doc:

		# Check if it is in the vocabulary
		for i, word in enumerate(vocabulary):

			# If it is in the vocabular
			if word == w:

				# Increment the number of its occurences in the positive corpora by 1
				vocab_freq[i] += 1
	q.append(vocab_freq)
	
	print(len(q))


def train_multinomial_NB(train_files_pos, train_files_neg, laplace_smoothing=False):

	prior_pos = len(train_files_pos)/(len(train_files_pos) + len(train_files_neg))
	prior_neg = len(train_files_neg)/(len(train_files_pos)+len(train_files_neg))

	train_files = train_files_pos + train_files_neg
	
	# Generating the embeddings
	vocab, docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, train_files)

	pos_docs_tokens = docs_tokenized[:len(train_files_pos)]
	neg_docs_tokens = docs_tokenized[len(train_files_neg):]

	vocab_length = len(vocab)

	m = multiprocessing.Manager()
	pos_list = m.list()
	neg_list = m.list()

	print("Started iterating positive documents")
	with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
		pool.map(partial(count_word_occurences, vocab_length, pos_list, vocab), pos_docs_tokens)

	print("Started iterating negative documents")
	with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
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
		print("Laplace Smootinng Applied")
		vocab_pos_freq = [(x+1) / (count_all_pos+1) for x in vocab_pos_freq]
		vocab_neg_freq = [(x+1) / (count_all_neg+1) for x in vocab_neg_freq]

	else:
		print("No Laplace Smoothing")
		vocab_pos_freq = [x / count_all_pos for x in vocab_pos_freq]
		vocab_neg_freq = [x / count_all_neg for x in vocab_neg_freq]

	return vocab, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq

def apply_multinomial_NB(tokens, vocab, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq):

	bag = [0] * len(vocab)

	for w in tokens:
		for i, word in enumerate(vocab):
			if word == w:
				bag[i] = 1

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
		return 1

	else:
		return 0
	# print(final_pos)
	# print(final_neg)
	# print("-------------------------")


