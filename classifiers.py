import data_preprocessing
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

def train_multinomial_NB(train_pos_data_path, train_neg_data_path,laplace_smoothing=False):
	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos = train_files_pos[:100]
	train_files_neg = train_files_neg[:100]

	print(len(train_files_pos))
	print(len(train_files_neg))

	prior_pos = len(train_files_pos)/(len(train_files_pos) + len(train_files_neg))
	prior_neg = len(train_files_neg)/(len(train_files_pos)+len(train_files_neg))

	train_files = train_files_pos + train_files_neg

	vocab, docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, train_files)

	pos_docs_tokes = docs_tokenized[:len(train_files_pos)]
	neg_docs_tokens = docs_tokenized[len(train_files_neg):]

	vocab_pos_freq = [0]*len(vocab)
	vocab_neg_freq = [0]*len(vocab)

	print("Started iterating positive documents")
	for doc in tqdm(pos_docs_tokes):
		for w in doc:
			for i, word in enumerate(vocab):
				if word == w:
					vocab_pos_freq[i] += 1

	count_all_pos = sum(vocab_pos_freq)

	print("Started iterating negative documents")
	for doc in tqdm(neg_docs_tokens):
		for w in doc:
			for i, word in enumerate(vocab):
				if word == w:
					vocab_neg_freq[i] += 1

	count_all_neg = sum(vocab_neg_freq)

	if laplace_smoothing:
		print("Using laplace laplace_smoothing")
		vocab_pos_freq = [(x+1) / (count_all_pos+1) for x in vocab_pos_freq]
		vocab_neg_freq = [(x+1) / (count_all_neg+1) for x in vocab_neg_freq]

	else:
		vocab_pos_freq = [x / count_all_pos for x in vocab_pos_freq]
		vocab_neg_freq = [x / count_all_neg for x in vocab_neg_freq]

	return vocab, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq

def apply_multinomial_NB(doc, vocab, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq):
	tokens = data_preprocessing.tokenize_text(doc)

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

	print(final_pos)
	print(final_neg)


	#print(rel_scores_neg)

	# print(score_pos)
	# # print(np.where(rel_scores_pos != 0.0))
	# # print(vocab[177])
	# print(np.sum(rel_scores_pos))
	# rel_scores_pos = np.sum(np.log(rel_scores_pos))
	# rel_scores_neg = np.sum(np.log(rel_scores_neg))

	# final_pos_score = score_pos + rel_scores_pos

	# final_neg_score = score_neg + rel_scores_neg

	# print(final_pos_score)
	# print(final_neg_score)







