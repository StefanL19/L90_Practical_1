from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import collections
import operator
import numpy as np

def collect_train_data(train_data_files, stopwords):
	all_docs = []
	for file in tqdm(train_data_files):
		with open(file, "r") as f:
			text = f.readlines()
			all_tokens = []
			for token in text:
				token = token.split("\t")
				all_tokens.append(token[0])

			# Removing the new line token
			for i, token in enumerate(all_tokens):
				if token in stopwords:
					del all_tokens[i]

		all_docs.append(all_tokens)

	return all_docs

def load_data_kfold_10(train_pos_data_path, train_neg_data_path, stopwords, test_category):

	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos.sort(key = lambda x: x.split("_")[0].replace("cv", ""))
	train_files_neg.sort(key = lambda x: x.split("_")[0].replace("cv", ""))

	print("The count of all positive files is: ", len(train_files_pos))
	print("The count of all negative files is: ", len(train_files_neg))

	all_pos_docs = collect_train_data(train_files_pos, stopwords)
	all_neg_docs = collect_train_data(train_files_neg, stopwords)

	pos_train = []
	neg_train = []
	pos_test = []
	neg_test = []
	remove_pos = []
	remove_neg = []

	for idx, pos_entry in enumerate(all_pos_docs):
		# If the index falls into the test category
		if (idx%10) == test_category:
			# Add the entry to the positive test set
			pos_test.append(pos_entry)
		else:
			# Add the entry to the positive train set
			pos_train.append(pos_entry)


	# Do the same for the negative set
	for idx, neg_entry in enumerate(all_neg_docs):
		# If the index falls into the test category
		if (idx%10) == test_category:
			# Add the entry to the positive test set
			neg_test.append(neg_entry)
		else:
			neg_train.append(neg_entry)

	for i in remove_neg:
		del all_neg_docs[i]

	print("The size of the positive training set is: ", len(pos_train))
	print("The size of the negative training set is: ", len(neg_train))
	print("The size of the positive test set is: ", len(pos_test))
	print("The size of the negative test set is: ", len(neg_test))
	print(pos_test[0])
	return all_pos_docs, pos_test, all_neg_docs, neg_test
