from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import collections
import operator

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

def load_data(train_pos_data_path, train_neg_data_path, stopwords, test_start_index=900, test_end_index=1000):

	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos.sort(key = lambda x: x.split("_")[0].replace("cv", ""))
	train_files_neg.sort(key = lambda x: x.split("_")[0].replace("cv", ""))

	print("The count of all positive files is: ", len(train_files_pos))
	print("The count of all negative files is: ", len(train_files_neg))

	all_pos_docs = collect_train_data(train_files_pos, stopwords)
	all_neg_docs = collect_train_data(train_files_neg, stopwords)

	#Get the test data
	pos_test = all_pos_docs[test_start_index:test_end_index]
	neg_test = all_neg_docs[test_start_index:test_end_index]

	# Remove the test data from all documents
	del all_pos_docs[test_start_index:test_end_index]
	del all_neg_docs[test_start_index:test_end_index]

	# The remaining documents are the training set

	#pos_test = all_pos_docs[training_end_index:]

	#neg_train = all_neg_docs[training_start_index:training_end_index]
	#neg_test = all_neg_docs[training_end_index:]

	return all_pos_docs, pos_test, all_neg_docs, neg_test

