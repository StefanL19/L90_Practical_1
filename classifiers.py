import data_preprocessing
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def train_multinomial_NB(train_pos_data_path, train_neg_data_path):
	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

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
		for w in enumerate(doc):
			for i, word in enumerate(vocab):
				if word == w:
					vocab_pos_freq[i] += 1

	print(vocab_pos_freq)

