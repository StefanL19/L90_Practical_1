from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm

def tokenize_text(text):
	words = re.sub("[^\w]", " ",  text).split()
	tokenized = [w.lower() for w in words]
	return tokenized

# Initial implementation for unigram
def generate_embeddings(train_pos_data_path, train_neg_data_path):
	all_words = []
	tokenized_docs = []

	train_files = [f for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	
	for train_file in tqdm(train_files[:1000]):
		file_full_path = train_pos_data_path+"/"+train_file
		with open(file_full_path) as f:
			text = f.read()
			tokenized_text = tokenize_text(text)
			tokenized_docs.append(tokenized_text)

			all_words.extend(tokenized_text)

	# Remove all words that repeat
	vocab = sorted(list(set(all_words)))
	
	print("There are {0} document with {1} unique unigrams".format(len(tokenized_docs), len(vocab)))

	bag_vectors = []

	for doc in tqdm(tokenized_docs):
		bag = [0] * len(vocab)

		for w in doc:
			for i, word in enumerate(vocab):
				if word == w:
					bag[i] += 1

		bag_vectors.append(bag)

	# print(vocab)
	# print(bag_vectors[0])





