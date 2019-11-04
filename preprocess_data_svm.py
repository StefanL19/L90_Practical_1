
#  In Python, an individual
#  * document should look like this:
#  * >> (<label>, [(<feature>, <value>), ...]) */
# static int unpack_document(

import data_loading
import data_preprocessing
from tqdm import tqdm
import pandas as pd
import multiprocessing
from classifiers import count_word_occurences
from functools import partial 

def generate_BoW_vectors(doc_files):
   bow_embeddings = []
   for doc in tqdm(doc_files):

   # Put the data in the format [(<label>, [(<feature>, <value>), ...]), ...]
      doc_embedding = [0]*len(vocab)
      
      for w in doc:
         # Check if it is in the vocabulary
         for i, word in enumerate(vocab):
            if word == w:
               doc_embedding[i] = 1

      bow_embeddings.append(doc_embedding)
   return bow_embeddings


def convert_embeddings_to_svm_format(embeddings, doc_class):
   """
      embeddings: a list of vectors
   """

   embeddings_formatted = []

   for embedding in embeddings:
      embedding_formatted = (doc_class, [])
      for idx, feature_value in enumerate(embedding):
         embedding_formatted[1].append((idx, feature_value))

      embeddings_formatted.append(embedding_formatted)

   return embeddings_formatted


train_pos_path = "data/data-tagged/POS/"
train_neg_path = "data/data-tagged/NEG/"
stopwords = ["\n"]
split = 0

# Preprocess and split the data into train and test sets
pos_train, pos_test, neg_train, neg_test = data_loading.load_data_kfold_10(train_pos_path, train_neg_path, stopwords, split)

# Files that will be used for training
train_files = pos_train + neg_train
print(len(train_files))

# Generate n-grams for embeddings
vocab, docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, train_files)
pos_docs_tokenized_train = docs_tokenized[:len(pos_train)]
neg_docs_tokenized_train = docs_tokenized[len(pos_train):]

m = multiprocessing.Manager()
pos_list = m.list()
neg_list = m.list()
pos_list_test = m.list()
neg_list_test = m.list()

vocab_length = len(vocab)

print("Started iterating positive documents")
with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
   pool.map(partial(count_word_occurences, vocab_length, pos_list, vocab), pos_docs_tokenized_train)

print("Started iterating negative documents")
with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
   pool.map(partial(count_word_occurences, vocab_length, neg_list, vocab), neg_docs_tokenized_train)

# Generate training docs SVM Embeddings and save them to a dataframe in a csv file
pos_train_embeddings_format = convert_embeddings_to_svm_format(pos_list, 1)
pos_train_embeddings_format.to_csv('pos_train_split_0.csv', index=False, header=False)

neg_train_embeddings_format = convert_embeddings_to_svm_format(neg_list, 0)
neg_train_embeddings_format.to_csv('neg_train_split_0.csv', index=False, header=False)


# test_files = pos_test + neg_test
# _, test_docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, test_files)

# pos_test_embeddings = generate_BoW_vectors(test_files[len(pos_test):])
# pos_test_embeddings_format = convert_embeddings_to_svm_format(pos_test_embeddings, 1)

# neg_test_embeddings = generate_BoW_vectors(test_files[:len(pos_test)])
# neg_test_embeddings_format = convert_embeddings_to_svm_format(neg_test_embeddings, 0)



