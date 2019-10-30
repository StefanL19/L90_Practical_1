
#  In Python, an individual
#  * document should look like this:
#  * >> (<label>, [(<feature>, <value>), ...]) */
# static int unpack_document(

import data_loading
import data_preprocessing
from tqdm import tqdm

def generate_BoW_vectors(doc_files):
   bow_embeddings = []
   for doc in tqdm(doc_files):

   # Put the data in the format [(<label>, [(<feature>, <value>), ...]), ...]
   doc_embedding  = [0]*len(vocab)
   
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

   return embeddings


train_pos_path = "data/data-tagged/POS/"
train_neg_path = "data/data-tagged/NEG/"
stopwords = ["\n"]
split = 0
pos_train, pos_test, neg_train, neg_test = data_loading.load_data_kfold_10(train_pos_path, train_neg_path, stopwords, split)

train_files = pos_train + neg_train

print(len(train_files))
vocab, docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, train_files)

# Generate SVM Embeddings
pos_train_embeddings = generate_BoW_vectors(train_files[:len(pos_train)])
pos_train_embeddings_format = convert_embeddings_to_svm_format(pos_train_embeddings, 1)


neg_train_embeddings = generate_BoW_vectors(train_files[len(pos_train):])
neg_train_embeddings_format  = convert_embeddings_to_svm_format(neg_train_embeddings, 0)

test_files = pos_test + neg_test
_, test_docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, test_files)

pos_test_embeddings = generate_BoW_vectors(test_files[len(pos_test):])
pos_test_embeddings_format = convert_embeddings_to_svm_format(pos_test_embeddings, 1)

neg_test_embeddings = generate_BoW_vectors(test_files[:len(pos_test)])
neg_test_embeddings_format = convert_embeddings_to_svm_format(neg_test_embeddings, 0)

