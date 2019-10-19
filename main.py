import data_preprocessing
from os import listdir

TRAIN_POS_PATH = "data/aclImdb/aclImdb/train/pos"
TRAIN_NEG_PATH = "data/aclImdb/aclImdb/train/neg"

data_preprocessing.generate_embeddings_generic(1, 2, TRAIN_POS_PATH, TRAIN_NEG_PATH)

