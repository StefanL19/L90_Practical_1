import data_preprocessing
from os import listdir

TRAIN_POS_PATH = "data/aclImdb/aclImdb/train/pos"
TRAIN_NEG_PATH = "data/aclImdb/aclImdb/train/neg"

data_preprocessing.generate_embeddings(TRAIN_POS_PATH, TRAIN_NEG_PATH)

