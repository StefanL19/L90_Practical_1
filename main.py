import classifiers
from os import listdir

TRAIN_POS_PATH = "data/aclImdb/aclImdb/train/pos"
TRAIN_NEG_PATH = "data/aclImdb/aclImdb/train/neg"

#data_preprocessing.generate_embeddings_generic(1, 2, TRAIN_POS_PATH, TRAIN_NEG_PATH)

vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(TRAIN_POS_PATH, TRAIN_NEG_PATH, False)

doc_1 = "i really liked this movie, the performance of the actors was marvelous and enjoyed every second"
doc = "This movie was really bad. The actors were awful and I can say that was the worst performance I have witnessed in my life"
classifiers.apply_multinomial_NB(doc_1, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq)
