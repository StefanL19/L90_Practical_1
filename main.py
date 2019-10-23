import classifiers
from os import listdir
import data_loading

#TRAIN_POS_PATH = "data/aclImdb/aclImdb/train/pos"
#TRAIN_NEG_PATH = "data/aclImdb/aclImdb/train/neg"
TRAIN_POS_PATH = "data/data-tagged/POS"
TRAIN_NEG_PATH = "data/data-tagged/NEG"
#data_preprocessing.generate_embeddings_generic(1, 2, TRAIN_POS_PATH, TRAIN_NEG_PATH)

# Step 1 Load the data
pos_train, pos_test, neg_train, neg_test = data_loading.load_data(TRAIN_POS_PATH, TRAIN_NEG_PATH)

# Step 2 Train Naive Bayes Classifier on the training data
vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(pos_train, neg_train, False)



# vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(TRAIN_POS_PATH, TRAIN_NEG_PATH, False)

# doc_1 = "i really liked this movie, the performance of the actors was marvelous and enjoyed every second"
# doc = "This movie was really bad. The actors were awful and I can say that was the worst performance I have witnessed in my life"
# classifiers.apply_multinomial_NB(doc_1, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq)
