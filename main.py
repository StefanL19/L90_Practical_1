import classifiers
from os import listdir
import data_loading
from tqdm import tqdm
import metrics

def generate_predictions(vocab_path, vocab_pos_freq_path, vocab_neg_freq_path, prior_pos_path, prior_neg_path, pos_test, neg_test):
	with open(vocab_path, 'r') as f:
    	vocabulary = f.read().splitlines()

   	with open(vocab_pos_freq_path, 'r') as f:
		vocab_pos_freq = []
   		pos_freq = f.read().splitlines()

    	for item in pos_freq:
    		vocab_pos_freq.append(float(item))

   	with open(vocab_neg_freq_path, 'r') as f:
    	vocab_neg_freq = []
    	neg_freq = f.read().splitlines()

    	for item in neg_freq:
    		vocab_neg_freq.append(float(item))

   	with open(prior_pos_path, 'r') as f:
    	prior_pos = float(f.read().splitlines()[0])

   	with open(prior_neg_path, 'r') as f:
    	prior_neg = float(f.read().splitlines()[0])





#TRAIN_POS_PATH = "data/aclImdb/aclImdb/train/pos"
#TRAIN_NEG_PATH = "data/aclImdb/aclImdb/train/neg"

# Train a new model
stopwords = ["\n"]
TRAIN_POS_PATH = "data/data-tagged/POS"
TRAIN_NEG_PATH = "data/data-tagged/NEG"

# Step 1 Load the data
pos_train, pos_test, neg_train, neg_test = data_loading.load_data(TRAIN_POS_PATH, TRAIN_NEG_PATH, stopwords)

# # Step 2 Train Naive Bayes Classifier on the training data
# vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(pos_train, neg_train, False)

# with open('data/experiments/no_smoothing/vocab.txt', 'w') as f:
#     for item in vocabulary:
#         f.write("%s\n" % item)

# with open('data/experiments/no_smoothing/vocab_pos_freq.txt', 'w') as f:
#     for item in vocab_pos_freq:
#         f.write("%s\n" % item)

# with open('data/experiments/no_smoothing/vocab_neg_freq.txt', 'w') as f:
#     for item in vocab_neg_freq:
#         f.write("%s\n" % item)

# with open('data/experiments/no_smoothing/prior_pos.txt', 'w') as f:
#      f.write(str(prior_pos))

# with open('data/experiments/no_smoothing/prior_neg.txt', 'w') as f:
#      f.write(str(prior_neg))

# # Read data from a saved model
with open('data/experiments/no_smoothing/vocab.txt', 'r') as f:
    vocabulary = f.read().splitlines()

with open('data/experiments/no_smoothing/vocab_pos_freq.txt', 'r') as f:
    vocab_pos_freq = []
    pos_freq = f.read().splitlines()

    for item in pos_freq:
    	vocab_pos_freq.append(float(item))

with open('data/experiments/no_smoothing/vocab_neg_freq.txt', 'r') as f:
    vocab_neg_freq = []
    neg_freq = f.read().splitlines()

    for item in neg_freq:
    	vocab_neg_freq.append(float(item)) 

with open('data/experiments/no_smoothing/prior_pos.txt', 'r') as f:
    prior_pos = float(f.read().splitlines()[0])

with open('data/experiments/no_smoothing/prior_neg.txt', 'r') as f:
    prior_neg = float(f.read().splitlines()[0])

 
# # doc_1 = "i really liked this movie, the performance of the actors was marvelous and enjoyed every second"
# doc = "This movie was really bad. The actors were awful and I can say that was the worst performance I have witnessed in my life"

preds = []
for sample in tqdm(pos_test):
	prediction = classifiers.apply_multinomial_NB(sample, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq)
	preds.append(prediction)

for sample in tqdm(neg_test):
	prediction = classifiers.apply_multinomial_NB(sample, vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq)
	preds.append(prediction)

pos_gt = [1]*len(pos_test)
neg_gt = [0]*len(neg_test)

all_gt = pos_gt + neg_gt

overall_accuracy = metrics.acc(preds, all_gt)
print(overall_accuracy)


