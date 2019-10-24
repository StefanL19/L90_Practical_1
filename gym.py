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


    # Generate the predictions by using a saved model
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

def train_model(train_pos_path, train_neg_path, stopwords, laplace_smoothing, out_directory):
    # Step 1 Load the data
    pos_train, pos_test, neg_train, neg_test = data_loading.load_data(TRAIN_POS_PATH, TRAIN_NEG_PATH, stopwords)

    # Step 2 Train Naive Bayes Classifier on the training data
    vocabulary, prior_pos, prior_neg, vocab_pos_freq, vocab_neg_freq = classifiers.train_multinomial_NB(pos_train, neg_train, laplace_smoothing)


    # Step 3 Save the trained model in a new directory
    with open(out_directory+'vocab.txt', 'w') as f:
        for item in vocabulary:
            f.write("%s\n" % item)

    with open(out_directory+'vocab_pos_freq.txt', 'w') as f:
        for item in vocab_pos_freq:
            f.write("%s\n" % item)

    with open(out_directory+'vocab_neg_freq.txt', 'w') as f:
        for item in vocab_neg_freq:
            f.write("%s\n" % item)

    with open(out_directory+'prior_pos.txt', 'w') as f:
         f.write(str(prior_pos))

    with open(out_directory+'prior_neg.txt', 'w') as f:
         f.write(str(prior_neg))

    print("Model training finished, the model was saved in directory: ", out_directory)




