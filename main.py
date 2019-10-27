import classifiers
from os import listdir, mkdir
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

def train_model(train_pos_path, train_neg_path, stopwords, laplace_smoothing, out_directory, split):
    # Step 1 Load the data
    pos_train, pos_test, neg_train, neg_test = data_loading.load_data_kfold_10(train_pos_path, train_neg_path, stopwords, split)

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

    # Return the test sets, so the model can be tested on the appropriate data
    return pos_test, neg_test


def train_validation_naive_bayes():
    stopwords = ["\n"]
    test_splits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    laplace_smoothing = True
    for idx, split in tqdm(enumerate(test_splits)):

        if laplace_smoothing:
            data_dir = "data/experiments/k_fold/laplace_smoothing/"+str(idx)
        else:
            data_dir = "data/experiments/k_fold/no_laplace_smoothing/"+str(idx)
        
        mkdir(data_dir)
        
        pos_test, neg_test = train_model("data/data-tagged/POS/", "data/data-tagged/NEG/", stopwords, laplace_smoothing, data_dir+"/", split)
        
        vocabulary_path = data_dir+"/"+"vocab.txt"
        vocab_pos_freq_path = data_dir+"/"+'vocab_pos_freq.txt'
        vocab_neg_freq_path = data_dir+"/"+'vocab_neg_freq.txt'
        prior_pos_path = data_dir+"/"+'prior_pos.txt'
        prior_neg_path = data_dir+"/"+'prior_neg.txt'

        #generate_predictions(vocabulary_path, vocab_pos_freq_path, vocab_neg_freq_path, prior_pos_path, prior_neg_path, pos_test, neg_test)

#data_loading.load_data_kfold_10("data/data-tagged/POS/", "data/data-tagged/NEG/", ["\n"], test_category=0)
train_validation_naive_bayes()

