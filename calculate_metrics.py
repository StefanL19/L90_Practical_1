import metrics 
import numpy as np
import math 

def fold_acc():
	gt = [1]*100 + [0]*100
	#system_a_pred = np.loadtxt("data/trained_models/predictions/cross_fold/0/unigram_false_bigram_true_laplace_false_stopwords_false.txt")
	#system_b_pred = np.loadtxt("data/trained_models/predictions/cross_fold/0/unigram_true_bigram_false_laplace_false_stopwords_false.txt")

	overall_acc = []
	for i in range(0, 10):
		system_a_pred = np.loadtxt("data/trained_models_new/predictions/cross_fold/"+str(i)+"/unigram_true_bigram_true_laplace_true_stopwords_false.txt")
		fold_acc = metrics.acc(system_a_pred, gt)
		print("Accuracy for fold: ", str(i), " is ", fold_acc)
		overall_acc.append(fold_acc)

	all_acc = sum(overall_acc)/10.

	print("The overall accuracy is: ", all_acc)

	print("The mean is: ", np.mean(all_acc))
	print("The variance is: ", math.sqrt(np.mean(abs(overall_acc - np.mean(overall_acc))**2)))


def significance_test_nfold():
	system_a_pred_total = []
	system_b_pred_total = []
	gt_total = []
	for i in range(0,10):
		gt = [1]*100 + [0]*100
		system_a_pred_fold = np.loadtxt("data/trained_models/predictions/cross_fold/"+str(i)+"/unigram_false_bigram_true_laplace_false_stopwords_false.txt")
		system_b_pred_fold = np.loadtxt("data/trained_models/predictions/cross_fold/"+str(i)+"/unigram_true_bigram_false_laplace_false_stopwords_false.txt")
		#print("Significance test result: ", metrics.sign_test_precision(system_a_pred_fold, system_b_pred_fold, gt))

		system_a_pred_total.extend(system_a_pred_fold)
		system_b_pred_total.extend(system_b_pred_fold)
		gt_total.extend(gt)


	print("Significance test result: ", metrics.sign_test_precision(system_a_pred_total, system_b_pred_total, gt_total))

def significance_test_holdout():
	system_a_pred = np.loadtxt("data/trained_models/predictions_old/no_fold/unigram_false_bigram_true_laplace_true.txt")
	system_b_pred = np.loadtxt("data/trained_models/predictions_old/no_fold/unigram_false_bigram_true_laplace_true.txt")

	gt = [1]*100 + [0]*100

	sign_test_result = metrics.sign_test_precision(system_a_pred, system_b_pred, gt)
	system_a_acc = metrics.acc(system_a_pred, gt)
	system_b_acc = metrics.acc(system_b_pred, gt)

	print("The accuracy of system A is: ", system_a_acc)
	print("The accuracy of system B is: ", system_b_acc)
	print("The result of the sign test is: ", sign_test_result)

def dictionary_stats():
	all_unigrams = []
	all_bigrams = []
	for i in range(0, 10):
		c_unigrams, c_bigrams = metrics.vocabulary_stats("data/trained_models_new/10_fold_no_test/unigram_true_bigram_true_laplace_true_stopwords_false/val_fold_"+str(i)+"/vocab.txt")
		all_unigrams.append(c_unigrams)
		all_bigrams.append(c_bigrams)

	print("The average count of the unigrams is: ", np.mean(all_unigrams))
	print("The average count of the bigrams is: ", np.mean(all_bigrams))

significance_test_nfold()

#significance_test_nfold()




#print("System A accuracy: ", metrics.acc(system_a_pred, gt))
#print("System B accuracy: ", metrics.acc(system_b_pred, gt))

#fold_acc()

