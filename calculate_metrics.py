import metrics 
import numpy as np

gt = [1]*100 + [0]*100
system_a_pred = np.loadtxt("data/trained_models/predictions/no_fold/unigram_true_bigram_false_laplace_true.txt")
system_b_pred = np.loadtxt("data/trained_models/predictions/no_fold/unigram_true_bigram_true_laplace_false.txt")

print("System A accuracy: ", metrics.acc(system_a_pred, gt))
print("System B accuracy: ", metrics.acc(system_b_pred, gt))

print("Significance test result: ", metrics.sign_test(system_a_pred, system_b_pred, gt))