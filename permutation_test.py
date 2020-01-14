import random
from tqdm import tqdm
import numpy as np
import metrics 

def perm_test(s1, s2, R):
    n, bigger = len(s1), 0
    diff = np.abs(np.mean(s1) - np.mean(s2))
    concat = np.concatenate([s1, s2])
    for j in range(R):
        np.random.shuffle(concat)
        bigger += diff <= np.abs(np.mean(concat[:n]) - np.mean(concat[n:]))
    return bigger / R

R = 5000
system_a = "data/predictions/svm_bow.txt"
system_b = "data/predictions/svm_poly.txt"

system_a_preds_2d = np.loadtxt(system_a)
system_b_preds_2d = np.loadtxt(system_b)

system_a_preds = system_a_preds_2d[:, 1]
system_b_preds = system_b_preds_2d[:, 1]
print(perm_test(system_a_preds, system_b_preds, R))
