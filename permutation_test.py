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

# gt = [1]*100 + [0]*100

# print("The sign test result is: ", metrics.sign_test_precision(np.argmax(system_a_preds_2d, axis=1), np.argmax(system_b_preds_2d, axis=1), gt))

# print(system_a_preds)
# errors = 0
# for i, g in enumerate(gt):
# 	if np.argmax(system_b_preds_2d, axis=1)[i] != g:
# 		errors+=1


# print((200-errors) / 200)

# true_mean_diff = abs((sum(system_a_preds)/len(system_a_preds)) - (sum(system_b_preds)/len(system_b_preds)))

# print("The true mean is: ", true_mean_diff)
# permutation_means = []

# for i in tqdm(range(0, R)):
# 	new_a = []
# 	new_b = []
# 	flips = []
# 	for idx, pred in enumerate(system_a_preds):
# 		flip = random.uniform(0, 1)
# 		flips.append(flip)
# 		if flip > 0.5:
# 			new_a.append(system_b_preds[idx])
# 			new_b.append(pred)
# 		else:
# 			new_a.append(pred)
# 			new_b.append(system_b_preds[idx])

# 	new_mean = abs((sum(new_a)/len(new_a)) - (sum(new_b)/len(new_b)))
# 	permutation_means.append(new_mean)

# mean_bigger = 0
# for m in permutation_means:
# 	if m > true_mean_diff:
# 		mean_bigger += 1


# # print(mean_bigger)
# print("The p value is: ", (mean_bigger+1)/(R+1))

