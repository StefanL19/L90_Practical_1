import numpy as np
import math

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def acc(results, ground_truth):
	errors = 0
	for i, res in enumerate(results):
		if res != ground_truth[i]:
			errors += 1

	all_items = len(results)
	correct_preds = all_items - errors

	return correct_preds / all_items

def sign_test(preds_system_A, preds_system_B, ground_truth, q=0.5):
	ties = 0
	plus = 0
	minus = 0

	for i, gt in enumerate(ground_truth):
		if preds_system_A[i] == preds_system_B[i]:
			ties += 1
			continue

		else:
			if preds_system_A == gt:
				plus += 1
				continue

			else:
				minus += 1
				continue

	all_samples = len(ground_truth)
	k = np.ceil(ties/2) + min(plus, minus)

	N = 2 * np.ceil(ties/2) + plus + minus

	res = 0
	for idx in range(0, k):
		res += (nCr(N, idx)*np.pow(q, idx)*np.pow((1-q), (N-idx)))

	return (2*res)


