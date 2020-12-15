from utils import *
from metadata import load_question_meta

import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# TODO:
# - Bagging to reduce overfitting
# - Formal description
# - Figure (matrix? show that only some abilities apply to each question)

N_students = 542
N_questions = 1774
N_subjects = 388

def sigmoid(x):
	""" Apply sigmoid function.
	"""
	return np.exp(x) / (1 + np.exp(x))


def _get_subject_matrix(metadata):
	""" Returns a 0,1-matrix where the entry at (j,k) is 1 iff question j involves subject k
	
	:param metadata: A dictionary mapping question_id to subject_id
	:return: N_questions x N_subjects matrix
	"""
	subj_mat = np.zeros((N_questions, N_subjects))
	for j in metadata:
		for k in metadata[j]:
			subj_mat[j][k] = 1.0
	return subj_mat


def neg_log_likelihood(data, metadata, theta, beta):
	""" Compute the negative log-likelihood.

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param metadata: A dictionary mapping question_id to subject_id
	:param theta: N_students x N_subjects matrix
	:param beta: Vector
	:return: float
	"""
	subj_mat = _get_subject_matrix(metadata)
	theta_2 = theta @ subj_mat.T
	x = theta_2 - beta[np.newaxis,:]
	sig_x = sigmoid(x)
	log_correct = np.log(sig_x)
	log_incorrect = np.log(1-sig_x)
	
	log_lklihood = 0.0
	for i in range(len(data['is_correct'])):
		if data['is_correct'][i]:
			log_lklihood += log_correct[data['user_id'][i],data['question_id'][i]]
		else:
			log_lklihood += log_incorrect[data['user_id'][i],data['question_id'][i]]
	
	return -log_lklihood


def update_theta_beta(data, metadata, lr, theta, beta):
	""" Update theta and beta using gradient descent.

	You are using alternating gradient descent. Your update should look:
	for i in iterations ...
		theta <- new_theta
		beta <- new_beta

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param metadata: A dictionary mapping question_id to subject_id
	:param lr: float
	:param theta: N_students x N_subjects matrix
	:param beta: Vector
	:return: tuple of vectors
	"""
	subj_mat = _get_subject_matrix(metadata)
	theta_2 = theta @ subj_mat.T
	x1 = beta[np.newaxis,:] - theta_2
	sig1 = sigmoid(x1)
	theta_grad = np.zeros(theta.shape)
	for i in range(len(data['is_correct'])):
		for k in metadata[data['question_id'][i]]:
			if data['is_correct'][i]:
				theta_grad[data['user_id'][i]][k] += sig1[data['user_id'][i]][data['question_id'][i]]
			else:
				theta_grad[data['user_id'][i]][k] -= 1-sig1[data['user_id'][i]][data['question_id'][i]]
	new_theta = theta + (lr * theta_grad)
	new_theta_2 = new_theta @ subj_mat.T
	
	x2 = beta[np.newaxis,:] - new_theta_2
	sig2 = sigmoid(x2)
	beta_grad = np.zeros(beta.shape)
	for i in range(len(data['is_correct'])):
		if data['is_correct'][i]:
			beta_grad[data['question_id'][i]] -= sig2[data['user_id'][i]][data['question_id'][i]]
		else:
			beta_grad[data['question_id'][i]] += 1-sig2[data['user_id'][i]][data['question_id'][i]]
	new_beta = beta + (lr * beta_grad)
	
	return new_theta, new_beta


def irt(data, val_data, metadata, lr, iterations):
	""" Train IRT model.

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param val_data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param metadata: A dictionary mapping question_id to subject_id
	:param lr: float
	:param iterations: int
	:return: (theta, beta, val_acc_lst)
	"""
	theta = np.zeros((N_students,N_subjects))
	beta = np.zeros((N_questions,))
	
	val_lst = []
	
	for i in range(iterations):
		neg_lld = neg_log_likelihood(data, metadata=metadata, theta=theta, beta=beta)
		pred = predict(data=val_data, metadata=metadata, theta=theta, beta=beta)
		score = evaluate(data=val_data, pred=pred)
		val_lld = -neg_log_likelihood(val_data, metadata=metadata, theta=theta, beta=beta)
		val_lst.append((score,val_lld))
		print("NLLK: {} \t Score: {}".format(neg_lld, score))
		theta, beta = update_theta_beta(data, metadata, lr, theta, beta)
	
	return theta, beta, val_lst


def predict(data, metadata, theta, beta):
	""" Make predictions on data using the given model.
	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param metadata: A dictionary mapping question_id to subject_id
	:param theta: N_students x N_subjects matrix
	:param beta: Vector
	:return: list
	"""
	pred = []
	subj_mat = _get_subject_matrix(metadata)
	theta_2 = theta @ subj_mat.T
	for i, q in enumerate(data["question_id"]):
		u = data["user_id"][i]
		x = (theta_2[u][q] - beta[q]).sum()
		p_a = sigmoid(x)
		pred.append(p_a >= 0.5)
	return pred


def evaluate(data, pred):
	""" Evaluate predictions given data and return the accuracy.
	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param pred: list
	:return: float
	"""
	return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
	train_data = load_train_csv("../data")
	val_data = load_valid_csv("../data")
	test_data = load_public_test_csv("../data")
	metadata = load_question_meta("../data")
	
	
	print("== WITHOUT BAGGING ==")
	alpha = 0.001
	n_iterations = 90
	theta, beta, _ = irt(train_data, val_data, metadata, alpha, n_iterations)
	print('Validation score:', evaluate(val_data, predict(val_data, metadata, theta, beta)))
	print('Test score:', evaluate(test_data, predict(test_data, metadata, theta, beta)))
	
	print("== WITH BAGGING ==")
	alpha = 0.01
	n_iterations = 50
	n_batch = 5
	preds_val = []
	preds_test = []
	n_train = len(train_data['user_id'])
	for i in range(n_batch):
		print("-- Batch", i+1, "==")
		indices = random.choices(range(n_train), k=n_train)
		batch_data = {
			'user_id': [train_data['user_id'][i] for i in indices],
			'question_id': [train_data['question_id'][i] for i in indices],
			'is_correct': [train_data['is_correct'][i] for i in indices]
		}
		theta_b, beta_b, _ = irt(batch_data, val_data, metadata, alpha, n_iterations)
		preds_val.append(predict(val_data, metadata, theta_b, beta_b))
		preds_test.append(predict(test_data, metadata, theta_b, beta_b))
	final_pred_val = list(scipy.stats.mode(preds_val, axis=0)[0][0])
	final_pred_test = list(scipy.stats.mode(preds_test, axis=0)[0][0])
	print('Validation score:', evaluate(val_data, final_pred_val))
	print('Test score:', evaluate(test_data, final_pred_test))


if __name__ == "__main__":
	main()
