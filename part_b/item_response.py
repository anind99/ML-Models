from utils import *
from metadata import load_question_meta

import numpy as np
import matplotlib.pyplot as plt


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
		score = evaluate(data=val_data, metadata=metadata, theta=theta, beta=beta)
		val_lld = -neg_log_likelihood(val_data, metadata=metadata, theta=theta, beta=beta)
		val_lst.append((score,val_lld))
		print("NLLK: {} \t Score: {}".format(neg_lld, score))
		theta, beta = update_theta_beta(data, metadata, lr, theta, beta)
	
	return theta, beta, val_lst


def evaluate(data, metadata, theta, beta):
	""" Evaluate the model given data and return the accuracy.
	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param metadata: A dictionary mapping question_id to subject_id
	:param theta: N_students x N_subjects matrix
	:param beta: Vector
	:return: float
	"""
	pred = []
	subj_mat = _get_subject_matrix(metadata)
	theta_2 = theta @ subj_mat.T
	for i, q in enumerate(data["question_id"]):
		u = data["user_id"][i]
		x = (theta_2[u][q] - beta[q]).sum()
		p_a = sigmoid(x)
		pred.append(p_a >= 0.5)
	return np.sum((data["is_correct"] == np.array(pred))) \
		   / len(data["is_correct"])


def main():
	train_data = load_train_csv("../data")
	val_data = load_valid_csv("../data")
	test_data = load_public_test_csv("../data")
	metadata = load_question_meta("../data")
	
	alpha = 0.001
	n_iterations = 200
	
	theta, beta, val_lst = irt(train_data, val_data, metadata, alpha, n_iterations)
	print('Validation score:', evaluate(data=val_data, metadata=metadata, theta=theta, beta=beta))
	print('Test score:', evaluate(data=test_data, metadata=metadata, theta=theta, beta=beta))
	
	# Accuracy plots
	iters = range(1,n_iterations+1)
	plt.plot(iters, [v[0] for v in val_lst],'g-')
	plt.title('Validation accuracy')
	plt.show()
	plt.plot(iters, [v[1] for v in val_lst],'b-')
	plt.title('Validation log-likelihood')
	plt.show()


if __name__ == "__main__":
	main()
