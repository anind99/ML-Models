from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
	""" Apply sigmoid function.
	"""
	return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
	""" Compute the negative log-likelihood.

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param theta: Vector
	:param beta: Vector
	:return: float
	"""
	#####################################################################
	# TODO:                                                             #
	# Implement the function as described in the docstring.             #
	#####################################################################
	x = theta[:,np.newaxis] - beta[np.newaxis,:]
	sig_x = sigmoid(x)
	log_correct = np.log(sig_x)
	log_incorrect = np.log(1-sig_x)
	
	log_lklihood = 0.0
	for i in range(len(data['is_correct'])):
		if data['is_correct'][i]:
			log_lklihood += log_correct[data['user_id'][i],data['question_id'][i]]
		else:
			log_lklihood += log_incorrect[data['user_id'][i],data['question_id'][i]]
	#####################################################################
	#                       END OF YOUR CODE                            #
	#####################################################################
	return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
	""" Update theta and beta using gradient descent.

	You are using alternating gradient descent. Your update should look:
	for i in iterations ...
		theta <- new_theta
		beta <- new_beta

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param lr: float
	:param theta: Vector
	:param beta: Vector
	:return: tuple of vectors
	"""
	#####################################################################
	# TODO:                                                             #
	# Implement the function as described in the docstring.             #
	#####################################################################
	x1 = beta[np.newaxis,:] - theta[:,np.newaxis]
	sig1 = sigmoid(x1)
	theta_grad = np.zeros(theta.shape)
	for i in range(len(data['is_correct'])):
		if data['is_correct'][i]:
			theta_grad[data['user_id'][i]] += sig1[data['user_id'][i]][data['question_id'][i]]
		else:
			theta_grad[data['user_id'][i]] -= 1-sig1[data['user_id'][i]][data['question_id'][i]]
	new_theta = theta + (lr * theta_grad)
	
	x2 = beta[np.newaxis,:] - new_theta[:,np.newaxis]
	sig2 = sigmoid(x2)
	beta_grad = np.zeros(beta.shape)
	for i in range(len(data['is_correct'])):
		if data['is_correct'][i]:
			beta_grad[data['question_id'][i]] -= sig2[data['user_id'][i]][data['question_id'][i]]
		else:
			beta_grad[data['question_id'][i]] += 1-sig2[data['user_id'][i]][data['question_id'][i]]
	new_beta = beta + (lr * beta_grad)
	#####################################################################
	#                       END OF YOUR CODE                            #
	#####################################################################
	return new_theta, new_beta


def irt(data, val_data, lr, iterations):
	""" Train IRT model.

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param val_data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param lr: float
	:param iterations: int
	:return: (theta, beta, val_acc_lst)
	"""
	# TODO: Initialize theta and beta.
	N_students = 542
	N_questions = 1774
	theta = np.zeros((N_students,))
	beta = np.zeros((N_questions,))

	val_lst = []

	for i in range(iterations):
		neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
		score = evaluate(data=val_data, theta=theta, beta=beta)
		val_lld = -neg_log_likelihood(val_data, theta=theta, beta=beta)
		val_lst.append((score,val_lld))
		print("NLLK: {} \t Score: {}".format(neg_lld, score))
		theta, beta = update_theta_beta(data, lr, theta, beta)

	# TODO: You may change the return values to achieve what you want.
	return theta, beta, val_lst


def evaluate(data, theta, beta):
	""" Evaluate the model given data and return the accuracy.
	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}

	:param theta: Vector
	:param beta: Vector
	:return: float
	"""
	pred = []
	for i, q in enumerate(data["question_id"]):
		u = data["user_id"][i]
		x = (theta[u] - beta[q]).sum()
		p_a = sigmoid(x)
		pred.append(p_a >= 0.5)
	return np.sum((data["is_correct"] == np.array(pred))) \
		   / len(data["is_correct"])


def main():
	train_data = load_train_csv("../data")
	# You may optionally use the sparse matrix.
	sparse_matrix = load_train_sparse("../data")
	val_data = load_valid_csv("../data")
	test_data = load_public_test_csv("../data")

	#####################################################################
	# TODO:                                                             #
	# Tune learning rate and number of iterations. With the implemented #
	# code, report the validation and test accuracy.                    #
	#####################################################################
	alpha = 0.01
	n_iterations = 50
	#####################################################################
	#                       END OF YOUR CODE                            #
	#####################################################################

	#####################################################################
	# TODO:                                                             #
	# Implement part (c)                                                #
	#####################################################################
	theta, beta, val_lst = irt(train_data, val_data, alpha, n_iterations)
	print('Validation score:', evaluate(data=val_data, theta=theta, beta=beta))
	print('Test score:', evaluate(data=test_data, theta=theta, beta=beta))
	
	# Accuracy plots
	iters = range(1,n_iterations+1)
	plt.plot(iters, [v[0] for v in val_lst],'g-')
	plt.title('Validation accuracy')
	plt.show()
	plt.plot(iters, [v[1] for v in val_lst],'b-')
	plt.title('Validation log-likelihood')
	plt.show()
	
	# Probability plots
	js = [866,890,1165,1202,1410]
	xs = np.linspace(-5,5)
	for j in js:
		plt.plot(xs,sigmoid(xs-beta[j]))
		plt.vlines(beta[j],0,1)
		plt.title("Probability of answering question "+str(j)+" correctly")
		plt.show()
	#####################################################################
	#                       END OF YOUR CODE                            #
	#####################################################################


if __name__ == "__main__":
	main()
