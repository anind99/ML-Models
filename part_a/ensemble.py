from sklearn.impute import KNNImputer
from item_response import irt
from neural_network import *
import random
from utils import *


def sample_knn_prediction(matrix, test_data):
    """Returns knn prediction using sample of test_data"""
    matrix_c = np.copy(matrix.T)
    nbsr = KNNImputer(n_neighbors=11)
    idx = np.random.randint(542, size=542)
    mat1 = matrix[idx,:]
    nbsr.fit(mat1)
    mat_student = nbsr.transform(matrix)
    idx = np.random.randint(1774, size=1774)
    nbsr = KNNImputer(n_neighbors=21)
    mat2 = matrix_c[idx, :]
    nbsr.fit(mat2)
    mat_item = nbsr.transform(matrix_c).T
    mat_avg = (mat_item + mat_student)*0.5
    return sparse_matrix_predictions(test_data, mat_avg, threshold=0.5)

def sample_irt_prediction(data, val_data, test_data):
	n_data = len(data['user_id'])
	indices = random.choices(range(n_data), k=n_data)
	sample_data = {
		'user_id': [data['user_id'][i] for i in indices],
		'question_id': [data['question_id'][i] for i in indices],
		'is_correct': [data['is_correct'][i] for i in indices]
	}
	
	alpha = 0.01
	n_iterations = 50
	theta, beta, val_lst = irt(sample_data, val_data, alpha, n_iterations)
	
	val_pred = []
	for i, q in enumerate(val_data["question_id"]):
		u = val_data["user_id"][i]
		x = (theta[u] - beta[q]).sum()
		p_a = sigmoid(x)
		val_pred.append(p_a >= 0.5)
	
	test_pred = []
	for i, q in enumerate(test_data["question_id"]):
		u = test_data["user_id"][i]
		x = (theta[u] - beta[q]).sum()
		p_a = sigmoid(x)
		test_pred.append(p_a >= 0.5)
	
	return val_pred, test_pred


def sample_nn_predictions(train_matrix, test_data):
	"""
    Used in Part A Question 4 to use the neural net as
    part of the bagging ensemble.

    Returns a list of predictions.
    """
	zero_train_matrix = train_matrix.copy()
	# Fill in the missing entries to 0.
	zero_train_matrix[np.isnan(train_matrix)] = 0
	# Change to Float Tensor for PyTorch.
	zero_train_matrix = torch.FloatTensor(zero_train_matrix)
	train_matrix = torch.FloatTensor(train_matrix)

	k = 10
	model = AutoEncoder(train_matrix.size()[1], k)
	lr = 0.005
	num_epoch = 200
	lamb = 0.001

	result = train(model, lr, lamb, train_matrix,
				   zero_train_matrix, test_data, num_epoch)

	test_acc = evaluate(model, zero_train_matrix, test_data)
	print("Test Acc: {}".format(test_acc))

	predictions = []

	for i, u in enumerate(test_data["user_id"]):
		inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
		output = model(inputs)

		guess = output[0][test_data["question_id"][i]].item() >= 0.5
		predictions.append(guess)

	return predictions

def main():
	zero_train_matrix, train_matrix, valid_data, test_data = load_data()
	pred1_val, pred1_test =  sample_nn_predictions(train_matrix, valid_data), sample_nn_predictions(train_matrix, test_data)
	pred2_val, pred2_test = sample_knn_prediction(train_matrix, valid_data), sample_knn_prediction(train_matrix, test_data)
	pred3_val, pred3_test = sample_irt_prediction(data, val_data, test_data)

	final_test = (np.array(pred1_test) + np.array(pred2_test) + np.array(pred3_test))/3 > 0.5
	print("Bagging Test Accuracy: ".format(evaluate(test_data, final_test)))

	final_val = (np.array(pred1_val) + np.array(pred2_val) + np.array(pred3_val))/3 > 0.5
	print("Bagging Validation Accuracy: ".format(evaluate(valid_data, final_val)))

if __name__ == "__main__":
    main()


