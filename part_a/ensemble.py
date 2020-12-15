from sklearn.impute import KNNImputer
from item_response import irt
import random

def sample_knn_prediction(matrix, test_data):
    """Returns knn prediction using sample of test_data"""
    nbsr = KNNImputer(n_neighbors=11)
    idx = np.random.randint(541, size=400)
    mat1 = matrix[idx,:]
    nbsr.fit(mat1)
    mat_student = nbsr.transform(matrix)
    idx = np.random.randint(1773, size=800)
    nbsr = KNNImputer(n_neighbors=21)
    mat2 = matrix.T[idx, :]
    nbsr.fit(mat2)
    mat_item = nbsr.transform(matrix.T).T
    mat_avg = (mat_item + mat_student)*0.5
    return sparse_matrix_predictions(test_data, mat_avg, threshold=0.5)

def sample_irt_prediction(data, val_data, test_data):
	n_data = len(data['user_id'])
	indices = random.choice(range(n_data), k=n_data)
	sample_data = {
		'user_id': [data['user_id'][i] for i in indices],
		'question_id': [data['question_id'][i] for i in indices],
		'is_correct': [data['is_correct'][i] for i in indices]
	}
	
	alpha = 0.01
	n_iterations = 50
	theta, beta, val_lst = irt(sample_data, val_data, alpha, n_iterations)
	pred = []
	for i, q in enumerate(test_data["question_id"]):
		u = test_data["user_id"][i]
		x = (theta[u] - beta[q]).sum()
		p_a = sigmoid(x)
		pred.append(p_a >= 0.5)
	return pred
