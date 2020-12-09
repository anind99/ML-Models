# TODO: complete this file.
def sample_knn_prediction(matrix, test_data):
    """Returns knn prediction using sample of test_data"""
    nbsr = KNNImputer(n_neighbors=7)
    idx = np.random.randint(541, size=400)
    mat1 = matrix[idx,:]
    nbsr.fit(mat1)
    mat_student = nbsr.transform(matrix)
    idx = np.random.randint(1773, size=800)
    nbsr = KNNImputer(n_neighbors=11)
    mat2 = matrix.T[idx, :]
    nbsr.fit(mat2)
    mat_item = nbsr.transform(matrix.T).T
    mat_avg = (mat_item + mat_student)*0.5
    return sparse_matrix_predictions(test_data, mat_avg, threshold=0.5)