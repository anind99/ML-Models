from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("user {}".format(k))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    print("item {}".format(k))
    return acc


def main():
    sparse_matrix = load_train_sparse("").toarray()
    val_data = load_valid_csv("")
    test_data = load_public_test_csv("")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    x = [1, 6, 11, 16, 21, 26]
    validacc_user = [knn_impute_by_user(sparse_matrix, val_data, i) for i in x]
    validacc_item = [knn_impute_by_item(sparse_matrix, val_data, i) for i in x]

    plt.plot(x, validacc_user, label="By user similarity")
    plt.plot(x, validacc_item, label="By item similarity")
    plt.ylabel("accuracy")
    plt.xlabel("k - nearest neighbour")
    plt.legend()

    k1 = x[np.argmax(validacc_user)]
    k2 = x[np.argmax(validacc_item)]

    print("Test accuracy - by user: {} with k = {}".format(knn_impute_by_user(sparse_matrix, test_data, k1), k1))
    print("Test accuracy - by item: {} with k = {}".format(knn_impute_by_item(sparse_matrix, test_data, k2), k2))


if __name__ == "__main__":
    main()
