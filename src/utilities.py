import matplotlib.pyplot as plt
import numpy as np

from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm, discriminant_analysis, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize


# extract labels from a list of tuples class-genome
def get_labels(li):
    labels = set()

    for i in range(len(li)):
        class_id = li[i][0]
        labels.add(class_id)
    return labels


# get statistics about data
def get_stats(li, dataset_name):
    labels = get_labels(li)
    data_distribution = {}

    for item in labels:
        data_distribution[item] = 0

    sequences_lengths = []
    for i in range(len(li)):
        data_distribution[li[i][0]] += 1
        sequences_lengths.append(len(li[i][1]))

    print("-----------Statistics about " + dataset_name + ": ------------")
    print("# of samples: ", len(li))
    print("# of classes: ", len(data_distribution))
    print("min seq length: " + str(min(sequences_lengths)))
    print("mean seq length: " + str(int((sum(sequences_lengths) / len(sequences_lengths)))))
    print("max seq length: " + str(max(sequences_lengths)))
    print("data distribution: ")
    for key, item in data_distribution.items():
        print(f"{key:8} => {item:8}")
    print("-------------------------------------------------")

    # label numerical assignment
    label_assignment = {}
    class_id = 0
    for item in data_distribution:
        class_id += 1
        label_assignment[item] = class_id

    return label_assignment, data_distribution


# plot histogram of class occurrences
def plot_data(d, path, title):
    plt.bar(range(len(d)), d.values(), align="center", color="lightsteelblue", zorder=3, linestyle="-.")
    plt.margins(0.05)
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)
    plt.xticks(range(len(d)), list(d.keys()))
    plt.rcParams["axes.axisbelow"] = True
    plt.xticks(rotation=90)
    plt.grid(True, zorder=0)
    plt.title(title)
    plt.savefig(path)
    plt.show()


def kmer_count(seq, k):
    """
    Compute the kmer counts for a given sequence
    :param seq:
    :param k:
    :return: Counts.
    """
    kmer_dict = {}

    for k_mer in product("ACGU", repeat=k):
        kmer = "".join(k_mer)
        kmer_dict[kmer] = 0

    idx = 0
    while idx < len(seq) - k:
        try:
            kmer_dict[seq[idx:idx + k]] += 1
        except KeyError:
            pass
        idx += 1
    return list(kmer_dict.values())


def feature_extraction(data, k):
    unique_labels = sorted(set(map(lambda x: x[0], data)))

    data_features = []
    data_labels = []
    accession_numbers = []
    for i in range(len(data)):
        t = kmer_count(data[i][1], k)
        t = np.array(t)
        t = normalize(t[:, np.newaxis], axis=0).ravel()

        data_features.append(t)

        label = unique_labels.index(data[i][0])
        data_labels.append(label)

        accession_numbers.append(data[i][2])

    x = np.asarray(data_features).astype("float32")
    y = np.asarray(data_labels)
    return x, y, accession_numbers


def build_model(classifier):
    # setup normalizers if needed
    normalizers = []

    # Classifiers
    # 10-nearest-neighbors
    if classifier == "10-nearest-neighbors":
        normalizers.append(("classifier", neighbors.KNeighborsClassifier(n_neighbors=10, metric="euclidean")))

    # nearest-centroid-mean
    if classifier == "nearest-centroid-mean":
        normalizers.append(("classifier", neighbors.NearestCentroid(metric="euclidean")))

    # nearest-centroid-median
    if classifier == "nearest-centroid-median":
        normalizers.append(("classifier", neighbors.NearestCentroid(metric="manhattan")))

    # logistic-regression
    if classifier == "logistic-regression":
        normalizers.append(("classifier", LogisticRegression()))

    # linear-svm
    if classifier == "linear-svm":
        normalizers.append(("classifier", svm.SVC(kernel="linear", C=1)))

    # quadratic-svm
    if classifier == "quadratic-svm":
        normalizers.append(("classifier", svm.SVC(kernel="poly", degree=2)))

    # cubic-svm
    if classifier == "cubic-svm":
        normalizers.append(("classifier", svm.SVC(kernel="poly", degree=3)))

    # sgd
    if classifier == "sgd":
        normalizers.append(("classifier", SGDClassifier(max_iter=5)))

    # decision-tree
    if classifier == "decision-tree":
        normalizers.append(("classifier", DecisionTreeClassifier()))

    # random-forest
    if classifier == "random-forest":
        normalizers.append(("classifier", RandomForestClassifier(n_estimators=10)))

    # adaboost
    if classifier == "adaboost":
        normalizers.append(("classifier", AdaBoostClassifier(n_estimators=50)))

    # gaussian-naive-bayes
    if classifier == "gaussian-naive-bayes":
        normalizers.append(("classifier", GaussianNB()))

    # lda
    if classifier == "lda":
        normalizers.append(("classifier", discriminant_analysis.LinearDiscriminantAnalysis()))

    # qda
    if classifier == "qda":
        normalizers.append(("classifier", discriminant_analysis.QuadraticDiscriminantAnalysis()))

    # multilayer-perceptron
    if classifier == "multilayer-perceptron":
        normalizers.append(("classifier", MLPClassifier(solver="sgd")))

    return Pipeline(normalizers)


def train_step(X, y, classifier):
    # Run the classification Pipeline
    pipeline = build_model(classifier)
    pipeline.fit(X, y)
    return pipeline


def test_step(X_test, y_test, model):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # print("Confusion Matrix:")
    # print(w)
    ind = linear_assignment(w.max() - w)
    return ind, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


