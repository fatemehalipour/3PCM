from utilities import feature_extraction, linear_assignment, cluster_acc
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

UNSUPERVISED_ALGORITHM = "k-means++"
K = 6  # change this hyperparameter if needed
RUN_COUNT = 10


def unsupervised_clustering(training_data,
                            class_names,
                            clusters_count,
                            testing_data=None,  # if None -> use training data for testing
                            k=K,
                            algorithm=UNSUPERVISED_ALGORITHM):
    print("Prong 2 starting...")
    print(f"Clustering algorithm: {algorithm}")

    # if testing_data is None, use training data for testing
    if not testing_data:
        testing_data = training_data.copy()

    # extracting features by counting the k-mers
    X_train, y_train, train_ids = feature_extraction(training_data, k)
    X_test, y_test, test_ids = feature_extraction(testing_data, k)
    y_pred = None

    # save the evaluation metrics to take an average at the end
    accuracies = []
    NMIs = []
    ARIs = []
    silhouettes = []

    if algorithm == "k-means++" or algorithm == "GMM":
        for i in range(RUN_COUNT):
            if algorithm == "k-means++":
                kmeans_model = KMeans(n_clusters=clusters_count, init="k-means++", random_state=i, n_init=10)
                kmeans_model.fit(X_train)

                y_pred = kmeans_model.predict(X_test)
            else:
                GMM_model = GaussianMixture(n_components=clusters_count, random_state=i, init_params="k-means++")
                GMM_model.fit(X_train)

                y_pred = GMM_model.predict(X_test)

            ind, acc = cluster_acc(y_pred, y_test)
            accuracies.append(acc)
            NMIs.append(normalized_mutual_info_score(y_test, y_pred))
            ARIs.append(adjusted_rand_score(y_test, y_pred))
            silhouettes.append(silhouette_score(X_test, y_pred))

        print(f"Accuracy: {(sum(accuracies) / RUN_COUNT) * 100:.4f}%")
        print(f"NMI: {(sum(NMIs) / RUN_COUNT):.4f}")
        print(f"ARI: {(sum(ARIs) / RUN_COUNT):.4f}")
        print(f"Silhouette Score: {(sum(silhouettes) / RUN_COUNT):.4f}")

    if algorithm == "HClust":
        hclust_model = AgglomerativeClustering(n_clusters=clusters_count, linkage="ward").fit(X_train)
        y_pred = hclust_model.labels_

        print(f"Accuracy: {cluster_acc(y_pred, y_test) * 100:.4f}%")
        print(f"NMI: {normalized_mutual_info_score(y_test, y_pred):.4f}")
        print(f"ARI: {adjusted_rand_score(y_test, y_pred):.4f}")
        print(f"Silhouette Score: {silhouette_score(X_test, y_pred):.4f}")

    # Hungarian algorithm
    d = {}
    for i, j in ind:
        d[i] = j

    for i in range(len(y_pred)):
        y_pred[i] = d[y_pred[i]]

    # dictionary of predictions
    y_pred_dict = {}
    for i in range(len(test_ids)):
        y_pred_dict[test_ids[i]] = class_names[y_pred[i]]

    print("Prong 2 predictions:")
    print(y_pred_dict)
    print("--------------------------------------------------")
    return y_pred_dict
