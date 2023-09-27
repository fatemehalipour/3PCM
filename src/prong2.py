from utilities import feature_extraction, linear_assignment, cluster_acc
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering(train, test, k):
    print('Prong 2 starting...')

    # Change number of clusters if necessary
    num_classes = 2
    run_count = 20

    # Select one of 'k-means++', 'GMM', 'HClust
    algorithm = 'k-means++'
    print('Clustering algorithm being used is ' + algorithm + '.')

    x_train, y_train, train_accession_numbers = feature_extraction(train, k)
    x_test, y_test, accession_numbers = feature_extraction(test, k)
    y_pred = None

    accuracies = []
    NMIs = []
    ARIs = []
    silhouettes = []

    if algorithm == 'k-means++' or algorithm == 'GMM':
        for i in range(run_count):
            if algorithm == 'k-means++':
                kmeans_model = KMeans(n_clusters=num_classes, init="k-means++", random_state=i, n_init=10)
                kmeans_model.fit(x_train)

                y_pred = kmeans_model.predict(x_test)
            else:
                GMM_model = GaussianMixture(n_components=num_classes, random_state=5, init_params='k-means++')
                GMM_model.fit(x_train)

                y_pred = GMM_model.predict(x_test)

            ind, acc = cluster_acc(y_pred, y_test)
            accuracies.append(acc)
            NMIs.append(normalized_mutual_info_score(y_test, y_pred))
            ARIs.append(adjusted_rand_score(y_test, y_pred))
            silhouettes.append(silhouette_score(x_test, y_pred))

        print('Accuracy = ' + str(sum(accuracies) / run_count))
        print('NMI = ' + str(sum(NMIs) / run_count))
        print('ARI = ' + str(sum(ARIs) / run_count))
        print('Silhouette Score = ' + str(sum(silhouettes) / run_count))

    if algorithm == 'HClust':
        hclust_model = AgglomerativeClustering(n_clusters=num_classes, linkage='ward').fit(x_train)
        y_pred = hclust_model.labels_

        print('Accuracy = ' + str(cluster_acc(y_pred, y_test)))
        print('NMI = ' + str(normalized_mutual_info_score(y_test, y_pred)))
        print('ARI = ' + str(adjusted_rand_score(y_test, y_pred)))
        print('Silhouette Score = ' + str(silhouette_score(x_test, y_pred)))

    d = {}
    for i, j in ind:
        d[i] = j

    for i in range(len(y_pred)):  # we do this for each sample or sample batch
        y_pred[i] = d[y_pred[i]]
    dict_y_pred = {}
    for i in range(len(accession_numbers)):
        if y_pred[i] == 0:
            dict_y_pred[accession_numbers[i]] = 'Avastrovirus'
        if y_pred[i] == 1:
            dict_y_pred[accession_numbers[i]] = 'Mamastrovirus'
    print('Prong 2 predictions:')
    print(dict_y_pred)
    print('--------------------------------------------------')
    return dict_y_pred
