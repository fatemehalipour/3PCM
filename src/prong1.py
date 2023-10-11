import statistics

from sklearn.model_selection import StratifiedKFold
from utilities import feature_extraction, training, testing


def supervised_classification(train, test, k, cross_validation_flag, level):
    print('Prong 1 starting...')

    # Select one of '10-nearest-neighbors', 'nearest-centroid-mean', 'nearest-centroid-median', 'logistic-regression',
    # 'linear-svm', 'quadratic-svm', 'cubic-svm', 'sgd', 'decision-tree', 'random-forest', 'adaboost',
    # 'gaussian-naive-bayes', 'lda', 'qda', 'multilayer-perceptron'
    algorithm = 'quadratic-svm'
    print('Classification algorithm being used is ' + algorithm + '.')
    if cross_validation_flag:
        print('10-fold cross validation is being performed ...')
        x_train, y_train, accession_numbers = feature_extraction(train, k)
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        accuracies = []
        for train_index, test_index in kf.split(x_train, y_train):
            X_tr, X_te = x_train[train_index], x_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            pipeline = training(X_tr, y_tr, algorithm)
            accuracy = testing(X_te, y_te, pipeline)
            accuracies.append(accuracy)
        print('Accuracy (average of 10 different splits) = ' + str(statistics.mean(accuracies)))
        x_test = x_train
    else:
        x_train, y_train, train_accession_numbers = feature_extraction(train, k)
        x_test, y_test, accession_numbers = feature_extraction(test, k)
        pipeline = training(x_train, y_train, algorithm)
        accuracy = testing(x_test, y_test, pipeline)
        print('Accuracy = ' + str(accuracy))

    y_pred = pipeline.predict(x_test)

    dict_y_pred = {}
    if level == "genus":
        for i in range(len(accession_numbers)):
            if y_pred[i] == 0:
                dict_y_pred[accession_numbers[i]] = 'Avastrovirus'
            if y_pred[i] == 1:
                dict_y_pred[accession_numbers[i]] = 'Mamastrovirus'
            # if y_pred[i] == 2:
            #     dict_y_pred[accession_numbers[i]] = 'Unknown'
    if level == "family":
        for i in range(len(accession_numbers)):
            if y_pred[i] == 0:
                dict_y_pred[accession_numbers[i]] = 'Astrovirus'
            if y_pred[i] == 1:
                dict_y_pred[accession_numbers[i]] = 'Potyvirus'


    print('Prong 1 predictions:')
    print(dict_y_pred)
    print('--------------------------------------------------')
    return dict_y_pred
