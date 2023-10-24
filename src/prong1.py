import statistics

from sklearn.model_selection import StratifiedKFold
from utilities import feature_extraction, train_step, test_step

SUPERVISED_ALGORITHM = "linear-svm"
K = 6  # change this hyperparameter if needed


def supervised_classification(training_data,
                              class_names,
                              testing_data=None, # if None -> we need to perform 10-fold cross validation
                              k=K,
                              algorithm=SUPERVISED_ALGORITHM):
    print('Prong 1 starting...')
    print(f"Classification algorithm: {algorithm}")
    if not testing_data:
        print("10-fold cross validation is being performed ...")

        # save the accuracies to take an average at the end
        accuracies = []

        X, y, accession_numbers = feature_extraction(training_data, k)
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = train_step(X_train, y_train, algorithm)
            accuracy = test_step(X_test, y_test, model)
            accuracies.append(accuracy)

        print(f"Accuracy (average of 10 different splits): {statistics.mean(accuracies) * 100}%")
        print("--------------------------------------------------")
        return
    else:
        X_train, y_train, train_ids = feature_extraction(training_data, k)
        X_test, y_test, test_ids = feature_extraction(testing_data, k)
        model = train_step(X_train, y_train, algorithm)
        accuracy = test_step(X_test, y_test, model)
        print(f"Accuracy: {accuracy * 100:.4f}%")

        y_pred = model.predict(X_test)

        # dictionary of predictions
        y_pred_dict = {}
        for i in range(len(test_ids)):
            y_pred_dict[test_ids[i]] = class_names[y_pred[i]]

        print("Prong 1 predictions:")
        print(y_pred_dict)
        print("--------------------------------------------------")
        return y_pred_dict
