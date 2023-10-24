import pickle
from pathlib import Path
from utilities import get_stats
from prong1 import supervised_classification
from prong2 import unsupervised_clustering
from prong3 import host_identification

K = 6
CLUSTER_COUNT = 2
SUPERVISED_ALGORITHM = "linear-svm"
UNSUPERVISED_ALGORITHM = "k-means++"


def extract_class_names(training_data_file):
    if training_data_file in ["dataset1.p", "dataset2.p", "dataset2_NR.p", "dataset3.p", "dataset3_NR.p"]:
        class_names = sorted(["Avastrovirus", "Mamastrovirus"])
    elif training_data_file == "potyvirus.p":
        class_names = sorted(["Astrovirus", "Potyvirus"])
    elif training_data_file == "mamastrovirus.p":
        class_names = sorted(["HAstV", "Non-HAstV Mamastroviurs"])
    elif training_data_file == "avastrovirus.p":
        class_names = sorted(["GoAstV", "Non-GoAstV Avastrovirus"])
    return class_names


if __name__ == "__main__":
    data_path = Path("data/")

    # prong should be one of Prong 1, Prong 2, Prong 3, or All
    prong = "All"
    training_data_file = "dataset2.p"  # one of dataset1.p, dataset2.p, dataset2_NR.p, potyvirus.p, mamastrovirus.p, or avastrovirus.p
    testing_data_file = "dataset3.p"

    if training_data_file in ["potyvirus.p", "mamastrovirus.p", "avastrovirus.p"]:
        testing_data_file = training_data_file
        print(f"{training_data_file} will be used as training and testing dataset.")
    elif training_data_file == "dataset1.p":
        prong = "Prong 3"
        testing_data_file = testing_data_file
        print(f"Prong 3 is the only prong applicable on {training_data_file}")
    elif training_data_file in ["dataset2.p", "dataset2_NR.p"]:
        if testing_data_file not in ["dataset2.p", "dataset2_NR.p", "dataset3.p", "dataset3_NR.p"]:
            print("Please change the testing data.")
            exit()

    train_data = pickle.load(open(data_path / training_data_file, "rb"))
    get_stats(train_data, training_data_file)

    if testing_data_file == training_data_file:
        test_data = None
    else:
        test_data = pickle.load(open(data_path / testing_data_file, "rb"))
        get_stats(test_data, testing_data_file)

    class_names = extract_class_names(training_data_file)

    if prong == "Prong 1":
        supervised_classification(training_data=train_data,
                                  class_names=class_names,
                                  testing_data=test_data,
                                  k=K,
                                  algorithm=SUPERVISED_ALGORITHM)

    if prong == "Prong 2":
        unsupervised_clustering(training_data=train_data,
                                class_names=class_names,
                                clusters_count=CLUSTER_COUNT,
                                testing_data=test_data,
                                k=K,
                                algorithm=UNSUPERVISED_ALGORITHM)

    if prong == "Prong 3":
        host_identification(testing_data=test_data)

    if prong == "All":
        prong1_pred = supervised_classification(training_data=train_data,
                                                class_names=class_names,
                                                testing_data=test_data,
                                                k=K,
                                                algorithm=SUPERVISED_ALGORITHM)
        prong2_pred = unsupervised_clustering(training_data=train_data,
                                              class_names=class_names,
                                              clusters_count=CLUSTER_COUNT,
                                              testing_data=test_data,
                                              k=K,
                                              algorithm=UNSUPERVISED_ALGORITHM)
        prong3_pred = None
        if testing_data_file != "potyvirus.p":
            if training_data_file == "mamastrovirus.p" or training_data_file == "avastrovirus.p":
                blah, prong3_pred = host_identification(testing_data=train_data)
            else:
                prong3_pred, blah = host_identification(testing_data=test_data)
        print("Predictions of all prongs:")
        if prong1_pred:
            print(prong1_pred)
        print(prong2_pred)
        if prong3_pred:
            print(prong3_pred)
