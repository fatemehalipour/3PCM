import pickle

from utilities import get_stats
from prong1 import supervised_classification
from prong2 import clustering
from prong3 import host_identification


def read_file(file_name):
    data = pickle.load(open('data/' + file_name, "rb"))
    labels_assignment, data_distribution = get_stats(data, file_name)
    return data


if __name__ == '__main__':
    # change the next line if you want to change the value of k in k-mers
    k = 6
    cross_validation_flag = False

    # The input should be one of Prong 1, Prong 2, Prong 3, or All
    prong = input('Please select the prong you want to run (e.g. Prong 1, Prong 2, Prong 3, or All) ')

    train_data = train = test = None
    if prong != 'Prong 3':
        # The input should be one of dataset1.p, dataset2.p, dataset2_NR.p, dataset3.p,'
        #             ' dataset3_NR.p, Potyvirus.p, 5_hosts.p, Mamastrovirus.p, or Avastrovirus.p
        train_data = input(
            'Please select the training dataset (e.g. dataset1.p, dataset2.p, dataset2_NR.p, dataset3.p,'
            ' dataset3_NR.p, Potyvirus.p, 5_hosts.p, Mamastrovirus.p, or Avastrovirus.p) ')
        train = read_file(train_data)

    # The input should be one of dataset1.p, dataset2.p, dataset2_NR.p, dataset3.p,'
    #         #             ' dataset3_NR.p, Potyvirus.p, 5_hosts.p, Mamastrovirus.p, or Avastrovirus.p
    if train_data == 'Potyvirus.p':
        test_data = 'Potyvirus.p'
        print("Potyvirus.p will be used as training and testing dataset")
    else:
        test_data = input('Please select the testing dataset (e.g. dataset1.p, dataset2.p, dataset2_NR.p, dataset3.p,'
                          ' dataset3_NR.p, 5_hosts.p, Mamastrovirus.p, or Avastrovirus.p) ')
    if test_data == train_data:
        print('You selected the same dataset for training and testing.')
        test = train
        cross_validation_flag = True
    else:
        test = read_file(test_data)

    if train_data == "dataset1.p" or train_data == "dataset2.p" or train_data == "dataset2_NR.p" or train_data == "dataset3.p" or train_data == "dataset3_NR.p":
        level = "genus"
    if train_data == 'Potyvirus.p':
        level = "family"

    if prong == 'Prong 1':
        supervised_classification(train, test, k, cross_validation_flag, level=level)

    if prong == 'Prong 2':
        clustering(train, test, k, level=level)

    if prong == 'Prong 3':
        if test_data == 'Potyvirus.p':
            print(f"Prong 3 is not applicable on {test_data}")
            exit()
        host_identification(test)

    if prong == 'All':
        prong1_pred = supervised_classification(train, test, k, cross_validation_flag, level=level)
        prong2_pred = clustering(train, test, k, level=level)
        if test_data == 'Potyvirus.p':
            print(f"Prong 3 is not applicable on {test_data}")
        else:
            if test_data == 'Mamastrovirus.p' or test_data == 'Avastrovirus.p':
                blah, prong3_pred = host_identification(test)
            else:
                prong3_pred, blah = host_identification(test)
        print('Prediction of all three prongs:')
        print(prong1_pred)
        print(prong2_pred)
        if test_data == 'Potyvirus.p':
            # print("Prong 3 is not applicable on the testing data.")
            exit()
        print(prong3_pred)
