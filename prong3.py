import pandas as pd


def host_identification(test):
    # Read xl file with labels
    xl_file = pd.ExcelFile('Data/Cleaned Dataset.xlsx')

    # extract different sheets in xl file
    dfs = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}
    df = dfs['1039 documents']
    df = df.sample(frac=1)

    # extract list of column in the main sheet
    columns = df.columns

    # extract accession numbers and hosts information
    ac_numbers = df.loc[:, columns[0]].tolist()
    hosts_class = df.loc[:, columns[4]].tolist()
    # hosts_family = df.loc[:, columns[3]].tolist()
    host_species = df.loc[:, columns[2]].tolist()
    host_class_dict = {}
    host_species_dict = {}
    for i in range(len(ac_numbers)):
        host_class_dict[ac_numbers[i]] = hosts_class[i]
        host_species_dict[ac_numbers[i]] = host_species[i]

    # Extract accession IDs of the test set
    accession_numbers = []
    for i in range(len(test)):
        accession_numbers.append(test[i][-1])

    dict_y_pred_class = {}
    dict_y_pred_species = {}
    for i in range(len(accession_numbers)):
        dict_y_pred_class[accession_numbers[i]] = host_class_dict[ac_numbers[i]]
        dict_y_pred_species[accession_numbers[i]] = host_species_dict[ac_numbers[i]]
    print('Host (class level):')
    print(dict_y_pred_class)
    print('Host (species level):')
    print(dict_y_pred_species)
    print('--------------------------------------------------')
    return dict_y_pred_class, dict_y_pred_species
