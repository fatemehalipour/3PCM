import pandas as pd
from pathlib import Path

def host_identification(testing_data):
    data_path = Path("data/")
    xl_file = pd.ExcelFile(data_path / "metadata_host.xlsx")
    # extract different sheets in xl file
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    metadata_host_df = dfs["1039 documents"]
    df = metadata_host_df.set_index(["Name"])

    dict_y_pred_class = {}
    dict_y_pred_species = {}
    for label, seq, name in testing_data:
        dict_y_pred_class[name] = df.loc[name, "Class"]
        dict_y_pred_species[name] = df.loc[name, "Host"]

    print("Host (class level):")
    print(dict_y_pred_class)
    print("Host (species level):")
    print(dict_y_pred_species)
    print("--------------------------------------------------")
    return dict_y_pred_class, dict_y_pred_species