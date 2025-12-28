import pandas as pd
import os

def load_emg_data(folder_path):
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    dataframes = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dataframes, ignore_index=True)

    return data
