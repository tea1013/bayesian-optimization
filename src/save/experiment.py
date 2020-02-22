import numpy as np
import pandas as pd
import os

def save(info):
    dir_path = info['dir_path']
    file_name = info['file_name']
    n_iteration = info['n_iteration']
    csv_path = dir_path + '/' + file_name
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    index = [i + 1 for i in range(n_iteration)]
    columns = list(info['regret'].keys())
    df = pd.DataFrame(index=index, columns=columns)
    for key in columns:
        df[key] = info['regret'][key]

    df.to_csv(csv_path)
