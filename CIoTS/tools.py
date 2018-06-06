import numpy as np
import pandas as pd


def transform_ts(ts, p):
    l = ts.shape[0]
    time_shifted = []
    for t in range(p+1):
        time_shifted.append(ts.iloc[p-t, l-t].add_suffix('_'+str(i)))
    data_frame = pd.concat(time_shifted, axis=1, ignore_index=True)

    data_matrix = data_frame.as_matrix()
    node_mapping = {i: data_frame.columns[i]
                    for i in range(data_frame.shape[1])}
    return node_mapping, data_matrix
