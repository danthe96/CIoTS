import pandas as pd


def transform_ts(ts, p):
    length = ts.shape[0]
    time_shifted = []
    for t in range(p+1):
        to_append = ts.iloc[p - t: length - t].add_suffix('_' + str(t))
        to_append.reset_index(drop=True, inplace=True)
        time_shifted.append(to_append)
    data_frame = pd.concat(time_shifted, axis=1)
    data_matrix = data_frame.values
    node_mapping = {i: data_frame.columns[i]
                    for i in range(data_frame.shape[1])}
    return node_mapping, data_matrix
