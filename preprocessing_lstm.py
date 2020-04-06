import pandas as pd
import numpy as np
import pickle

stv = pd.read_csv('sales_train_validation.csv')

def df_to_ts(df, ts_cols, lead_zeros=0):
    data = []
    for index, row in df.iterrows():
        ts = row[ts_cols].to_numpy(dtype=np.int32)
        len_nonzero = len(np.trim_zeros(ts, 'f'))
        len_nonzero = max(100, len_nonzero + lead_zeros)
        len_nonzero = min(len_nonzero, len(ts))
        data.append(ts[-len_nonzero:])
    return data

ts_cols = [f'd_{i}' for i in range(1, 1914)]
plain_ts = df_to_ts(stv, ts_cols)

with open('plain_ts2.pickle', 'wb') as handle:
    pickle.dump(plain_ts, handle)