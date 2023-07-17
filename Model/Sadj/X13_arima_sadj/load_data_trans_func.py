
import os

import numpy as np
import pandas as pd
import openpyxl

from statsmodels.tsa.stattools import adfuller


# data for function test
path = 'D:\2023\Data_WareHouse\ECOS(한국은행경제통계)\Data_02\bak'
file = 'dataset_03_S.xlsx'
data_path = os.path.join(path, file)

# function


def adf_test(transformed):
    if (result := adfuller(transformed.values))[1] < 0.05:
        test_result = "{}".format("S")
    else:
        test_result = "{}".format("N")
    return test_result


def transform(df, var_info, start, end):
    for col in df.columns:
        diff = var_info[var_info['ID' == col]]['transform']
        df_trans = df.copy()
        # transform N test
        if diff == 'Origin':
            transformed = df[col].loc[start:end].dropna()
            res = adf_test(transformed)
            df_trans[col] = transformed
        elif diff == 'Diff-1':
            transformed = diff.loc[start:end].diff().dropna()
            res = adf_test(transformed)
            df_trans[col] = transformed
        elif diff == 'Log-1':
            # error transform log list
            if col in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
                res = 'X'
            else:
                log_1 = diff.loc[start:end]
                transformed = np.log(log_1).dropna()
                res = adf_test(transformed)
                df_trans[col] = transformed
        elif diff == 'Diff-2':
            transformed = diff.loc[start:end].diff().diff().dropna()
            res = adf_test(transformed)
            df_trans[col] = transformed
        else:
            print(f"transformation not orderred")

        if res == 'N' or res == 'X':
            return print(f"transformed data column variable stationary Adfuller Test is fail: {res}")
    return df_trans


def load_data_DFM(data_path, start, end):
    # load data
    df_Q = pd.read_excel(data_path, sheet_name='df_Q', header=0)
    df_M = pd.read_excel(data_path, sheet_name='df_M', header=0)
    var_info = pd.read_excel(data_path, sheet_name='df_var_info', header=0)
    df_Q_trans = transform(df_Q, var_info, start, end)
    df_M_trans = transform(df_M, var_info, start, end)
    return df_Q_trans, df_M_trans, var_info


def load_data_NN(data_path):
    # load data
    df_Q = pd.read_excel(data_path, sheet_name='df_Q', header=0)
    df_M = pd.read_excel(data_path, sheet_name='df_M', header=0)
    var_info = pd.read_excel(data_path, sheet_name='df_var_info', header=0)

    return df_Q, df_M, var_info
