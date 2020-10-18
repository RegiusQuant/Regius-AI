# -*- coding: utf-8 -*-
# @Time       : 2020/10/08 20:09:56
# @Author     : RegiusQuant <315135833@qq.com>
# @Project    : Regius-AI
# @File       : preprocessing.py
# @Description: 数据预处理

import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def generate_scaler_and_encoder(file_path_list, pkl_path):
    temp_data_list = []
    for file_path in file_path_list:
        temp_data = pd.read_csv(file_path)
        temp_data_list.append(temp_data)
    all_data = pd.concat(temp_data_list)

    cont_data = all_data.filter(regex="X_*")
    standard_scaler = StandardScaler()
    standard_scaler.fit(cont_data)

    cate_data = all_data.filter(regex="C_*")
    label_encoders = []
    for col in cate_data.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(cate_data[col])
        label_encoders.append(label_encoder)

    # for col, label_encoder in zip(cate_data.columns, label_encoders):
    #     print(col, label_encoder.classes_)

    pkl_dict = {"StandardScaler": standard_scaler, "LabelEncoders": label_encoders}
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_dict, f)


def load_processed_data(file_path_list, pkl_path):
    temp_data_list = []
    for file_path in file_path_list:
        temp_data = pd.read_csv(file_path)
        temp_data_list.append(temp_data)
    if len(temp_data_list) == 1:
        raw_data = temp_data_list[0]
    else:
        raw_data = pd.concat(temp_data_list)

    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)

    cont_cols = raw_data.filter(regex="X_*").columns
    raw_data[cont_cols] = pkl_dict["StandardScaler"].transform(raw_data[cont_cols])

    cate_cols = raw_data.filter(regex="C_*").columns
    for i, col in enumerate(cate_cols):
        raw_data[col] = pkl_dict["LabelEncoders"][i].transform(raw_data[col])

    return raw_data
