# -*- coding: utf-8 -*-
# @Time       : 2020/10/18 18:52:42
# @Author     : Jiang Yize <yize.jiang@galixir.com>
# @Project    : Regius-AI
# @Description: 深度学习TabNet模型


import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split

from regius.preprocessing import load_processed_data
from regius.metrics import calc_regression_metrics


def run_cross_validation(file_path_list, pkl_path):
    assert len(file_path_list) >= 3

    cv_metric_hist = defaultdict(list)
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)

    for i in range(len(file_path_list)):
        print("Run Fold:", i)
        train_data = load_processed_data(file_path_list[:i] + file_path_list[i + 1:], pkl_path)
        valid_data = load_processed_data([file_path_list[i]], pkl_path)

        cont_cols = train_data.filter(regex='X_*').columns
        cate_cols = train_data.filter(regex="C_*").columns
        cate_dims = [len(e.classes_) for e in pkl_dict["LabelEncoders"]]
        feat_cols = list(cont_cols) + list(cate_cols)
        cate_idxs = [i for i, f in enumerate(feat_cols) if f in cate_cols]

        x_train, x_valid = train_data[feat_cols].values, valid_data[feat_cols].values
        y_train, y_valid = train_data["Y"].values.reshape(-1, 1), valid_data["Y"].values.reshape(-1, 1)

        model = TabNetRegressor(
            cat_dims=cate_dims,
            cat_idxs=cate_idxs,
            n_steps=3,
        )

        model.fit(
            X_train=x_train,
            y_train=y_train,
            eval_set=[(x_valid, y_valid)],
            max_epochs=10,
            batch_size=1024,
            num_workers=4,
            drop_last=False,
        )
        y_pred = model.predict(x_valid)

        metric_result = calc_regression_metrics(y_valid, y_pred)
        for k, v in metric_result.items():
            cv_metric_hist[k].append(v)

    cv_result = {k: np.mean(v) for k, v in cv_metric_hist.items()}
    print("CV Result:", cv_result)


def run_final_predict(train_file_path_list, test_file_path, pkl_path, folder_name):
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)

    train_data = load_processed_data(train_file_path_list, pkl_path)
    test_data = load_processed_data([test_file_path], pkl_path)

    cont_cols = train_data.filter(regex='X_*').columns
    cate_cols = train_data.filter(regex="C_*").columns
    cate_dims = [len(e.classes_) for e in pkl_dict["LabelEncoders"]]
    feat_cols = list(cont_cols) + list(cate_cols)
    cate_idxs = [i for i, f in enumerate(feat_cols) if f in cate_cols]

    x_train, x_test = train_data[feat_cols].values, test_data[feat_cols].values
    y_train, y_test = train_data["Y"].values.reshape(-1, 1), test_data["Y"].values.reshape(-1, 1)

    model = TabNetRegressor(
        cat_dims=cate_dims,
        cat_idxs=cate_idxs,
        n_steps=3,
    )
    model.fit(
        X_train=x_train,
        y_train=y_train,
        max_epochs=10,
        batch_size=1024,
        num_workers=4,
        drop_last=False,
    )

    model_path = f"model/{folder_name}/tabnet"
    model.save_model(model_path)
    model.load_model(model_path + ".zip")
    y_pred = model.predict(x_test)
    print("Final Result:", calc_regression_metrics(y_test, y_pred))

    col_name = "TABNET_RESULT"
    test_data[col_name] = y_pred
    _, name = os.path.split(test_file_path)
    result_path = f"result/{folder_name}/model_tabnet_{name}"
    test_data.to_csv(result_path, index=False)


def run_random_shuffle_test(file_path_list, pkl_path, folder_name):
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)

    train_data = load_processed_data(file_path_list, pkl_path)
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    cont_cols = train_data.filter(regex='X_*').columns
    cate_cols = train_data.filter(regex="C_*").columns
    cate_dims = [len(e.classes_) for e in pkl_dict["LabelEncoders"]]
    feat_cols = list(cont_cols) + list(cate_cols)
    cate_idxs = [i for i, f in enumerate(feat_cols) if f in cate_cols]

    x_train, x_test = train_data[feat_cols].values, test_data[feat_cols].values
    y_train, y_test = train_data["Y"].values.reshape(-1, 1), test_data["Y"].values.reshape(-1, 1)

    model = TabNetRegressor(
        cat_dims=cate_dims,
        cat_idxs=cate_idxs,
        n_steps=3,
    )
    model.fit(
        X_train=x_train,
        y_train=y_train,
        max_epochs=10,
        batch_size=1024,
        num_workers=4,
        drop_last=False,
    )

    y_pred = model.predict(x_test)
    print("Random Shuffle Result:", calc_regression_metrics(y_test, y_pred))

    result_data = pd.DataFrame({
        "True": y_test.squeeze(),
        "Predict": y_pred.squeeze()
    })
    result_path = f"result/{folder_name}/model_tabnet_randomshuffle.csv"
    result_data.to_csv(result_path, index=False)
