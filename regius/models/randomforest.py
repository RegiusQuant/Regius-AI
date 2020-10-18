# -*- coding: utf-8 -*-
# @Time       : 2020/10/17 23:08:24
# @Author     : RegiusQuant <315135833@qq.com>
# @Project    : Regius-AI
# @File       : randomforest.py
# @Description: 随机森林模型

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from regius.preprocessing import load_processed_data
from regius.metrics import calc_regression_metrics


def run_cross_validation(file_path_list, pkl_path):
    assert len(file_path_list) >= 3

    cv_metric_hist = defaultdict(list)

    for i in range(len(file_path_list)):
        print("Run Fold:", i)
        train_data = load_processed_data(file_path_list[:i] + file_path_list[i + 1:], pkl_path)
        valid_data = load_processed_data([file_path_list[i]], pkl_path)

        feature_cols = train_data.filter(regex="X_*|C_*").columns
        target_col = "Y"

        x_train, y_train = train_data[feature_cols].values, train_data[target_col].values
        x_valid, y_valid = valid_data[feature_cols].values, valid_data[target_col].values

        model = RandomForestRegressor(
            n_estimators=1000,
            criterion="mse",
            max_depth=5,
            max_features=0.8,
            max_samples=0.8,
            n_jobs=6,
            random_state=42,
            verbose=1,
        )
        model.fit(x_train, y_train)

        y_pred = model.predict(x_valid)
        metric_result = calc_regression_metrics(y_valid, y_pred)
        for k, v in metric_result.items():
            cv_metric_hist[k].append(v)

    cv_result = {k: np.mean(v) for k, v in cv_metric_hist.items()}
    print("CV Result:", cv_result)


def run_final_predict(train_file_path_list, test_file_path, pkl_path, folder_name):
    train_data = load_processed_data(train_file_path_list, pkl_path)
    test_data = load_processed_data([test_file_path], pkl_path)

    feature_cols = train_data.filter(regex="X_*|C_*").columns
    target_col = "Y"

    x_train, y_train = train_data[feature_cols].values, train_data[target_col].values
    x_test, y_test = test_data[feature_cols].values, test_data[target_col].values

    model = RandomForestRegressor(
        n_estimators=1000,
        criterion="mse",
        max_depth=5,
        max_features=0.8,
        max_samples=0.8,
        n_jobs=6,
        random_state=42,
        verbose=1,
    )
    model.fit(x_train, y_train)

    model_path = f"model/{folder_name}/randomforest.pkl"
    joblib.dump(model, model_path)
    model = joblib.load(model_path)
    y_pred = model.predict(x_test)
    print("Final Result:", calc_regression_metrics(y_test, y_pred))

    col_name = "RF_RESULT"
    test_data[col_name] = y_pred
    _, name = os.path.split(test_file_path)
    result_path = f"result/{folder_name}/model_randomforest_{name}"
    test_data.to_csv(result_path, index=False)


def run_random_shuffle_test(file_path_list, pkl_path, folder_name):
    train_data = load_processed_data(file_path_list, pkl_path)

    feature_cols = train_data.filter(regex="X_*|C_*").columns
    target_col = "Y"

    x_train, y_train = train_data[feature_cols].values, train_data[target_col].values
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=1000,
        criterion="mse",
        max_depth=5,
        max_features=0.8,
        max_samples=0.8,
        n_jobs=6,
        random_state=42,
        verbose=1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Random Shuffle Result:", calc_regression_metrics(y_test, y_pred))

    result_data = pd.DataFrame({
        "True": y_test,
        "Predict": y_pred
    })
    result_path = f"result/{folder_name}/model_randomforest_randomshuffle.csv"
    result_data.to_csv(result_path, index=False)
