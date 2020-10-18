# -*- coding: utf-8 -*-
# @Time       : 2020/10/08 21:41:48
# @Author     : RegiusQuant <315135833@qq.com>
# @Project    : Regius-AI
# @File       : metrics.py
# @Description: 评估指标


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def calc_regression_metrics(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
