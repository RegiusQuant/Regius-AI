# -*- coding: utf-8 -*-
# @Time       : 2020/10/08 20:14:28
# @Author     : RegiusQuant <315135833@qq.com>
# @Project    : Regius-AI
# @File       : main.py
# @Description: 主程序入口

import pytorch_lightning as pl

from regius.preprocessing import generate_scaler_and_encoder
# from regius.models.randomforest import run_cross_validation, run_final_predict, run_random_shuffle_test
# from regius.models.xgboost import run_cross_validation, run_final_predict, run_random_shuffle_test
# from regius.models.lightgbm import run_cross_validation, run_final_predict, run_random_shuffle_test
# from regius.models.deepdense import run_cross_validation, run_final_predict, run_random_shuffle_test
# from regius.models.widedeep import run_cross_validation, run_final_predict, run_random_shuffle_test
from regius.models.tabnet import run_cross_validation, run_final_predict, run_random_shuffle_test

# FOLDER_NAME = "meteorological_aod"
# FOLDER_NAME = "meteorological_reflectance"
FOLDER_NAME = "only_aod"
# FOLDER_NAME = "only_reflectance"


if __name__ == "__main__":
    pl.seed_everything(42)

    file_path_list = [
        f"data/{FOLDER_NAME}/data2016.csv",
        f"data/{FOLDER_NAME}/data2017.csv",
        f"data/{FOLDER_NAME}/data2018.csv",
        f"data/{FOLDER_NAME}/data2019.csv"
    ]
    pkl_path = f"pickle/{FOLDER_NAME}.pkl"
    generate_scaler_and_encoder(file_path_list, pkl_path)

    # run_cross_validation(file_path_list[:3], pkl_path)
    run_final_predict(file_path_list[:3], file_path_list[-1], pkl_path, FOLDER_NAME)
    # run_random_shuffle_test(file_path_list, pkl_path, FOLDER_NAME)
