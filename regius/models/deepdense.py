# -*- coding: utf-8 -*-
# @Time       : 2020/10/18 15:50:53
# @Author     : RegiusQuant <315135833@qq.com>
# @Project    : Regius-AI
# @File       : deep.py
# @Description: 深度学习DeepDense模型

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from regius.preprocessing import load_processed_data
from regius.data import DeepDenseDataset
from regius.metrics import calc_regression_metrics


ARGS = {
    "embedding_dim": 16,
    "dropout_prob": 0.2,
    "hidden_nodes": 128,
    "hidden_layers": 2,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "max_epochs": 10,
    "sigmoid_coef": 7
}


class DeepDense(nn.Module):

    def __init__(self, cont_num, cate_input):
        super(DeepDense, self).__init__()
        self.cate_input = cate_input

        self.embed_layers = nn.ModuleDict({
            'embed_layer_' + str(i): nn.Embedding(num, ARGS["embedding_dim"])
            for i, c, num in cate_input
        })
        self.embed_dropout = nn.Dropout(ARGS["dropout_prob"])

        in_dim = ARGS["embedding_dim"] * len(cate_input) + cont_num
        nodes = [in_dim] + [ARGS["hidden_nodes"]] * ARGS["hidden_layers"]
        self.dense_layers = nn.Sequential()
        for i in range(1, len(nodes)):
            self.dense_layers.add_module(
                'dense_layer_{}'.format(i - 1),
                self._create_dense_layer(
                    nodes[i - 1],
                    nodes[i],
                    ARGS["dropout_prob"]
                )
            )
        self.out_layer = nn.Linear(nodes[-1], 1)

    def _create_dense_layer(self, in_dim, out_dim, drop_p):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_p)
        )

    def forward(self, x_cont, x_cate):
        if x_cate is not None:
            x_emb = [
                self.embed_layers['embed_layer_' + str(i)](
                    x_cate[:, i].long())
                for i, c, num in self.cate_input
            ]
            x_emb = torch.cat(x_emb, 1)
            x_emb = self.embed_dropout(x_emb)

            x_out = torch.cat([x_emb, x_cont], 1)
        else:
            x_out = x_cont
        x_out = self.dense_layers(x_out)
        return self.out_layer(x_out)

    def get_embeddings(self):
        result = {}
        for n, p in self.named_parameters():
            if 'embed_layer' in n:
                result[n.split('.')[1]] = p.cpu().data.numpy()
        return result


class Wide(nn.Module):

    def __init__(self, wide_num):
        super(Wide, self).__init__()
        self.linear = nn.Linear(wide_num, 1, bias=False)

    def forward(self, x_wide):
        return self.linear(x_wide.float())


class WideDeep(nn.Module):

    def __init__(self, wide_num, cont_num, cate_input):
        super(WideDeep, self).__init__()
        self.wide = Wide(wide_num)
        self.deepdense = DeepDense(cont_num, cate_input)

    def forward(self, x_wide, x_cont, x_cate):
        x_out = self.wide(x_wide)
        x_out.add_(self.deepdense(x_cont, x_cate))
        return x_out


class DeepDenseModel(pl.LightningModule):

    def __init__(self, cont_num, cate_input):
        super().__init__()
        self.model = DeepDense(cont_num, cate_input)

    def forward(self, x_cont, x_cate):
        out = self.model(x_cont, x_cate)
        return nn.Sigmoid()(out) * ARGS["sigmoid_coef"]

    def training_step(self, batch, batch_idx):
        x_cont, x_cate, y_true = batch[0].float(), batch[1].long(), batch[2].float()
        y_pred = self(x_cont, x_cate).squeeze()
        loss = F.mse_loss(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x_cont, x_cate, _ = batch[0].float(), batch[1].long(), batch[2].float()
        y_pred = self(x_cont, x_cate).squeeze()
        return y_pred

    def test_epoch_end(self, test_step_outputs):
        y_pred = []
        for out in test_step_outputs:
            y_pred.extend(out.cpu().numpy())
        return {"predict": np.array(y_pred)}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=ARGS["learning_rate"])


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
        x_cont_train, x_cont_valid = train_data[cont_cols].values, valid_data[cont_cols].values
        cont_num = x_cont_train.shape[1]

        cate_cols = train_data.filter(regex="C_*").columns
        cate_input = [(i, c, len(e.classes_)) for i, (c, e) in enumerate(zip(cate_cols, pkl_dict["LabelEncoders"]))]
        x_cate_train, x_cate_valid = train_data[cate_cols].values, valid_data[cate_cols].values

        y_train, y_valid = np.log1p(train_data["Y"].values), np.log1p(valid_data["Y"].values)

        train_set = DeepDenseDataset(x_cont_train, x_cate_train, y_train)
        valid_set = DeepDenseDataset(x_cont_valid, x_cate_valid, y_valid)
        train_loader = DataLoader(train_set, batch_size=ARGS["batch_size"], num_workers=4, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=ARGS["batch_size"], num_workers=4)

        model = DeepDenseModel(cont_num, cate_input)
        trainer = pl.Trainer(max_epochs=ARGS["max_epochs"], gpus=1)
        trainer.fit(model, train_loader)
        result = trainer.test(model, valid_loader)
        y_pred, y_valid = np.expm1(result[0]["predict"]), np.expm1(y_valid)

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
    x_cont_train, x_cont_test = train_data[cont_cols].values, test_data[cont_cols].values
    cont_num = x_cont_train.shape[1]

    cate_cols = train_data.filter(regex="C_*").columns
    cate_input = [(i, c, len(e.classes_)) for i, (c, e) in enumerate(zip(cate_cols, pkl_dict["LabelEncoders"]))]
    x_cate_train, x_cate_test = train_data[cate_cols].values, test_data[cate_cols].values

    y_train, y_test = np.log1p(train_data["Y"].values), np.log1p(test_data["Y"].values)

    train_set = DeepDenseDataset(x_cont_train, x_cate_train, y_train)
    test_set = DeepDenseDataset(x_cont_test, x_cate_test, y_test)
    train_loader = DataLoader(train_set, batch_size=ARGS["batch_size"], num_workers=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=ARGS["batch_size"], num_workers=4)

    model = DeepDenseModel(cont_num, cate_input)
    trainer = pl.Trainer(max_epochs=ARGS["max_epochs"], gpus=1)
    trainer.fit(model, train_loader)
    model_path = f"model/{folder_name}/deepdense.ckpt"
    trainer.save_checkpoint(model_path)

    model = DeepDenseModel.load_from_checkpoint(
        checkpoint_path=model_path,
        cont_num=cont_num,
        cate_input=cate_input,
    )
    result = trainer.test(model, test_loader)
    y_pred, y_test = np.expm1(result[0]["predict"]), np.expm1(y_test)
    print("Final Result:", calc_regression_metrics(y_test, y_pred))

    col_name = "DEEPDENSE_RESULT"
    test_data[col_name] = y_pred
    _, name = os.path.split(test_file_path)
    result_path = f"result/{folder_name}/model_deepdense_{name}"
    test_data.to_csv(result_path, index=False)


def run_random_shuffle_test(file_path_list, pkl_path, folder_name):
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)

    train_data = load_processed_data(file_path_list, pkl_path)
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    cont_cols = train_data.filter(regex='X_*').columns
    x_cont_train, x_cont_test = train_data[cont_cols].values, test_data[cont_cols].values
    cont_num = x_cont_train.shape[1]

    cate_cols = train_data.filter(regex="C_*").columns
    cate_input = [(i, c, len(e.classes_)) for i, (c, e) in enumerate(zip(cate_cols, pkl_dict["LabelEncoders"]))]
    x_cate_train, x_cate_test = train_data[cate_cols].values, test_data[cate_cols].values

    y_train, y_test = np.log1p(train_data["Y"].values), np.log1p(test_data["Y"].values)

    train_set = DeepDenseDataset(x_cont_train, x_cate_train, y_train)
    test_set = DeepDenseDataset(x_cont_test, x_cate_test, y_test)
    train_loader = DataLoader(train_set, batch_size=ARGS["batch_size"], num_workers=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=ARGS["batch_size"], num_workers=4)

    model = DeepDenseModel(cont_num, cate_input)
    trainer = pl.Trainer(max_epochs=ARGS["max_epochs"], gpus=1)
    trainer.fit(model, train_loader)

    result = trainer.test(model, test_loader)
    y_pred, y_test = np.expm1(result[0]["predict"]), np.expm1(y_test)
    print("Random Shuffle Result:", calc_regression_metrics(y_test, y_pred))

    result_data = pd.DataFrame({
        "True": y_test,
        "Predict": y_pred
    })
    result_path = f"result/{folder_name}/model_deepdense_randomshuffle.csv"
    result_data.to_csv(result_path, index=False)
