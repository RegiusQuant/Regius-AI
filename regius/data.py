# -*- coding: utf-8 -*-
# @Time       : 2020/10/18 15:47:59
# @Author     : RegiusQuant <315135833@qq.com>
# @Project    : Regius-AI
# @File       : data.py
# @Description: 深度学习模型数据集定义


from torch.utils.data import Dataset


class DeepDenseDataset(Dataset):

    def __init__(self, x_cont, x_cate, y):
        self.x_cont = x_cont
        self.x_cate = x_cate
        self.y = y

    def __getitem__(self, idx):
        return self.x_cont[idx], self.x_cate[idx], self.y[idx]

    def __len__(self):
        return len(self.x_cont)


class WideDeepDataset(Dataset):

    def __init__(self, x_wide, x_cont, x_cate, y):
        self.x_wide = x_wide
        self.x_cont = x_cont
        self.x_cate = x_cate
        self.y = y

    def __getitem__(self, idx):
        return self.x_wide[idx], self.x_cont[idx], self.x_cate[idx], self.y[idx]

    def __len__(self):
        return len(self.x_cont)
