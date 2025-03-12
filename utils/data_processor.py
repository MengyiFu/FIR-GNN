import ast
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, f_classif, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch


class DataProcessor:
    def __init__(self, config: Dict, graph: str):
        self.config = config
        self.graph_path = graph
        self.drop_columns = ['index', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Function',
                'Application', 'Pkt Len']
        self.selected_features = []

    def load_data(self) -> [pd.DataFrame, pd.DataFrame]:
        node_path = os.path.join(self.graph_path, 'nodes.csv')
        if not os.path.exists(node_path):
            raise FileNotFoundError(f"File {node_path} not exists.")
        node_data = pd.read_csv(node_path)

        edge_path = os.path.join(self.graph_path, f'edges_T{self.config["dataset"]["window_t"]}_P{self.config["dataset"]["pkt_num"]}.csv')
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"File {edge_path} not exists.")
        edge_data = pd.read_csv(edge_path)
        return node_data, edge_data

    def split_data(self, node_data):
        n_samples = len(node_data)
        train_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        # 对于每一个不同的标签类别，进行分层抽样
        for label in node_data[self.config['dataset']['label_level']].unique():
            # 获取该类别的所有样本索引
            indices = node_data[node_data[self.config['dataset']['label_level']] == label].index
            # 使用train_test_split函数进行分层抽样
            train_indices, test_indices = train_test_split(indices, train_size=self.config['train']['label_ratio'], shuffle=True,
                                                           random_state=42)
            # 更新掩码
            train_mask[train_indices] = True
            test_mask[test_indices] = True
        # 确认划分正确
        # assert (train_mask + test_mask).all(), "Some samples are not assigned to either set."
        assert not (train_mask & test_mask).any(), "Some samples are assigned to both sets."
        return train_mask, test_mask

    def feature_selection(self, node_data, train_mask):
        method = self.config.get('fs_method', 'SHAP')
        k = self.config.get('feature_num', 20)
        # 准备数据
        X = node_data.drop(columns=self.drop_columns + ['Label'])
        y = node_data['Label']
        X_train = X.iloc[train_mask]
        y_train = y.iloc[train_mask]

        # 执行特征选择
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_train, y_train)
            selected = X.columns[selector.get_support()]

        elif method == 'RFE':
            estimator = LogisticRegression(max_iter=1000)
            selector = RFE(estimator, n_features_to_select=k)
            selector.fit(X_train, y_train)
            selected = X.columns[selector.get_support()]

        elif method == 'RFI':
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            indices = np.argsort(clf.feature_importances_)[-k:]
            selected = X.columns[indices]

        elif method == 'FSFS':
            clf = RandomForestClassifier(n_estimators=5)
            selector = SFS(clf,
                           k_features=k,
                           forward=True,
                           cv=3,
                           scoring='accuracy')
            selector.fit(X_train, y_train)
            selected = list(selector.k_feature_names_)

        elif method == 'IG':
            selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(X_train, y_train)
            selected = X.columns[selector.get_support()]

        else:  # SHAP方法使用预设特征
            selected = [
                           'Bwd IAT Tot', 'Bwd IAT Max', 'Fwd IAT Tot',
                           'Flow Duration', 'Flow IAT Max', 'Bwd IAT Std',
                           'Init Fwd Win Byts', 'Flow Byts/s', 'Fwd IAT Max',
                           'Bwd Pkts/s', 'Flow Pkts/s', 'Fwd Pkts/s',
                           'TotLen Fwd Pkts', 'Flow IAT Std', 'Pkt Len Var',
                           'Fwd IAT Std', 'Subflow Fwd Byts', 'Fwd Header Len',
                           'Fwd Pkt Len Max'
                       ][:k]

        self.selected_features = list(selected)
        print(self.selected_features)
        return self.selected_features

    def create_pyg_data(self, node_data, edge_data):
        # 划分掩码
        train_mask, test_mask = self.split_data(node_data)

        if self.config["train"]['exp'] == 'FS':
            self.feature_selection(node_data, train_mask)
            x = node_data[self.selected_features]
        else:
            x = node_data.drop(columns=self.drop_columns)

        x_tensor = torch.tensor(x.values, dtype=torch.float)
        y_tensor = torch.tensor(node_data['Label'].values, dtype=torch.long)
        num_class = torch.max(y_tensor).item() + 1

        edge_index = torch.tensor(edge_data[['src', 'dst']].values.T, dtype=torch.long)

        if 'Pkt_Len' in edge_data.columns:
            edge_attr = edge_data['Pkt_Len'].apply(ast.literal_eval)
            edge_attr = torch.tensor(np.stack(edge_attr.values), dtype=torch.float)
        else:
            edge_attr = None

        return Data(
            x=x_tensor,
            edge_index=edge_index.contiguous(),
            edge_attr=edge_attr,
            y=y_tensor,
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool)
        ), num_class