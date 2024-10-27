import ast
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from ptflops import get_model_complexity_info
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, Linear, BatchNorm
from torch_geometric.profile import profileit, timeit, count_parameters, get_model_size, get_data_size


def df_to_tensor(df):
    df = df.to_numpy()
    return torch.from_numpy(df)


def visualize_tsne(h, color):
    z = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.detach().cpu().numpy(), cmap="Set2")
    plt.show()


# 'VPN', 'NonVPN', 'BoT-IoT', 'CICIDS'
dataset = 'BoT-IoT'
exp = 'LR'
model_name = 'GCN'
label_level = 'Application'
# 超参数
label_ratio = 0.4
hidden = 32
hidden_1 = 16
heads = 4
lr = 0.01
epoch = 300

drop_columns = ['index', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Function',
                'Application', 'Pkt Len']
node_csv = f'./datasets/FIRG/{dataset}/{label_level}/nodes.csv'
edge_csv = f'./datasets/FIRG/{dataset}/{label_level}/edges.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # 读取节点特征和边
    node_data = pd.read_csv(node_csv)
    print(node_data[label_level].value_counts())

    # y
    le = LabelEncoder()
    node_label = le.fit_transform(node_data[label_level])
    y = torch.from_numpy(node_label)

    # x
    node_attr = node_data.drop(columns=drop_columns)
    x = df_to_tensor(node_attr)

    # edge_index
    edge_data = pd.read_csv(edge_csv)
    edge_weight = edge_data.pop('Delta_Start_Time')
    edge_data['Pkt_Len'] = edge_data['Pkt_Len'].apply(ast.literal_eval)
    edge_attr = np.array(edge_data['Pkt_Len'].tolist())
    edge_data = edge_data.drop(columns=['Pkt_Len'])

    # edge_weight = torch.from_numpy(edge_weight)
    edge_attr = torch.from_numpy(edge_attr)
    edge_index = df_to_tensor(edge_data)

    # 划分训练集和测试集(train_mask, test_mask) 有标签和无标签的比例
    # 创建一个空的掩码数组
    n_samples = len(y)
    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    # 对于每一个不同的标签类别，进行分层抽样
    for label in node_data[label_level].unique():
        # 获取该类别的所有样本索引
        indices = node_data[node_data[label_level] == label].index
        # 使用train_test_split函数进行分层抽样
        train_indices, test_indices = train_test_split(indices, train_size=label_ratio, shuffle=True, random_state=42)
        # 更新掩码
        train_mask[train_indices] = True
        test_mask[test_indices] = True
    # 确认划分正确
    # assert (train_mask + test_mask).all(), "Some samples are not assigned to either set."
    assert not (train_mask & test_mask).any(), "Some samples are assigned to both sets."

    # 输出划分后的标签分布
    print("Train label distribution:")
    print(node_data[train_mask][label_level].value_counts())
    print("Test label distribution:")
    print(node_data[test_mask][label_level].value_counts())

    # 生成图数据(x, edge_index, edge_attr, y) torch.Tensor
    # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], test_mask=[2708])
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    num_classes = torch.unique(data.y).size(0)
    data = data.to(device)
    print()
    print(f'Dataset: {dataset}')
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of node features: {data.num_node_features}')
    print(f'Number of classes: {num_classes}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Number of testing nodes: {data.test_mask.sum()}')
    print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(data.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x


    # GAT
    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, heads):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GATConv(data.num_node_features, hidden_channels, heads, concat=False)
            self.conv2 = GATConv(hidden_channels, num_classes, heads, concat=False)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x


    # GCN_GAT
    class GCN_GAT(torch.nn.Module):
        def __init__(self, hidden_channels, hidden_channels1, heads):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(data.num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels1, heads, concat=False)
            self.conv3 = GCNConv(hidden_channels1, num_classes)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv3(x, edge_index)
            return x


    # 创建模型
    if model_name == 'GCN':
        model = GCN(hidden_channels=hidden).to(device)
    elif model_name == 'GAT':
        model = GAT(hidden_channels=hidden, heads=heads).to(device)
    else:
        model = GCN_GAT(hidden_channels=hidden, hidden_channels1=hidden_1, heads=heads).to(device)
    print(model)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    @profileit("cuda")
    def train(model, optimizer, x, edge_index, y):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)


    train_time = 0
    max_active_gpu = 0
    for epoch in range(1, epoch):
        loss, stats = train(model, optimizer, data.x, data.edge_index, data.y)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        train_time += stats.time
        max_active_gpu = stats.max_active_gpu
        # print(stats)
    print(f'Training time: {train_time:.4f}s, Max_active_gpu: {max_active_gpu:.4f}MB')


    @torch.no_grad()
    def test(model, x, edge_index):
        return model(x, edge_index)


    with timeit() as t:
        z = test(model, data.x, data.edge_index)
    time = t.duration
    total_params = count_parameters(model)
    model_size = get_model_size(model)
    data_size = get_data_size(data)
    print(f'Testing time: {time:.4f}s, Parameters: {total_params}, '
          f'Model Size: {model_size / (1024 * 1024):.4f}MB, Data Size: {data_size / (1024 * 1024):.4f}MB')

    pred = z.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    y_pred = pred[data.test_mask].detach().cpu().numpy()
    y_true = data.y[data.test_mask].detach().cpu().numpy()
    test_report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
    print(test_report)

    result_dir = f'./result/{exp}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # pred_true = pd.DataFrame(columns=['y_pred', 'y_true'])
    # pred_true['y_pred'] = y_pred
    # pred_true['y_true'] = y_true
    report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4, output_dict=True)
    df = pd.DataFrame(report).transpose()

    # pred_true.to_csv(os.path.join(result_dir, f'{dataset}_{model_name}_{label_level}_pred.csv'), index=False)
    df.to_csv(os.path.join(result_dir, f'{dataset}_{model_name}_{label_level}_report.csv'), index=True)