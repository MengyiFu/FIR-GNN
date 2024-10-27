import ast
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


def parse_time(time_str):
    date, time, ampm = time_str.split(' ')
    hour, minute, second = map(int, time.split(':')[:3])
    if ampm == '下午' and hour != 12:
        hour += 12
    elif ampm == '上午' and hour == 12:
        hour = 0
    return f'{date} {hour:02d}:{minute:02d}:{second:02d}'


def visualize_graph(edge):
    G = nx.Graph()
    G.add_edges_from(edge)
    nx.draw(G)
    plt.show()


if __name__ == "__main__":
    dataset = {'ISCX-VPN': 0, 'ISCX-NonVPN': 1, 'BoT-IoT': 2, 'CICIDS': 3}
    dataset_num = 3
    label_level = 'Application'

    '''
    graphdata_dir: The generated FIRG data.
    rawdata_dir: The preprocessed traffic flow features (packet-level features & flow-level features).
    '''
    if dataset_num == 0:
        graphdata_dir = f'datasets/FIRG/VPN/{label_level}/'
        rawdata_dir = 'datasets/ISCX-VPN-NonVPN/VPN-CSV-pktlen'
    elif dataset_num == 1:
        graphdata_dir = f'datasets/FIRG/NonVPN/{label_level}/'
        rawdata_dir = 'datasets/ISCX-VPN-NonVPN/NonVPN-CSV-pktlen'
    elif dataset_num == 2:
        graphdata_dir = f'datasets/FIRG/BoT-IoT/{label_level}/'
        rawdata_dir = 'datasets/BoT-IoT/BoT-IoT_CSV_pktlen'
    else:
        graphdata_dir = f'datasets/FIRG/CICIDS/{label_level}/'
        rawdata_dir = 'datasets/CICIDS/CICIDS_CSV_pktlen'

    # Merge the CSV of node features and label the flow.--------------------------------------------------------------
    csv_list = glob.glob(os.path.join(rawdata_dir, '*.csv'))

    # File name segmentation. Before '_' is the coarse classification, and after '_' is the fine classification.
    all_data = []
    for file in csv_list:
        print(f'read {file}')
        FIRG = pd.read_csv(file)
        FIRG.drop(columns=['Label'], inplace=True)

        # Get the flow label
        function_label, application_label = str(file).split('/')[-1].split('.')[0].split('_')
        FIRG['Function'] = function_label
        FIRG['Application'] = application_label
        all_data.append(FIRG)
    combined_data = pd.concat(all_data, ignore_index=True)

    combined_data.drop_duplicates(inplace=True)
    combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_data.dropna(inplace=True)
    print(combined_data['Function'].value_counts())
    print(combined_data['Application'].value_counts())

    combined_data = combined_data.groupby(label_level).apply(
        lambda x: x.sample(n=len(x)) if len(x) < 3000 else x.sample(n=3000, random_state=42))
    print(combined_data[label_level].value_counts())

    # Feature Normalization
    columns = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
               'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
               'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
               'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
               'Fwd IAT Min',
               'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
               'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
               'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
               'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count',
               'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
               'Fwd Byts/b Avg',
               'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
               'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
               'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
               'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

    scaler = MinMaxScaler()
    combined_data[columns] = scaler.fit_transform(combined_data[columns])

    # Sort in ascending order by timestamp and number the nodes.
    combined_data['Timestamp'] = combined_data['Timestamp'].apply(parse_time)
    combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'], format='%d/%m/%Y %H:%M:%S')
    combined_data['Timestamp'] = combined_data['Timestamp'].apply(lambda x: x.timestamp())
    combined_data.sort_values(by='Timestamp', inplace=True)
    combined_data = combined_data.reset_index(drop=True).reset_index()

    # Output the normalized nodes.csv.
    combined_data.to_csv(f'{graphdata_dir}nodes.csv', index=False)

    # Extract the five-tuple and generate the adjacency matrix.-----------------------------------------------------
    pkt_num = 3
    T = 10

    combined_data = pd.read_csv(f'{graphdata_dir}nodes.csv')
    combined_data['Pkt Len'] = combined_data['Pkt Len'].apply(ast.literal_eval)
    nodes = combined_data[['Src IP', 'Dst IP', 'Timestamp', 'Pkt Len']].values

    with open(f'{graphdata_dir}edges_T{T}_P{pkt_num}.csv', 'a') as file:
        file.write('Src,Dst,Delta_Start_Time,Pkt_Len\n')
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                delta_t = nodes[j][2] - nodes[i][2]
                pkt_len = nodes[i][3][:pkt_num] + nodes[j][3][:pkt_num]

                if delta_t <= T:
                    # # If the source IPs are the same, there are bidirectional edges.
                    if nodes[i][0] == nodes[j][0]:
                        print(f'There are bidirectional edges between node {i} and node {j}.')
                        file.write(f'{i},{j},{delta_t},"{pkt_len}"\n{j},{i},{delta_t},"{pkt_len}"\n')
                    # If the destination IP of i is the source IP of j, then there is a unidirectional edge of i -> j.
                    elif nodes[i][1] == nodes[j][0]:
                        print(f'There is a unidirectional edge of i -> j between node {i} and node {j}.')
                        file.write(f'{i},{j},{delta_t},"{pkt_len}"\n')
                else:
                    continue