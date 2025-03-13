import ast
import os
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm


class GraphConstructor:
    def __init__(self, config):
        self.config = config['dataset']
        self.dataset_dir = config['dataset']['dataset_dir']
        self.processed_dir = config['dataset']['processed_dir']
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def _parse_time(self, time_str: str) -> float:
        """
        Args:
            time_str: Original time string, with the format like "28/04/2016 4:50:44 下午"
        Returns:
            float: Unix Timestamp
        """
        try:
            date_part, time_part, period = time_str.split()
            hour, minute, second = map(int, time_part.split(':'))

            # Handle the conversion of the 12-hour clock system.
            if period == "下午" and hour != 12:
                hour += 12
            elif period == "上午" and hour == 12:
                hour = 0

            # Construct a standard time string.
            normalized_time = f"{date_part} {hour:02d}:{minute:02d}:{second:02d}"
            return pd.to_datetime(normalized_time, format='%d/%m/%Y %H:%M:%S').timestamp()
        except Exception as e:
            print(f"Time parsing error: {time_str} -> {str(e)}")
            return 0.0

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_columns = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
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

        try:
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(subset=feature_columns, inplace=True)

            data[feature_columns] = self.scaler.fit_transform(data[feature_columns])
            return data
        except Exception as e:
            print(f"Feature normalization failed: {str(e)}")
            return data

    def _process_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Add hierarchical labels.
            data['Label'] = self.label_encoder.fit_transform(data[self.config['label_level']])
            return data
        except KeyError:
            print("The `label_level` parameter is missing in the configuration.")
            return data


    def construct_nodes(self) -> pd.DataFrame:
        raw_dir = self.dataset_dir

        save_dir = os.path.join(
            self.processed_dir,
            self.config['label_level']
        )
        os.makedirs(save_dir, exist_ok=True)

        all_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        dfs = []

        print(f"Merging {len(all_files)} CSV files...")
        for file in tqdm(all_files):
            try:
                df = pd.read_csv(os.path.join(raw_dir, file))
                label = file.split('.')[0]
                df['Function'] = label.split('_')[0]
                df['Application'] = label.split('_')[1]
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                dfs.append(df)
            except Exception as e:
                print(f"Failed to read the file {file}: {str(e)}")
                continue

        # Merge data.
        combined = pd.concat(dfs, ignore_index=True)
        combined = self._normalize_features(combined)
        combined = self._process_labels(combined)

        combined = combined.groupby(self.config['label_level']).apply(
            lambda x: x.sample(len(x)) if len(x) < self.config['per_class'] else x.sample(n=self.config['per_class'], random_state=42))
        print(combined[self.config['label_level']].value_counts())

        # Parse timestamp
        combined['Timestamp'] = combined['Timestamp'].apply(self._parse_time)
        combined.sort_values(by='Timestamp', inplace=True)
        combined = combined.reset_index(drop=True).reset_index()

        # Save nodes.csv
        node_path = os.path.join(save_dir, 'nodes.csv')
        combined.to_csv(node_path, index=False)
        print(f"The node data has been saved to {node_path}.")
        return combined

    def construct_edges(self, window_t: int = 10, pkt_num: int = 3) -> pd.DataFrame:
        """
        Args:
            window_t: Time window threshold (in seconds)
            pkt_num: Number of packet lengths to be retained.
        Returns:
            Edge Dataframe (src, dst, delta_time, pkt_len)
        """
        # Load node data.
        node_path = os.path.join(
            self.processed_dir,
            self.config['label_level'],
            'nodes.csv'
        )

        try:
            nodes = pd.read_csv(node_path)
            nodes['Pkt Len'] = nodes['Pkt Len'].apply(ast.literal_eval)
        except FileNotFoundError:
            print(f"The node file {node_path} does not exist. Please run construct_nodes first.")
            return pd.DataFrame()

        # 初始化边列表
        edges = []
        node_records = nodes[['Src IP', 'Dst IP', 'Timestamp', 'Pkt Len']].values

        print("Start constructing edge relationships...")
        with tqdm(total=len(node_records) * (len(node_records) - 1) // 2) as pbar:
            for i in range(len(node_records)):
                for j in range(i + 1, len(node_records)):
                    delta_t = node_records[j][2] - node_records[i][2]

                    # Skip the pairs of nodes that are outside the time window.
                    if delta_t > window_t:
                        pbar.update(len(node_records) - i - 1)
                        break

                    # Merge packet length features.
                    pkt_len = node_records[i][3][:pkt_num] + node_records[j][3][:pkt_num]

                    # Judge the connection relationship.
                    src_ip_i, dst_ip_i = node_records[i][0], node_records[i][1]
                    src_ip_j, dst_ip_j = node_records[j][0], node_records[j][1]

                    # Case 1: Same Source IP -> Bidirectional edge
                    if src_ip_i == src_ip_j:
                        edges.append([i, j, delta_t, pkt_len])
                        edges.append([j, i, delta_t, pkt_len])
                    # Case 2: The destination IP of i is the source IP of j -> Unidirectional edge
                    elif dst_ip_i == src_ip_j:
                        edges.append([i, j, delta_t, pkt_len])

                    pbar.update(1)

        edge_df = pd.DataFrame(edges, columns=['src', 'dst', 'delta_time', 'pkt_len'])
        edge_path = os.path.join(
            self.processed_dir,
            self.config['label_level'],
            f'edges_T{window_t}_P{pkt_num}.csv'
        )
        edge_df.to_csv(edge_path, index=False)
        print(f"The edge data has been saved to {edge_path}.")
        return edge_df

    def process_dataset(self):
        print(f"Processing the dataset...")

        # Stage 1：Construct nodes.
        nodes = self.construct_nodes()
        if nodes.empty:
            print("Node construction failed. Terminate the process.")
            return

        # Stage 2：Construct edges.
        edges = self.construct_edges(
            window_t=self.config.get('window_t', 10),
            pkt_num=self.config.get('pkt_num', 3)
        )

        print(f"Dataset processing completed! Generated {len(nodes)} nodes, and {len(edges)} edges.")