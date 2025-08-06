import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
import logging
import json
import os
import sys
import networkx as nx
from torch_geometric.utils import from_networkx
import random
from sklearn.cluster import SpectralClustering
from torch.utils.data import Dataset, DataLoader
import metis
import community as community_louvain
import argparse
import time
from datetime import datetime
from sklearn.metrics import f1_score, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ADJ_PATH = "Adj.csv"
ADJ_PATH = "adj_traffic.csv"
FLOW_PATH = "Flow_s.xlsx"

logging.basicConfig(filename='stgcn_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_features, out_features, num_gcn_layers=2, num_lstm_layers=2):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_features, hidden_features))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_features, hidden_features))
        self.lstm = nn.LSTM(hidden_features, hidden_features, num_layers=num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_nodes, seq_len, num_features)
        x = x.reshape(batch_size * num_nodes, seq_len, num_features)
        x_gcn = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size * num_nodes, num_features)
            for gcn in self.gcn_layers:
                x_t = gcn(x_t, edge_index)
                x_t = torch.relu(x_t)
            x_gcn.append(x_t)
        x = torch.stack(x_gcn, dim=1)  # (batch_size * num_nodes, seq_len, hidden_features)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = x.reshape(batch_size, num_nodes, -1)
        return x


class GraphDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(train_size=0.7, val_size=0.15, test_size=0.15):
    df = pd.read_excel(FLOW_PATH)
    adj_matrix = pd.read_csv(ADJ_PATH, index_col=0).values

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df.values)

    sequence_length = 12
    X, y = [], []
    for i in range(len(normalized_data) - sequence_length):
        X.append(normalized_data[i:i+sequence_length])
        y.append(normalized_data[i+sequence_length])

    X = np.array(X)
    y = np.array(y)


    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    edge_index = torch.LongTensor(adj_matrix).nonzero().t().contiguous().to(device)

    num_nodes = adj_matrix.shape[0]

    logging.info(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]:.2%})")
    logging.info(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/X.shape[0]:.2%})")
    logging.info(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]:.2%})")

    return (X_train, y_train, X_val, y_val, X_test, y_test), edge_index, num_nodes, adj_matrix


def partition_graph(adj_matrix, num_partitions, method='spectral'):
    G = nx.from_numpy_array(adj_matrix)

    if method == 'spectral':

        sc = SpectralClustering(n_clusters=num_partitions, affinity='precomputed', n_init=100, assign_labels='discretize', random_state=42)
        labels = sc.fit_predict(adj_matrix)


        partition = [[] for _ in range(num_partitions)]
        for node, label in enumerate(labels):
            partition[label].append(node)

    elif method == 'metis':

        _, parts = metis.part_graph(G, nparts=num_partitions)
        partition = [[] for _ in range(num_partitions)]
        for node, part in enumerate(parts):
            partition[part].append(node)

    elif method == 'louvain':

        partition_dict = community_louvain.best_partition(G)
        unique_labels = set(partition_dict.values())
        partition = [[] for _ in unique_labels]
        for node, label in partition_dict.items():
            partition[label].append(node)

    elif method == 'spatial':

        raise NotImplementedError("Spatial partitioning not implemented due to lack of node positions.")
    else:
        raise ValueError("Unsupported partitioning method")

    subgraphs = []
    for nodes in partition:
        subgraph = G.subgraph(nodes)
        data = from_networkx(subgraph)
        subgraphs.append((data, nodes)) 

    return subgraphs


def train_subgraph(model, train_data, val_data, edge_index, optimizer, criterion, epochs, node_indices, batch_size=32):
    X_train, y_train = train_data
    X_val, y_val = val_data


    X_train_sub = X_train[:, :, node_indices]
    y_train_sub = y_train[:, node_indices]
    X_val_sub = X_val[:, :, node_indices]
    y_val_sub = y_val[:, node_indices]


    train_dataset = GraphDataset(X_train_sub, y_train_sub)
    val_dataset = GraphDataset(X_val_sub, y_val_sub)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    node_mapping = {old: new for new, old in enumerate(node_indices)}
    mask = [(edge_index[0, i].item() in node_indices and edge_index[1, i].item() in node_indices) for i in range(edge_index.size(1))]
    new_edge_index = edge_index[:, mask]
    new_edge_index = new_edge_index.clone()
    for i in range(new_edge_index.size(1)):
        new_edge_index[0, i] = node_mapping[new_edge_index[0, i].item()]
        new_edge_index[1, i] = node_mapping[new_edge_index[1, i].item()]
    new_edge_index = new_edge_index.to(device)


    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch, new_edge_index)
            loss = criterion(output.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)


        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_output = model(X_val_batch, new_edge_index)
                val_loss = criterion(val_output.squeeze(-1), y_val_batch)
                total_val_loss += val_loss.item() * X_val_batch.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        if (epoch + 1) % 10 == 0:
            logging.info(f'Subgraph Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


    training_time = time.time() - start_time

    return avg_loss, avg_val_loss, training_time 


def aggregate_predictions(subgraph_predictions, method='mean', subgraph_sizes=None):
    if method == 'mean':
        return torch.mean(torch.stack(subgraph_predictions), dim=0)
    elif method == 'median':
        return torch.median(torch.stack(subgraph_predictions), dim=0)[0]
    elif method == 'weighted_average':
        if subgraph_sizes is None:

            weights = torch.ones(len(subgraph_predictions), device=subgraph_predictions[0].device)
        else:
            weights = torch.tensor(subgraph_sizes, dtype=torch.float32, device=subgraph_predictions[0].device)
            weights = weights / weights.sum()
        weighted_preds = torch.stack([pred * weight for pred, weight in zip(subgraph_predictions, weights)])
        return weighted_preds.sum(dim=0)
    elif method == 'mlp':

        predictions = torch.stack(subgraph_predictions, dim=-1)  
        batch_size, num_nodes, num_subgraphs = predictions.shape
        predictions = predictions.view(-1, num_subgraphs) 
        

        mlp = nn.Sequential(
            nn.Linear(num_subgraphs, num_subgraphs * 2),
            nn.ReLU(),
            nn.Linear(num_subgraphs * 2, 1)
        ).to(predictions.device)
        

        output = mlp(predictions).view(batch_size, num_nodes)
        return output
    else:
        raise ValueError("Unsupported aggregation method")




def remove_nodes(adj_matrix, removal_rate):
    num_nodes = adj_matrix.shape[0]
    num_remove = int(num_nodes * removal_rate)
    nodes_to_remove = random.sample(range(num_nodes), num_remove)

    mask = np.ones(num_nodes, dtype=bool)
    mask[nodes_to_remove] = False

    new_adj_matrix = adj_matrix[np.ix_(mask, mask)]
    old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(mask)[0])}

    return new_adj_matrix, nodes_to_remove, old_to_new_indices, mask





def evaluate_model(predictions, y):
    with torch.no_grad():
        mse = nn.MSELoss()(predictions.squeeze(-1), y)
        mae = nn.L1Loss()(predictions.squeeze(-1), y)
        rmse = torch.sqrt(mse)
        mape = torch.mean(torch.abs((y - predictions.squeeze(-1)) / y)) * 100
        r2 = r2_score(y.cpu().numpy().flatten(), predictions.cpu().numpy().flatten())
        trend_f1_score = trend_f1(y, predictions.squeeze(-1))

    return {
        'MSE': mse.item(),
        'MAE': mae.item(),
        'RMSE': rmse.item(),
        'MAPE': mape.item(),
        'R2': r2,
        'Trend_F1': trend_f1_score
    }


def trend_f1(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    true_trend = np.sign(y_true[:, 1:] - y_true[:, :-1])
    pred_trend = np.sign(y_pred[:, 1:] - y_pred[:, :-1])


    true_trend = true_trend.flatten()
    pred_trend = pred_trend.flatten()


    mask = ~np.isnan(true_trend) & ~np.isnan(pred_trend)
    true_trend = true_trend[mask]
    pred_trend = pred_trend[mask]

    return f1_score(true_trend, pred_trend, average='weighted')


def save_results_to_excel(results, params, filename='stgcn_detailed_results.xlsx'):

    combined = {**results, **params}
    df_new = pd.DataFrame([combined])

    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(filename, index=False)

    print(f"Results saved to {filename}")


def main(epochs, num_partitions, partition_method, aggregation_method, removal_rates, num_gcn_layers=2, num_lstm_layers=2, hidden_features=64, sub_num_gcn_layers=2, sub_num_lstm_layers=2, sub_hidden_features=64):
    (X_train, y_train, X_val, y_val, X_test, y_test), edge_index, num_nodes, adj_matrix = load_data()

    print(f"Total number of nodes: {num_nodes}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge index max value: {edge_index.max()}")
    print(f"X_train shape: {X_train.shape}")

    criterion = nn.MSELoss()


    full_model = STGCN(num_nodes=num_nodes, num_features=1, hidden_features=hidden_features, out_features=1, num_gcn_layers=num_gcn_layers, num_lstm_layers=num_lstm_layers).to(device)
    optimizer = optim.Adam(full_model.parameters(), lr=0.01)

    start_time = time.time()
    all_node_indices = list(range(num_nodes))
    final_train_loss, final_val_loss, full_training_time = train_subgraph(full_model, (X_train, y_train), (X_val, y_val), edge_index, optimizer, criterion, epochs, all_node_indices, batch_size=32)


    full_model.eval()
    with torch.no_grad():
        full_predictions = full_model(X_test, edge_index)

    full_results = evaluate_model(full_predictions, y_test)
    full_results['model_type'] = 'full'
    full_results['training_time'] = full_training_time
    full_results['final_train_loss'] = final_train_loss
    full_results['final_val_loss'] = final_val_loss

    params = {
        'epochs': epochs,
        'num_partitions': num_partitions,
        'partition_method': partition_method,
        'aggregation_method': aggregation_method,
        'removal_rates': str(removal_rates),
        'num_gcn_layers': num_gcn_layers,
        'num_lstm_layers': num_lstm_layers,
        'hidden_features': hidden_features,
        'sub_num_gcn_layers': sub_num_gcn_layers,
        'sub_num_lstm_layers': sub_num_lstm_layers,
        'sub_hidden_features': sub_hidden_features,
        'model_type': 'full',
    }

    save_results_to_excel(full_results, params)


    subgraphs = partition_graph(adj_matrix, num_partitions, partition_method)
    subgraph_models = []


    total_subgraph_training_time = 0
    final_train_losses = []
    final_val_losses = []

    for i, (subgraph_data, node_indices) in enumerate(subgraphs):
        sub_model = STGCN(num_nodes=len(node_indices), num_features=1, hidden_features=sub_hidden_features, out_features=1, num_gcn_layers=sub_num_gcn_layers, num_lstm_layers=sub_num_lstm_layers).to(device)
        sub_optimizer = optim.Adam(sub_model.parameters(), lr=0.01)

        sub_train_loss, sub_val_loss, sub_training_time = train_subgraph(sub_model, (X_train, y_train), (X_val, y_val), edge_index, sub_optimizer, criterion, epochs, node_indices, batch_size=32)
        subgraph_models.append((sub_model, node_indices))
        total_subgraph_training_time += sub_training_time
        final_train_losses.append(sub_train_loss)
        final_val_losses.append(sub_val_loss)


    subgraph_predictions = []
    subgraph_sizes = []
    total_subgraph_inference_time = 0

    for sub_model, node_indices in subgraph_models:
        X_test_sub = X_test[:, :, node_indices]

        node_mapping = {old: new for new, old in enumerate(node_indices)}
        mask = [(edge_index[0, i].item() in node_indices and edge_index[1, i].item() in node_indices) for i in range(edge_index.size(1))]
        sub_edge_index = edge_index[:, mask]
        sub_edge_index = sub_edge_index.clone()
        for i in range(sub_edge_index.size(1)):
            sub_edge_index[0, i] = node_mapping[sub_edge_index[0, i].item()]
            sub_edge_index[1, i] = node_mapping[sub_edge_index[1, i].item()]
        sub_edge_index = sub_edge_index.to(device)

        start_inference_time = time.time()
        sub_pred = sub_model(X_test_sub, sub_edge_index)
        inference_time = time.time() - start_inference_time
        total_subgraph_inference_time += inference_time

        pred_full = torch.zeros(X_test.size(0), num_nodes, device=device)
        pred_full[:, node_indices] = sub_pred.squeeze(-1)
        subgraph_predictions.append(pred_full)
        subgraph_sizes.append(len(node_indices))


    start_aggregation_time = time.time()
    if aggregation_method in ['mean', 'median', 'weighted_average', 'mlp']:
        aggregated_predictions = aggregate_predictions(subgraph_predictions, method=aggregation_method, subgraph_sizes=subgraph_sizes)
    else:
        raise ValueError("Unsupported aggregation method")

    aggregation_time = time.time() - start_aggregation_time

    partitioned_results = evaluate_model(aggregated_predictions, y_test)
    partitioned_results['model_type'] = 'partitioned'
    partitioned_results['training_time'] = total_subgraph_training_time
    partitioned_results['aggregation_time'] = aggregation_time
    partitioned_results['final_train_loss'] = np.mean(final_train_losses)
    partitioned_results['final_val_loss'] = np.mean(final_val_losses)

    params['model_type'] = 'partitioned'
    save_results_to_excel(partitioned_results, params)


    for rate in removal_rates:

        new_adj_matrix, removed_nodes, old_to_new_indices, mask = remove_nodes(adj_matrix, rate)
        new_edge_index = torch.LongTensor(new_adj_matrix).nonzero().t().contiguous().to(device)
        new_num_nodes = new_adj_matrix.shape[0]


        new_X_train = X_train[:, :, mask]
        new_y_train = y_train[:, mask]
        new_X_val = X_val[:, :, mask]
        new_y_val = y_val[:, mask]
        new_X_test = X_test[:, :, mask]
        new_y_test = y_test[:, mask]


        retrained_subgraph_training_time = 0
        total_subgraph_inference_time = 0
        final_train_losses = []
        final_val_losses = []

        updated_subgraphs = []

        for i, (sub_model, node_indices) in enumerate(subgraph_models):

            updated_node_indices = [node for node in node_indices if node not in removed_nodes]
            if len(updated_node_indices) == 0:
                continue
            updated_node_indices_mapped = [old_to_new_indices[node] for node in updated_node_indices]


            sub_model = STGCN(num_nodes=len(updated_node_indices), num_features=1, hidden_features=sub_hidden_features, out_features=1, num_gcn_layers=sub_num_gcn_layers, num_lstm_layers=sub_num_lstm_layers).to(device)
            sub_optimizer = optim.Adam(sub_model.parameters(), lr=0.01)
            sub_train_loss, sub_val_loss, retraining_time = train_subgraph(sub_model, (new_X_train, new_y_train), (new_X_val, new_y_val), new_edge_index, sub_optimizer, criterion, epochs, updated_node_indices_mapped, batch_size=32)
            updated_subgraphs.append((sub_model, updated_node_indices_mapped))
            retrained_subgraph_training_time += retraining_time
            final_train_losses.append(sub_train_loss)
            final_val_losses.append(sub_val_loss)

        total_retraining_time = retrained_subgraph_training_time


        subgraph_predictions = []
        subgraph_sizes = []
        total_subgraph_inference_time = 0

        for sub_model, node_indices_mapped in updated_subgraphs:
            X_test_sub = new_X_test[:, :, node_indices_mapped]

            node_mapping = {old: new for new, old in enumerate(node_indices_mapped)}
            mask = [(new_edge_index[0, i].item() in node_indices_mapped and new_edge_index[1, i].item() in node_indices_mapped) for i in range(new_edge_index.size(1))]
            sub_edge_index = new_edge_index[:, mask]
            sub_edge_index = sub_edge_index.clone()
            for i in range(sub_edge_index.size(1)):
                sub_edge_index[0, i] = node_mapping[sub_edge_index[0, i].item()]
                sub_edge_index[1, i] = node_mapping[sub_edge_index[1, i].item()]
            sub_edge_index = sub_edge_index.to(device)

            start_inference_time = time.time()
            sub_pred = sub_model(X_test_sub, sub_edge_index)
            inference_time = time.time() - start_inference_time
            total_subgraph_inference_time += inference_time

            pred_full = torch.zeros(new_X_test.size(0), new_num_nodes, device=device)
            pred_full[:, node_indices_mapped] = sub_pred.squeeze(-1)
            subgraph_predictions.append(pred_full)
            subgraph_sizes.append(len(node_indices_mapped))


        start_aggregation_time = time.time()
        if aggregation_method in ['mean', 'median', 'weighted_average', 'mlp']:
            aggregated_predictions = aggregate_predictions(subgraph_predictions, method=aggregation_method, subgraph_sizes=subgraph_sizes)
        else:
            raise ValueError("Unsupported aggregation method")

        aggregation_time = time.time() - start_aggregation_time

        updated_results = evaluate_model(aggregated_predictions, new_y_test)
        updated_results['model_type'] = f'updated_{int(rate*100)}%'
        updated_results['retraining_time'] = total_retraining_time
        updated_results['aggregation_time'] = aggregation_time
        updated_results['final_train_loss'] = np.mean(final_train_losses) if final_train_losses else None
        updated_results['final_val_loss'] = np.mean(final_val_losses) if final_val_losses else None

        updated_params = params.copy()
        updated_params['removal_rate'] = rate
        save_results_to_excel(updated_results, updated_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate STGCN model.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_partitions', type=int, default=4, help='Number of partitions for the graph')
    parser.add_argument('--partition_method', type=str, default='spectral', help='Partition method')
    parser.add_argument('--aggregation_method', type=str, default='mean', help='Aggregation method')
    parser.add_argument('--removal_rates', type=str, default='0.05,0.1,0.15', help='Comma-separated list of node removal rates')
    parser.add_argument('--num_gcn_layers', type=int, default=2, help='Number of GCN layers for full model')
    parser.add_argument('--num_lstm_layers', type=int, default=1, help='Number of LSTM layers for full model')
    parser.add_argument('--hidden_features', type=int, default=32, help='Number of hidden features in GCN for full model')
    parser.add_argument('--sub_num_gcn_layers', type=int, default=2, help='Number of GCN layers for subgraph models')
    parser.add_argument('--sub_num_lstm_layers', type=int, default=1, help='Number of LSTM layers for subgraph models')
    parser.add_argument('--sub_hidden_features', type=int, default=16, help='Number of hidden features in GCN for subgraph models')
    args = parser.parse_args()

    removal_rates = [float(rate) for rate in args.removal_rates.split(',')]

    main(args.epochs, args.num_partitions, args.partition_method, args.aggregation_method, removal_rates,
         args.num_gcn_layers, args.num_lstm_layers, args.hidden_features,
         args.sub_num_gcn_layers, args.sub_num_lstm_layers, args.sub_hidden_features)
