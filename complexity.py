import numpy as np
import pandas as pd

def calculate_complexity(adj_matrix, time_series_data, alpha=1, beta=0.5, gamma=0.3):
    """
    Calculate the complexity of a spatio-temporal graph.
    """
    num_nodes = adj_matrix.shape[0]
    num_edges = np.sum(adj_matrix) / 2  
    num_time_steps = time_series_data.shape[1]
    num_features = time_series_data.shape[2]
    
    structural_complexity = (num_edges / num_nodes) * np.log(num_nodes)
    temporal_complexity = num_time_steps
    feature_complexity = num_features * np.log(num_features)
    
    C = alpha * structural_complexity + beta * temporal_complexity + gamma * feature_complexity
    
    return C

def interpret_complexity(C, threshold1=100, threshold2=500):
    """
    Interpret the complexity score and provide recommendations.
    """
    if C < threshold1:
        return f"Low complexity ({C:.2f}). Consider using a small-scale model."
    elif C < threshold2:
        return f"Medium complexity ({C:.2f}). A moderate-sized model may be appropriate."
    else:
        return f"High complexity ({C:.2f}). Consider using a large-scale model or advanced techniques."

def suggest_partitioning(C, threshold1=100, threshold2=500):
    """
    Suggest a partitioning strategy based on the complexity score.
    """
    if C < threshold1:
        return "Minimal partitioning needed. Consider 2-3 partitions."
    elif C < threshold2:
        return "Moderate partitioning recommended. Consider 4-6 partitions."
    else:
        return "Extensive partitioning advised. Consider 7+ partitions or hierarchical approaches."

def load_data(adj_matrix_file, time_series_file):
    """
    Load adjacency matrix and time series data from CSV or Excel files.
    """
    if adj_matrix_file.endswith('.csv'):
        adj_matrix = pd.read_csv(adj_matrix_file, index_col=0).values
    elif adj_matrix_file.endswith(('.xlsx', '.xls')):
        adj_matrix = pd.read_excel(adj_matrix_file, index_col=0).values
    else:
        raise ValueError("Unsupported file format for adjacency matrix")

    if time_series_file.endswith('.csv'):
        time_series_data = pd.read_csv(time_series_file).values
    elif time_series_file.endswith(('.xlsx', '.xls')):
        time_series_data = pd.read_excel(time_series_file).values
    else:
        raise ValueError("Unsupported file format for time series data")

    num_nodes = adj_matrix.shape[0]
    num_features = time_series_data.shape[1] // num_nodes
    num_time_steps = time_series_data.shape[0]
    time_series_data = time_series_data.T.reshape(num_nodes, num_features, num_time_steps).transpose(0, 2, 1)

    return adj_matrix, time_series_data

def main():
    adj_matrix_file = "adj_traffic.csv"  # or .xlsx
    time_series_file = "Flow_s.xlsx"  # or .xlsx
    
    adj_matrix, time_series_data = load_data(adj_matrix_file, time_series_file)
    
    num_nodes, num_time_steps, num_features = time_series_data.shape
    print(f"Data shape: {num_nodes} nodes, {num_time_steps} time steps, {num_features} features")

    C = calculate_complexity(adj_matrix, time_series_data)

    print(interpret_complexity(C))
    print(suggest_partitioning(C))

if __name__ == "__main__":
    main()