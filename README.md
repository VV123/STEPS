# STEPS: Efficient Unlearning for Spatio-temporal Graphs

[![Paper](https://ojs.aaai.org/index.php/AAAI/article/view/35259](https://ojs.aaai.org/index.php/AAAI/article/view/35259)

This repository contains the implementation of **STEPS** (Spatio-Temporal graph complexity Evaluation, Partition-aggregation strategy, and Sub-model scale), an efficient unlearning framework for spatio-temporal graphs presented at AAAI-25.

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{guo2025efficient,
  title={Efficient Unlearning for Spatio-temporal Graph (Student Abstract)},
  author={Guo, Qiming and Pan, Chen and Zhang, Hua and Wang, Wenlu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={28},
  pages={29382--29384},
  year={2025}
}
```

## 🎯 Overview

STEPS is a novel framework designed to address the challenges of machine unlearning in spatio-temporal graph neural networks. It efficiently removes the influence of specific training data while maintaining model accuracy and preserving data continuity and integrity. This is particularly important for compliance with privacy regulations like GDPR and CCPA, and for applications in environmental monitoring, urban systems, and location-based services.

## 🚀 Key Features

- **Efficient Unlearning**: Significantly reduces unlearning time compared to full model retraining
- **Flexible Partitioning**: Supports multiple graph partitioning methods (Spectral, Louvain, Metis)
- **Customizable Aggregation**: Various aggregation strategies (Weighted Average, Mean, Median, MLP)
- **Minimal Accuracy Loss**: Maintains model performance even with up to 15% node removal
- **STGCN Architecture**: Uses Spatio-Temporal Graph Convolutional Networks for modeling

## 📦 Installation

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/STEPS-unlearning.git
cd STEPS-unlearning
```

2. Create a conda environment using the provided configuration:
```bash
conda env create -f environment.yml
conda activate steps
```

3. Required packages:
```python
torch
torch.nn
torch.optim
torch_geometric
pandas
numpy
sklearn
networkx
metis
community
argparse
datetime
logging
```

## 📊 Usage

### Step 1: Data Preparation

Prepare your spatio-temporal graph data:
- **Adjacency matrix**: Save as `Adj.csv` (or `adj_traffic.csv`)
- **Time series data**: Save as `Depth_s.xls` (or `Flow_s.xlsx`)

The framework expects:
- Adjacency matrix in CSV format with node indices
- Time series data in Excel format with temporal sequences

### Step 2: Complexity Analysis

Before running the unlearning framework, analyze your graph's complexity to determine optimal partitioning parameters:

```bash
python complexity.py
```

The `complexity.py` script:
- Calculates the spatio-temporal graph complexity using Equation (1) from the paper
- Provides recommendations for model scale (low/medium/high complexity)
- Suggests optimal number of partitions based on your data

Example output:
```
Data shape: 170 nodes, 17856 time steps, 3 features
Medium complexity (324.56). A moderate-sized model may be appropriate.
Moderate partitioning recommended. Consider 4-6 partitions.
```

### Step 3: Run STEPS Unlearning

#### Basic Usage

```bash
python STEPs.py \
    --epochs 50 \
    --num_partitions 4 \
    --partition_method spectral \
    --aggregation_method weighted_average \
    --removal_rates 0.05,0.10,0.15
```

#### Advanced Configuration

```bash
python STEPs.py \
    --epochs 100 \
    --num_partitions 6 \
    --partition_method louvain \
    --aggregation_method mean \
    --removal_rates 0.05,0.10,0.15 \
    --num_gcn_layers 2 \
    --num_lstm_layers 1 \
    --hidden_features 32 \
    --sub_num_gcn_layers 2 \
    --sub_num_lstm_layers 1 \
    --sub_hidden_features 16
```

#### Command Line Arguments

- `--epochs`: Number of training epochs (default: 50)
- `--num_partitions`: Number of graph partitions (default: 4)
- `--partition_method`: Method for graph partitioning [`spectral`, `louvain`, `metis`] (default: spectral)
- `--aggregation_method`: Method for aggregating predictions [`mean`, `median`, `weighted_average`, `mlp`] (default: weighted_average)
- `--removal_rates`: Comma-separated node removal rates (default: 0.05,0.1,0.15)
- `--num_gcn_layers`: GCN layers for full model (default: 2)
- `--num_lstm_layers`: LSTM layers for full model (default: 1)
- `--hidden_features`: Hidden features for full model (default: 32)
- `--sub_num_gcn_layers`: GCN layers for sub-models (default: 2)
- `--sub_num_lstm_layers`: LSTM layers for sub-models (default: 1)
- `--sub_hidden_features`: Hidden features for sub-models (default: 16)

### Step 4: Custom Partitioning and Aggregation Methods

You can extend the framework with custom methods:

#### Custom Partitioning
```python
# Add to partition_graph() function in STEPs.py
elif method == 'your_custom_method':
    # Your partitioning logic
    partition = your_partitioning_function(G, num_partitions)
    # Return subgraphs
```

#### Custom Aggregation
```python
# Add to aggregate_predictions() function in STEPs.py
elif method == 'your_custom_aggregation':
    # Your aggregation logic
    return your_aggregation_function(subgraph_predictions)
```

### Step 5: Results Analysis

The framework automatically saves results to `stgcn_detailed_results.xlsx` including:
- Model performance metrics (RMSE, MAE, MAPE, R², Trend F1)
- Training and inference times
- Comparison between full model, partitioned model, and unlearned models
- Impact of different removal rates

## 🧪 Experiments

### Datasets

The framework has been tested on:
- **RWW Dataset**: Real-world water system dataset (urban water monitoring)
- **PEMS08**: Traffic flow dataset

### Performance Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Trend F1**: F1 score for trend prediction
- **Training Time**: Time for model training/retraining
- **Aggregation Time**: Time for prediction aggregation

### Example Results

With default Spectral partitioning and Weighted Average aggregation:
- 5% node removal: ~0.585 RMSE
- 10% node removal: ~0.584 RMSE  
- 15% node removal: ~0.589 RMSE

## 📁 Project Structure

```
STEPS-unlearning/
├── STEPs.py              # Main STEPS framework implementation
├── complexity.py          # Graph complexity analysis tool
├── environment.yml        # Conda environment configuration
├── data/
│   ├── Adj.csv           # Adjacency matrix
│   ├── adj_traffic.csv   # Alternative adjacency matrix
│   ├── Depth_s.xls       # Water depth time series
│   └── Flow_s.xlsx       # Traffic flow time series
├── stgcn_log.txt         # Training logs
├── stgcn_detailed_results.xlsx  # Experimental results
└── README.md             # This file
```

## 🔧 Hyperparameter Guidelines

Based on complexity analysis results:

- **Low Complexity** (C < 100): 
  - 2-3 partitions
  - Small sub-models (hidden_features: 16)
  - Fewer layers (1 GCN, 1 LSTM)

- **Medium Complexity** (100 ≤ C < 500): 
  - 4-6 partitions
  - Moderate sub-models (hidden_features: 32)
  - Standard layers (2 GCN, 1-2 LSTM)

- **High Complexity** (C ≥ 500): 
  - 7+ partitions
  - Large sub-models (hidden_features: 64+)
  - Deep layers (2+ GCN, 2+ LSTM)

## 📊 Output Format

Results are saved to Excel with the following columns:
- Performance metrics (MSE, MAE, RMSE, MAPE, R2, Trend_F1)
- Model configuration (epochs, partitions, methods)
- Timing information (training_time, aggregation_time)
- Model type (full, partitioned, updated_X%)

## 📧 Contact

For questions or support, please contact:
- Wenlu Wang: wenlu.wang.1@gmail.com; wenlu.wang@tamucc.edu
- Qiming Guo: qguo2@islander.tamucc.edu; 
