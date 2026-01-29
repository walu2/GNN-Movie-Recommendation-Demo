#!/usr/bin/env python3
"""
Train Graph Neural Network model for movie recommendations.
"""

import os
import json
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Try to import required libraries
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch/scikit-learn not available: {e}")
    print("Please install PyTorch, PyTorch Geometric, and scikit-learn to train GNN models.")
    TORCH_AVAILABLE = False
    sys.exit(1)

# Import our GNN model
try:
    from gnn_model import MovieGNN, create_model_directory
except ImportError as e:
    print(f"Failed to import GNN model: {e}")
    sys.exit(1)

class GNNTrainer:
    """Trainer class for Graph Neural Network models."""
    
    def __init__(self, graph_data_path: str, metadata_path: str):
        """
        Initialize the trainer.
        
        Args:
            graph_data_path: Path to graph data file
            metadata_path: Path to metadata JSON file
        """
        self.graph_data_path = graph_data_path
        self.metadata_path = metadata_path
        
        # Load data
        self.load_data()
        
        # Training parameters - Support Apple Silicon GPU (MPS)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: CUDA GPU - {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using device: Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: CPU")
        
        print(f"Device: {self.device}")
        
    def load_data(self):
        """Load graph data and metadata."""
        print("Loading graph data...")
        
        # Load graph data
        if not os.path.exists(self.graph_data_path):
            print(f"ERROR: Graph data not found at {self.graph_data_path}")
            print("Please run 'python3 scripts/extract_graph_data.py' first.")
            sys.exit(1)
        
        self.graph_data = torch.load(self.graph_data_path, map_location='cpu', weights_only=False)
        
        # Load metadata
        if not os.path.exists(self.metadata_path):
            print(f"ERROR: Metadata not found at {self.metadata_path}")
            sys.exit(1)
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded graph with {self.graph_data.num_nodes} nodes and {self.graph_data.edge_index.shape[1]} edges")
        
    def create_rating_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create dataset for rating prediction task.
        
        Returns:
            Tuple of (train_edges, train_ratings, test_edges, test_ratings)
        """
        print("Creating rating prediction dataset...")
        
        # Find rating edges
        rating_edges = []
        rating_values = []
        
        edge_index = self.graph_data.edge_index
        edge_attr = self.graph_data.edge_attr
        
        # Edge features: [is_rated, is_belongs_to, is_acted_in, is_directed_by, rating_value]
        for i in range(edge_index.shape[1]):
            edge_features = edge_attr[i]
            if edge_features[0] == 1.0:  # Is a rating edge
                src, dst = edge_index[:, i]
                rating_edges.append([src.item(), dst.item()])
                rating_values.append(edge_features[4].item() * 5.0)  # Denormalize rating
        
        if len(rating_edges) == 0:
            print("ERROR: No rating edges found in the graph data.")
            sys.exit(1)
        
        rating_edges = torch.tensor(rating_edges, dtype=torch.long)
        rating_values = torch.tensor(rating_values, dtype=torch.float)
        
        print(f"Found {len(rating_edges)} rating edges")
        
        # Split into train/test
        indices = torch.randperm(len(rating_edges))
        split_idx = int(0.8 * len(rating_edges))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_edges = rating_edges[train_indices]
        train_ratings = rating_values[train_indices]
        test_edges = rating_edges[test_indices]
        test_ratings = rating_values[test_indices]
        
        print(f"Train set: {len(train_edges)} ratings")
        print(f"Test set: {len(test_edges)} ratings")
        
        return train_edges, train_ratings, test_edges, test_ratings
    
    def train_model(self, model_config: Dict[str, Any], train_config: Dict[str, Any]):
        """
        Train the GNN model.
        
        Args:
            model_config: Model configuration parameters
            train_config: Training configuration parameters
        """
        print(f"Training GNN model with config: {model_config}")
        
        # Create model
        input_dim = self.graph_data.x.shape[1]
        model = MovieGNN(input_dim=input_dim, **model_config)
        model = model.to(self.device)
        
        # Move data to device
        graph_data = self.graph_data.to(self.device)
        
        # Create rating dataset
        train_edges, train_ratings, test_edges, test_ratings = self.create_rating_dataset()
        train_edges = train_edges.to(self.device)
        train_ratings = train_ratings.to(self.device)
        test_edges = test_edges.to(self.device)
        test_ratings = test_ratings.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=train_config['learning_rate'],
                                   weight_decay=train_config['weight_decay'])
        
        # Training loop
        model.train()
        best_test_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {train_config['num_epochs']} epochs...")
        print("-" * 50)
        
        for epoch in range(train_config['num_epochs']):
            start_time = time.time()
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get node embeddings
            node_embeddings, _ = model(graph_data.x, graph_data.edge_index)
            
            # Predict ratings for training edges
            src_embeddings = node_embeddings[train_edges[:, 0]]
            dst_embeddings = node_embeddings[train_edges[:, 1]]
            
            # Simple prediction: dot product of embeddings
            pred_ratings = torch.sum(src_embeddings * dst_embeddings, dim=1)
            pred_ratings = torch.sigmoid(pred_ratings) * 4 + 1  # Scale to 1-5
            
            # Calculate loss
            train_loss = F.mse_loss(pred_ratings, train_ratings)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Evaluation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    # Test predictions
                    test_src_embeddings = node_embeddings[test_edges[:, 0]]
                    test_dst_embeddings = node_embeddings[test_edges[:, 1]]
                    test_pred_ratings = torch.sum(test_src_embeddings * test_dst_embeddings, dim=1)
                    test_pred_ratings = torch.sigmoid(test_pred_ratings) * 4 + 1
                    
                    test_loss = F.mse_loss(test_pred_ratings, test_ratings)
                    
                    # Calculate metrics
                    test_pred_np = test_pred_ratings.cpu().numpy()
                    test_true_np = test_ratings.cpu().numpy()
                    
                    mse = mean_squared_error(test_true_np, test_pred_np)
                    mae = mean_absolute_error(test_true_np, test_pred_np)
                    
                    elapsed_time = time.time() - start_time
                    
                    print(f"Epoch {epoch:3d} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Test Loss: {test_loss:.4f} | "
                          f"MSE: {mse:.4f} | "
                          f"MAE: {mae:.4f} | "
                          f"Time: {elapsed_time:.2f}s")
                    
                    # Early stopping
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        patience_counter = 0
                        
                        # Save best model
                        self.save_model(model, model_config, train_config, epoch, test_loss)
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= train_config['patience']:
                            print(f"Early stopping at epoch {epoch}")
                            break
                
                model.train()
        
        print("-" * 50)
        print(f"Training completed! Best test loss: {best_test_loss:.4f}")
        
    def save_model(self, model: MovieGNN, model_config: Dict[str, Any], 
                   train_config: Dict[str, Any], epoch: int, test_loss: float):
        """Save the trained model."""
        create_model_directory()
        
        # Create filename based on configuration
        conv_type = model_config.get('conv_type', 'gcn')
        num_layers = model_config.get('num_layers', 2)
        filename = f"gnn_{conv_type}_{num_layers}layers.pt"
        
        model_path = os.path.join('models', filename)
        
        # Save model checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'train_config': train_config,
            'metadata': self.metadata,
            'epoch': epoch,
            'test_loss': test_loss
        }
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to: {model_path}")

def main():
    """Main training function."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available. Please install PyTorch and dependencies.")
        sys.exit(1)
    
    print("=== GNN Model Training ===\n")
    
    # Configuration
    model_configs = [
        # GCN configurations
        {
            'conv_type': 'gcn',
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1
        },
        {
            'conv_type': 'gcn',
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2
        },
    ]
    
    # Optimize training config for GPU acceleration
    train_config = {
        'num_epochs': 300,  # More epochs since GPU training is faster
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'patience': 25
    }
    
    # Check for graph data
    graph_data_path = 'data/graph_data.pt'
    metadata_path = 'data/metadata.json'
    
    if not os.path.exists(graph_data_path):
        print(f"ERROR: Graph data not found at {graph_data_path}")
        print("Please run 'python3 scripts/extract_graph_data.py' first to extract graph data from Neo4j.")
        sys.exit(1)
    
    # Initialize trainer
    trainer = GNNTrainer(graph_data_path, metadata_path)
    
    # Train models with different configurations
    for i, model_config in enumerate(model_configs, 1):
        print(f"\n{'='*60}")
        print(f"Training Model {i}/{len(model_configs)}")
        print(f"{'='*60}")
        
        try:
            trainer.train_model(model_config, train_config)
            print(f"✅ Model {i} training completed successfully!")
            
        except Exception as e:
            print(f"❌ Model {i} training failed: {e}")
            continue
        
        print()
    
    print("🎉 All training completed!")
    print("\nTo use the trained models:")
    print("1. Start the web application: python3 app.py")
    print("2. The app will automatically load the trained GNN model")
    print("3. Test GNN-powered recommendations through the web interface")

if __name__ == "__main__":
    main()