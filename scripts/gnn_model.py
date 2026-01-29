#!/usr/bin/env python3
"""
Graph Neural Network model for movie recommendations.
This module provides GNN-based movie recommendation functionality.

Note: This implementation requires PyTorch and PyTorch Geometric.
If these are not installed, the Flask app will fall back to basic recommendations.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

# Try to import PyTorch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch/PyG not available: {e}")
    print("GNN features will be disabled. Install PyTorch and PyTorch Geometric to enable GNN recommendations.")
    TORCH_AVAILABLE = False
    
    # Mock classes for when PyTorch is not available
    class nn:
        class Module:
            pass
    
    class torch:
        @staticmethod
        def load(*args, **kwargs):
            raise ImportError("PyTorch not available")
        
        @staticmethod
        def save(*args, **kwargs):
            raise ImportError("PyTorch not available")

class MovieGNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Graph Neural Network for movie recommendation.
    
    Architecture:
    - Multiple GNN layers (GCN, GAT, or GraphSAGE)
    - Node embeddings for movies, users, actors, directors, genres
    - Rating prediction head
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, 
                 num_layers: int = 2, conv_type: str = 'gcn', dropout: float = 0.1):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for rating prediction)
            num_layers: Number of GNN layers
            conv_type: Type of graph convolution ('gcn', 'gat', 'sage')
            dropout: Dropout rate
        """
        if not TORCH_AVAILABLE:
            return
            
        super(MovieGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if conv_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif conv_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        elif conv_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GNN model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph-level prediction
            
        Returns:
            Node embeddings or graph-level predictions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # For graph-level prediction
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Prediction
        out = self.predictor(x)
        
        return x, out  # Return both embeddings and predictions

class MovieRecommender:
    """
    Movie recommendation system using trained GNN model.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the recommender with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.metadata = None
        
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}")
    
    def load_model(self):
        """Load the trained model and associated metadata."""
        if not TORCH_AVAILABLE:
            return
        
        # Determine device for model loading
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            
            # Initialize model with saved configuration
            model_config = checkpoint.get('model_config', {})
            self.model = MovieGNN(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Move model to appropriate device
            self.model = self.model.to(device)
            
            # Load scaler and metadata
            self.scaler = checkpoint.get('scaler')
            self.metadata = checkpoint.get('metadata', {})
            
            print(f"Model loaded successfully from {self.model_path} on {device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def get_movie_embedding(self, movie_id: int, graph_data_path: str) -> Optional[np.ndarray]:
        """
        Get the learned embedding for a specific movie.
        
        Args:
            movie_id: Movie ID
            graph_data_path: Path to graph data file
            
        Returns:
            Movie embedding vector or None if not found
        """
        if not TORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # Load graph data
            graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
            
            # Find movie index
            if movie_id not in self.metadata.get('movie_id_to_idx', {}):
                return None
            
            movie_idx = self.metadata['movie_id_to_idx'][movie_id]
            
            # Get embeddings
            with torch.no_grad():
                embeddings, _ = self.model(graph_data.x, graph_data.edge_index)
                movie_embedding = embeddings[movie_idx].numpy()
            
            return movie_embedding
            
        except Exception as e:
            print(f"Failed to get movie embedding: {e}")
            return None
    
    def get_similar_movies(self, movie_id: int, graph_data_path: str, top_k: int = 5) -> List[int]:
        """
        Find movies similar to the given movie using learned embeddings.
        
        Args:
            movie_id: Target movie ID
            graph_data_path: Path to graph data file
            top_k: Number of similar movies to return
            
        Returns:
            List of similar movie indices
        """
        if not TORCH_AVAILABLE or self.model is None:
            return []
        
        try:
            # Load graph data
            graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
            
            # Find movie index
            if movie_id not in self.metadata.get('movie_id_to_idx', {}):
                return []
            
            movie_idx = self.metadata['movie_id_to_idx'][movie_id]
            
            # Get all embeddings
            with torch.no_grad():
                embeddings, _ = self.model(graph_data.x, graph_data.edge_index)
            
            # Calculate similarities using cosine similarity
            target_embedding = embeddings[movie_idx:movie_idx+1]  # Keep batch dimension
            similarities = F.cosine_similarity(embeddings, target_embedding, dim=1)
            
            # Get top-k most similar (excluding the target movie itself)
            similarities[movie_idx] = -1  # Exclude self
            top_indices = similarities.argsort(descending=True)[:top_k]
            
            return top_indices.tolist()
            
        except Exception as e:
            print(f"Failed to get similar movies: {e}")
            return []
    
    def recommend_for_user(self, user_id: int, graph_data_path: str, top_k: int = 10) -> List[int]:
        """
        Generate movie recommendations for a specific user.
        
        Args:
            user_id: Target user ID
            graph_data_path: Path to graph data file
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended movie IDs
        """
        if not TORCH_AVAILABLE or self.model is None:
            return []
        
        try:
            # Load graph data and metadata
            graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
            
            # This is a simplified recommendation approach
            # In a full implementation, you would:
            # 1. Get user embedding
            # 2. Find movies the user hasn't rated
            # 3. Predict ratings for unrated movies
            # 4. Return top-k highest predicted ratings
            
            # For now, return random movie indices as placeholder
            # You can implement more sophisticated logic here
            num_movies = len(self.metadata.get('movie_id_to_idx', {}))
            if num_movies == 0:
                return []
            
            # Return random movies as placeholder
            import random
            available_indices = list(range(min(num_movies, top_k * 2)))
            return random.sample(available_indices, min(top_k, len(available_indices)))
            
        except Exception as e:
            print(f"Failed to get user recommendations: {e}")
            return []

def create_model_directory():
    """Create models directory if it doesn't exist."""
    os.makedirs('models', exist_ok=True)

# Export functions for use in other modules
__all__ = ['MovieGNN', 'MovieRecommender', 'TORCH_AVAILABLE', 'create_model_directory']