#!/usr/bin/env python3
"""
Extract graph data from Neo4j for GNN training.
This script converts Neo4j graph data into PyTorch Geometric format.
"""

import os
import json
import pickle
import sys
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Neo4j imports
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    print("Neo4j driver not available. Please install: pip install neo4j")
    NEO4J_AVAILABLE = False

# Try to import PyTorch and PyTorch Geometric
try:
    import torch
    from torch_geometric.data import Data
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch/PyG not available. Please install PyTorch and PyTorch Geometric.")
    TORCH_AVAILABLE = False

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class GraphDataExtractor:
    """Extract and preprocess graph data from Neo4j for GNN training."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available")
            
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Connected to Neo4j successfully!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            sys.exit(1)
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def extract_nodes(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract all nodes from Neo4j with their features."""
        print("Extracting nodes from Neo4j...")
        
        nodes = []
        node_mappings = {
            'movie_id_to_idx': {},
            'user_id_to_idx': {},
            'person_id_to_idx': {},
            'genre_name_to_idx': {}
        }
        
        with self.driver.session() as session:
            # Extract movies
            print("  - Extracting movies...")
            result = session.run("""
                MATCH (m:Movie)
                RETURN m.id as id, m.title as title, m.year as year, 
                       m.rating as rating, m.budget as budget, m.box_office as box_office
                ORDER BY m.id
            """)
            
            for i, record in enumerate(result):
                movie_data = dict(record)
                nodes.append({
                    'node_id': i,
                    'original_id': movie_data['id'],
                    'node_type': 'movie',
                    'title': movie_data.get('title', ''),
                    'year': movie_data.get('year', 2000),
                    'rating': movie_data.get('rating', 5.0),
                    'budget': movie_data.get('budget', 1000000),
                    'box_office': movie_data.get('box_office', 1000000)
                })
                node_mappings['movie_id_to_idx'][movie_data['id']] = i
            
            movie_count = len(nodes)
            print(f"    Found {movie_count} movies")
            
            # Extract users
            print("  - Extracting users...")
            result = session.run("""
                MATCH (u:User)
                RETURN u.id as id, u.name as name, u.age as age, u.favorite_genres as favorite_genres
                ORDER BY u.id
            """)
            
            for record in result:
                user_data = dict(record)
                node_idx = len(nodes)
                nodes.append({
                    'node_id': node_idx,
                    'original_id': user_data['id'],
                    'node_type': 'user',
                    'name': user_data.get('name', ''),
                    'age': user_data.get('age', 30),
                    'favorite_genres': user_data.get('favorite_genres', [])
                })
                node_mappings['user_id_to_idx'][user_data['id']] = node_idx
            
            user_count = len(nodes) - movie_count
            print(f"    Found {user_count} users")
            
            # Extract persons (actors/directors)
            print("  - Extracting persons...")
            result = session.run("""
                MATCH (p:Person)
                RETURN p.id as id, p.name as name, p.birth_year as birth_year,
                       p.is_actor as is_actor, p.is_director as is_director
                ORDER BY p.id
            """)
            
            for record in result:
                person_data = dict(record)
                node_idx = len(nodes)
                nodes.append({
                    'node_id': node_idx,
                    'original_id': person_data['id'],
                    'node_type': 'person',
                    'name': person_data.get('name', ''),
                    'birth_year': person_data.get('birth_year', 1970),
                    'is_actor': person_data.get('is_actor', False),
                    'is_director': person_data.get('is_director', False)
                })
                node_mappings['person_id_to_idx'][person_data['id']] = node_idx
            
            person_count = len(nodes) - movie_count - user_count
            print(f"    Found {person_count} persons")
            
            # Extract genres
            print("  - Extracting genres...")
            result = session.run("""
                MATCH (g:Genre)
                RETURN g.name as name
                ORDER BY g.name
            """)
            
            for record in result:
                genre_data = dict(record)
                node_idx = len(nodes)
                nodes.append({
                    'node_id': node_idx,
                    'original_id': genre_data['name'],
                    'node_type': 'genre',
                    'name': genre_data.get('name', '')
                })
                node_mappings['genre_name_to_idx'][genre_data['name']] = node_idx
            
            genre_count = len(nodes) - movie_count - user_count - person_count
            print(f"    Found {genre_count} genres")
        
        print(f"Total nodes extracted: {len(nodes)}")
        
        return pd.DataFrame(nodes), node_mappings
    
    def extract_edges(self, node_mappings: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Extract all edges from Neo4j with their attributes."""
        print("Extracting edges from Neo4j...")
        
        edges = []
        
        with self.driver.session() as session:
            # Extract user ratings
            print("  - Extracting user ratings...")
            result = session.run("""
                MATCH (u:User)-[r:RATED]->(m:Movie)
                RETURN u.id as user_id, m.id as movie_id, r.rating as rating
            """)
            
            for record in result:
                user_idx = node_mappings['user_id_to_idx'].get(record['user_id'])
                movie_idx = node_mappings['movie_id_to_idx'].get(record['movie_id'])
                
                if user_idx is not None and movie_idx is not None:
                    edges.append((
                        user_idx, 
                        movie_idx, 
                        {'edge_type': 'rated', 'rating': float(record['rating'])}
                    ))
            
            rating_count = len(edges)
            print(f"    Found {rating_count} ratings")
            
            # Extract movie-genre relationships
            print("  - Extracting movie-genre relationships...")
            result = session.run("""
                MATCH (m:Movie)-[r:BELONGS_TO]->(g:Genre)
                RETURN m.id as movie_id, g.name as genre_name
            """)
            
            for record in result:
                movie_idx = node_mappings['movie_id_to_idx'].get(record['movie_id'])
                genre_idx = node_mappings['genre_name_to_idx'].get(record['genre_name'])
                
                if movie_idx is not None and genre_idx is not None:
                    edges.append((
                        movie_idx, 
                        genre_idx, 
                        {'edge_type': 'belongs_to'}
                    ))
            
            genre_edge_count = len(edges) - rating_count
            print(f"    Found {genre_edge_count} movie-genre relationships")
            
            # Extract actor-movie relationships
            print("  - Extracting actor-movie relationships...")
            result = session.run("""
                MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
                RETURN p.id as person_id, m.id as movie_id
            """)
            
            for record in result:
                person_idx = node_mappings['person_id_to_idx'].get(record['person_id'])
                movie_idx = node_mappings['movie_id_to_idx'].get(record['movie_id'])
                
                if person_idx is not None and movie_idx is not None:
                    edges.append((
                        person_idx, 
                        movie_idx, 
                        {'edge_type': 'acted_in'}
                    ))
            
            acted_in_count = len(edges) - rating_count - genre_edge_count
            print(f"    Found {acted_in_count} acted-in relationships")
            
            # Extract director-movie relationships
            print("  - Extracting director-movie relationships...")
            result = session.run("""
                MATCH (m:Movie)-[r:DIRECTED_BY]->(p:Person)
                RETURN m.id as movie_id, p.id as person_id
            """)
            
            for record in result:
                movie_idx = node_mappings['movie_id_to_idx'].get(record['movie_id'])
                person_idx = node_mappings['person_id_to_idx'].get(record['person_id'])
                
                if movie_idx is not None and person_idx is not None:
                    edges.append((
                        movie_idx, 
                        person_idx, 
                        {'edge_type': 'directed_by'}
                    ))
            
            directed_by_count = len(edges) - rating_count - genre_edge_count - acted_in_count
            print(f"    Found {directed_by_count} directed-by relationships")
        
        print(f"Total edges extracted: {len(edges)}")
        return edges
    
    def create_node_features(self, nodes_df: pd.DataFrame) -> np.ndarray:
        """Create numerical node features for GNN training."""
        print("Creating node features...")
        
        features = []
        
        for _, node in nodes_df.iterrows():
            node_features = []
            
            # Node type one-hot encoding
            node_types = ['movie', 'user', 'person', 'genre']
            for node_type in node_types:
                node_features.append(1.0 if node['node_type'] == node_type else 0.0)
            
            # Movie-specific features
            if node['node_type'] == 'movie':
                node_features.extend([
                    float(node.get('year', 2000)) / 100.0,  # Normalized year
                    float(node.get('rating', 5.0)) / 10.0,  # Normalized rating
                    np.log(float(node.get('budget', 1000000)) + 1) / 20.0,  # Log-normalized budget
                    np.log(float(node.get('box_office', 1000000)) + 1) / 20.0  # Log-normalized box office
                ])
            else:
                node_features.extend([0.0, 0.0, 0.0, 0.0])  # Padding for non-movies
            
            # User-specific features
            if node['node_type'] == 'user':
                node_features.append(float(node.get('age', 30)) / 100.0)  # Normalized age
            else:
                node_features.append(0.0)  # Padding for non-users
            
            # Person-specific features
            if node['node_type'] == 'person':
                node_features.extend([
                    float(node.get('birth_year', 1970)) / 100.0,  # Normalized birth year
                    1.0 if node.get('is_actor', False) else 0.0,  # Is actor
                    1.0 if node.get('is_director', False) else 0.0  # Is director
                ])
            else:
                node_features.extend([0.0, 0.0, 0.0])  # Padding for non-persons
            
            features.append(node_features)
        
        features_array = np.array(features, dtype=np.float32)
        print(f"Created features with shape: {features_array.shape}")
        
        return features_array

def main():
    """Main function to extract and save graph data."""
    if not NEO4J_AVAILABLE:
        print("ERROR: Neo4j driver not available. Please install: pip install neo4j")
        sys.exit(1)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available. Please install PyTorch and PyTorch Geometric.")
        sys.exit(1)
    
    print("=== Graph Data Extraction ===\n")
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Initialize extractor
    extractor = GraphDataExtractor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Extract nodes and edges
        nodes_df, node_mappings = extractor.extract_nodes()
        edges = extractor.extract_edges(node_mappings)
        
        # Create node features
        node_features = extractor.create_node_features(nodes_df)
        
        # Create edge index and edge attributes
        print("Processing edges...")
        edge_index = []
        edge_attr = []
        
        for src, dst, attr in edges:
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # Add reverse edge for undirected graph
            
            # Create edge features based on edge type
            if attr['edge_type'] == 'rated':
                edge_features = [1.0, 0.0, 0.0, 0.0, float(attr['rating']) / 5.0]
            elif attr['edge_type'] == 'belongs_to':
                edge_features = [0.0, 1.0, 0.0, 0.0, 0.0]
            elif attr['edge_type'] == 'acted_in':
                edge_features = [0.0, 0.0, 1.0, 0.0, 0.0]
            elif attr['edge_type'] == 'directed_by':
                edge_features = [0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                edge_features = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)  # Same for reverse edge
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        print(f"Final graph structure:")
        print(f"  - Nodes: {node_features.shape[0]}")
        print(f"  - Node features: {node_features.shape[1]}")
        print(f"  - Edges: {edge_index.shape[1]}")
        print(f"  - Edge features: {edge_attr.shape[1]}")
        
        # Create PyTorch Geometric data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes_df)
        )
        
        # Save graph data
        torch.save(graph_data, 'data/graph_data.pt')
        print("Saved graph data to: data/graph_data.pt")
        
        # Save metadata
        metadata = {
            'node_mappings': node_mappings,
            'nodes_df': nodes_df.to_dict('records'),
            'num_nodes': len(nodes_df),
            'num_edges': len(edges),
            'node_feature_dim': node_features.shape[1],
            'edge_feature_dim': edge_attr.shape[1]
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Saved metadata to: data/metadata.json")
        
        print("\n✅ Graph data extraction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        sys.exit(1)
    
    finally:
        extractor.close()

if __name__ == "__main__":
    main()