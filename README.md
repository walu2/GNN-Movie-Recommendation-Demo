# GNN Movie Recommendation Demo

A comprehensive demonstration of Graph Neural Networks (GNNs) for movie recommendation using Neo4j as the graph database and PyTorch Geometric for the neural network implementation.

## 🎯 Overview

This project showcases how Graph Neural Networks can be applied to recommendation systems by:

- **Modeling complex relationships** between movies, users, actors, directors, and genres as a graph
- **Learning node embeddings** that capture semantic relationships in the movie domain
- **Providing personalized recommendations** based on graph structure and learned representations
- **Demonstrating real-time inference** through an interactive web interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Neo4j Graph   │    │  PyTorch GNN    │    │   Flask Web App │
│   Database      │◄──►│   Model         │◄──►│   (Frontend)    │
│                 │    │                 │    │                 │
│ • Movies        │    │ • Node Features │    │ • Movie Search  │
│ • Users         │    │ • Graph Conv    │    │ • Recommendations│
│ • Actors        │    │ • Embeddings    │    │ • Statistics    │
│ • Directors     │    │ • Predictions   │    │                 │
│ • Genres        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- 4GB+ RAM (for Neo4j and PyTorch)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd gnn_demo

# Make setup script executable
chmod +x setup.sh

# Run setup (creates virtual environment and installs dependencies)
./setup.sh
```

**If you see "PyTorch/PyG not available"** when running the scripts, install them manually inside your venv:

```bash
source gnn_env/bin/activate
pip install "torch>=2.0.0"
pip install "torch-geometric>=2.3.0"
pip install scikit-learn
```

### 2. Start Neo4j Database

```bash
# Start Neo4j with Docker Compose
docker-compose up -d

# Wait for Neo4j to be ready (check logs)
docker-compose logs neo4j
```

### 3. Load Sample Data

```bash
# Activate virtual environment
source gnn_env/bin/activate

# Generate and load movie data
python3 data/generate_movie_data.py
python3 scripts/load_neo4j_data.py
```

### 4. Train GNN Model

```bash
# Extract graph features from Neo4j
python3 scripts/extract_graph_data.py

# Train the GNN model
python3 scripts/train_gnn.py
```

### 5. Start Web Demo

```bash
# Start the Flask web application
python3 app.py
```

Visit `http://localhost:5000` to explore the demo!

## 📸 Screenshots

The app shows a single-page interface with live Neo4j and GNN status, movie search, user recommendations, database statistics, and a movie grid with details and similar-movie suggestions.

| View | Description |
|------|-------------|
| [Homepage & status](./docs/screenshots/01-homepage.png) | Hero section with Neo4j/GNN connection status and tech stack |
| [Movies & statistics](./docs/screenshots/02-movies-stats.png) | Database stats cards and movie grid with genres and ratings |
| [Movie detail & similar](./docs/screenshots/03-movie-detail.png) | Movie modal with cast, crew, budget and “Find Similar Movies” |
| [Recommendations](./docs/screenshots/04-recommendations.png) | User or similarity-based recommendation results |

To add screenshots: start the server (`python3 app.py`), open `http://localhost:5000`, then capture each view and save into `docs/screenshots/` with the filenames above. See [docs/screenshots/README.md](./docs/screenshots/README.md) for what to capture in each shot.

## 📊 Dataset

The demo uses a synthetic movie dataset with:

- **50 Movies** with realistic attributes (title, year, rating, budget, box office)
- **100 Users** with rating patterns
- **Relationships**:
  - User-Movie ratings (1-5 stars)
  - Movie-Genre classifications (Action, Drama, Comedy, etc.)
  - Movie-Actor/Director associations
- **Graph Structure**: Heterogeneous graph with multiple node and edge types

## 🧠 GNN Model Architecture

### Graph Construction

- **Nodes**: Movies, Users, Actors, Directors, Genres
- **Edges**: Ratings, Acting roles, Directing roles, Genre classifications
- **Features**: Numerical (year, rating, budget) + Categorical (genres, roles)

### Neural Network

```python
class MovieGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        # Multiple GCN/GAT/SAGE convolutional layers
        # Batch normalization and dropout
        # Final prediction layer for movie ratings
```

### Training Objective

- **Task**: Predict movie ratings based on graph structure
- **Loss**: Mean Squared Error between predicted and actual ratings
- **Optimization**: Adam optimizer with weight decay

## 🎨 Web Interface Features

### Movie Exploration
- **Search**: Find movies by title
- **Browse**: View all movies with ratings and genres
- **Details**: Comprehensive movie information (cast, crew, budget, etc.)

### GNN-Powered Recommendations
- **Similar Movies**: Graph-based movie similarity using learned embeddings
- **Personalized**: User-specific recommendations (when GNN model is available)
- **Real-time**: Instant recommendations without precomputation

### Analytics Dashboard
- **Statistics**: Database overview (movies, users, ratings, genres)
- **Visualizations**: Rating distributions and genre popularity
- **Performance**: Model accuracy metrics

## 🔷 Cypher Examples

The app uses Neo4j and Cypher for all graph queries. Below are the main patterns you can run in Neo4j Browser (`http://localhost:7474`) or in your own scripts.

### Graph schema

Nodes: `Movie`, `User`, `Person` (actors/directors), `Genre`.  
Relationships: `RATED` (User→Movie, with `rating`), `BELONGS_TO` (Movie→Genre), `ACTED_IN` (Person→Movie), `DIRECTED_BY` (Movie→Person).

### List movies with genres

```cypher
MATCH (m:Movie)
RETURN m.id AS id, m.title AS title, m.year AS year, m.rating AS rating
ORDER BY m.rating DESC
LIMIT 50
```

To attach genres per movie:

```cypher
MATCH (m:Movie)-[:BELONGS_TO]->(g:Genre)
WITH m, collect(g.name) AS genres
RETURN m.id, m.title, m.year, m.rating, genres
ORDER BY m.rating DESC
LIMIT 50
```

### Movie details (cast, crew, ratings)

```cypher
MATCH (m:Movie {id: $movie_id})
OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
OPTIONAL MATCH (u:User)-[r:RATED]->(m)
RETURN m.id, m.title, m.year, m.rating, m.budget, m.box_office,
       collect(DISTINCT g.name) AS genres,
       collect(DISTINCT a.name) AS actors,
       collect(DISTINCT d.name) AS directors,
       count(r) AS num_ratings, avg(r.rating) AS avg_rating
```

### Similar movies (by shared genres)

Used when the GNN model is not available:

```cypher
MATCH (m1:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(m2:Movie)
WHERE m1 <> m2
WITH m2, count(g) AS common_genres
ORDER BY common_genres DESC, m2.rating DESC
LIMIT 5
RETURN m2.id, m2.title, m2.year, m2.rating
```

### User-based recommendations (collaborative-style)

Find users who rated the same movies highly, then recommend movies they liked that the target user has not rated:

```cypher
MATCH (u:User {id: $user_id})-[r:RATED]->(m:Movie)
WITH u, m ORDER BY r.rating DESC LIMIT 5
MATCH (other:User)-[r2:RATED]->(m)
WHERE other <> u AND r2.rating >= 4
WITH other, count(*) AS common_ratings
ORDER BY common_ratings DESC LIMIT 10
MATCH (other)-[r3:RATED]->(rec:Movie)
WHERE r3.rating >= 4 AND NOT exists((u)-[:RATED]->(rec))
WITH rec, avg(r3.rating) AS predicted_rating, count(*) AS num_similar_users
WHERE num_similar_users >= 2
RETURN rec.id, rec.title, rec.year, rec.rating, predicted_rating
ORDER BY predicted_rating DESC
LIMIT 10
```

### Search by title

```cypher
MATCH (m:Movie)
WHERE toLower(m.title) CONTAINS toLower($search_query)
RETURN m.id, m.title, m.year, m.rating
ORDER BY m.rating DESC
LIMIT 20
```

### Database statistics

```cypher
MATCH (m:Movie) RETURN count(m) AS movies;
MATCH (u:User) RETURN count(u) AS users;
MATCH (p:Person) RETURN count(p) AS persons;
MATCH ()-[r:RATED]-() RETURN count(r) AS ratings;
MATCH (g:Genre) RETURN count(g) AS genres;
MATCH ()-[r:RATED]-() RETURN round(avg(r.rating), 2) AS avg_rating;
```

### Optional: Neo4j GDS (PageRank on movies)

If you use the Graph Data Science plugin, you can compute importance scores on the rating graph:

```cypher
CALL gds.graph.project('movie-rating-graph', ['Movie', 'User'], 'RATED');
CALL gds.pageRank.write('movie-rating-graph', { writeProperty: 'pageRank' });
```

## 🛠️ Technical Details

### Dependencies

- **Backend**: Flask, Neo4j Python Driver
- **ML**: PyTorch, PyTorch Geometric, scikit-learn
- **Database**: Neo4j 5.x with Graph Data Science plugin
- **Frontend**: Bootstrap 5, Vanilla JavaScript

### Configuration Files

- `docker-compose.yml`: Neo4j container setup
- `requirements.txt`: Python dependencies
- `neo4j/conf/neo4j.conf`: Database configuration
- `setup.sh`: Environment setup script

### Key Scripts

- `data/generate_movie_data.py`: Synthetic data generation
- `scripts/load_neo4j_data.py`: Database population
- `scripts/extract_graph_data.py`: Feature extraction for GNN
- `scripts/train_gnn.py`: Model training pipeline
- `app.py`: Web application server

## 🔧 Advanced Usage

### Customizing the Dataset

```python
# Modify data/generate_movie_data.py
NUM_MOVIES = 200  # Increase dataset size
NUM_USERS = 500   # More users for better recommendations
```

### Experimenting with GNN Architectures

```python
# In scripts/train_gnn.py, modify configs
configs = [
    {'conv_type': 'gcn', 'hidden_dim': 128, 'num_layers': 4},
    {'conv_type': 'gat', 'hidden_dim': 64, 'num_layers': 3},
    {'conv_type': 'sage', 'hidden_dim': 96, 'num_layers': 2}
]
```

### Neo4j Graph Data Science

The setup includes Neo4j GDS for advanced graph algorithms:

```cypher
// Example: PageRank for movie importance
CALL gds.pageRank.write({
  nodeProjection: 'Movie',
  relationshipProjection: 'RATED',
  writeProperty: 'pageRank'
})
```

## 📈 Performance & Evaluation

### Model Metrics

- **MSE**: Mean Squared Error on test set
- **MAE**: Mean Absolute Error on predictions
- **Training Time**: ~2-5 minutes on CPU
- **Inference**: Real-time (<100ms per recommendation)

### Scalability

- **Current Scale**: 50 movies, 100 users, ~2K edges
- **Memory Usage**: ~500MB for model and data
- **Extensible**: Architecture supports larger datasets

## 🎯 Use Cases & Applications

### Recommendation Systems
- **Content-Based**: Genre and cast-based similarity
- **Collaborative**: User rating pattern analysis
- **Hybrid**: Combining content and collaborative approaches

### Graph ML Research
- **Node Classification**: Predict movie genres
- **Link Prediction**: Recommend movies to users
- **Graph Embeddings**: Semantic movie representations

### Educational Value
- **GNN Concepts**: Practical implementation examples
- **Graph Databases**: Real-world Neo4j usage
- **Full-Stack ML**: From data to deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

### Areas for Enhancement

- [ ] Larger, real-world datasets (MovieLens, IMDb)
- [ ] Advanced GNN architectures (GraphSAGE, GATv2)
- [ ] A/B testing framework for recommendations
- [ ] User interface improvements (React/Vue.js)
- [ ] REST API for external integrations
- [ ] Model serving with FastAPI/TorchServe

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Geometric** for the excellent GNN framework
- **Neo4j** for the powerful graph database
- **MovieLens** dataset for inspiration
- **Bootstrap** for the UI framework

## 🔧 Troubleshooting

### "PyTorch/PyG not available" when running scripts

If `extract_graph_data.py` or `train_gnn.py` report that PyTorch or PyTorch Geometric is missing, install them in your active environment:

```bash
source gnn_env/bin/activate   # if not already activated
pip install "torch>=2.0.0"
pip install "torch-geometric>=2.3.0"
pip install scikit-learn
```

Then run the scripts again. The demo does not require `torch-scatter`, `torch-sparse`, or `torch-cluster`.

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Check the inline code comments
- **Neo4j Community**: Graph database questions
- **PyTorch Forums**: GNN implementation help

---

**Happy Graph Learning! 🎬🤖**