#!/bin/bash

# Setup script for GNN Movie Recommendation Demo

echo "Setting up GNN Movie Recommendation Demo..."

# Check if Python 3.8+ is available
python3 --version
if [ $? -ne 0 ]; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv gnn_env
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Installing virtualenv..."
    pip3 install virtualenv
    virtualenv gnn_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source gnn_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (required before PyTorch Geometric)
echo "Installing PyTorch..."
pip install "torch>=2.0.0"

# Install PyTorch Geometric (core; required for scripts)
echo "Installing PyTorch Geometric..."
pip install "torch-geometric>=2.3.0"

# Optional: PyG extension packages (scatter/sparse/cluster). Skip if they fail (e.g. on macOS without build tools).
echo "Installing PyTorch Geometric extension packages (optional)..."
pip install --no-build-isolation "torch-scatter>=2.1.0" "torch-sparse>=0.6.15" "torch-cluster>=1.6.0" 2>/dev/null || echo "  (Extensions skipped; core PyG is enough for this demo.)"

# Install remaining dependencies
echo "Installing remaining Python dependencies..."
pip install neo4j pandas numpy scikit-learn flask flask-cors matplotlib seaborn networkx

# Create necessary directories
mkdir -p models results logs

echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "source gnn_env/bin/activate"
echo ""
echo "To start the demo:"
echo "1. Start Neo4j: docker-compose up -d"
echo "2. Load data: python3 scripts/load_neo4j_data.py"
echo "3. Train GNN: python3 scripts/train_gnn.py"
echo "4. Start web app: python3 app.py"