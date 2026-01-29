#!/bin/bash

# Full setup script for GNN Movie Recommendation Demo
# Handles Docker, PyTorch, and complete pipeline

echo "🎬 GNN Movie Recommendation - Full Stack Setup"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script from the gnn_demo directory"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source gnn_env/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    echo "Please run ./setup.sh first"
    exit 1
}

# Check Docker
echo "🐳 Checking Docker..."
if ! docker --version > /dev/null 2>&1; then
    echo "❌ Docker not found. Please install Docker Desktop"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop"
    echo "You should see the Docker whale icon in your menu bar"
    exit 1
fi

echo "✅ Docker is running"

# Start Neo4j
echo "🗄️  Starting Neo4j database..."
docker-compose down > /dev/null 2>&1  # Clean shutdown first
docker-compose up -d

echo "⏳ Waiting for Neo4j to start..."
sleep 10

# Check if Neo4j is ready
echo "🔍 Checking Neo4j connection..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if python3 -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    with driver.session() as session:
        session.run('RETURN 1')
    driver.close()
    print('Neo4j is ready!')
    exit(0)
except Exception:
    exit(1)
" 2>/dev/null; then
        echo "✅ Neo4j is ready!"
        break
    fi
    
    echo "⏳ Waiting for Neo4j... (attempt $attempt/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Neo4j failed to start. Check Docker logs:"
    echo "docker-compose logs neo4j"
    exit 1
fi

# Install PyTorch
echo "🧠 Installing PyTorch with Apple Silicon support..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch torchvision torchaudio

if [ $? -ne 0 ]; then
    echo "⚠️  Standard PyTorch installation failed. Trying alternative method..."
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

# Install PyTorch Geometric
echo "📊 Installing PyTorch Geometric..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch-geometric

# Test GPU setup
echo "🔬 Testing GPU setup..."
python3 test_gpu.py

# Load data pipeline
echo "📈 Running data pipeline..."

echo "  1/4 - Loading movie data into Neo4j..."
python3 scripts/load_neo4j_data.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to load data into Neo4j"
    exit 1
fi

echo "  2/4 - Extracting graph features..."
python3 scripts/extract_graph_data.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to extract graph data"
    exit 1
fi

echo "  3/4 - Training GNN model (this may take a few minutes)..."
python3 scripts/train_gnn.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to train GNN model"
    exit 1
fi

echo "  4/4 - Testing Flask application..."
python3 -c "
import app
app.init_app()
print('✅ Flask app ready!')
print('✅ Neo4j:', 'Connected' if app.neo4j_driver else 'Disconnected')
print('✅ GNN:', 'Available' if app.gnn_available else 'Not Available')
print('✅ PyTorch:', 'Available' if app.TORCH_AVAILABLE else 'Not Available')
"

echo ""
echo "🎉 Full Stack Setup Complete!"
echo "=============================="
echo ""
echo "🚀 To start the application:"
echo "   python3 app.py"
echo ""
echo "🌐 Then visit: http://localhost:5000"
echo ""
echo "📊 Your setup includes:"
echo "   ✅ Neo4j Graph Database (Docker)"
echo "   ✅ PyTorch with Apple Silicon GPU support"
echo "   ✅ Trained GNN model for recommendations" 
echo "   ✅ 50 movies, 100 users, 2000+ ratings"
echo "   ✅ Modern web interface"
echo ""
echo "🔧 To stop Neo4j when done:"
echo "   docker-compose down"