#!/bin/bash
# End-to-end test for GNN Movie Recommendation Demo
# Run from project root: ./scripts/e2e_test.sh

set -e
cd "$(dirname "$0")/.."
echo "=== E2E Test (cwd: $(pwd)) ==="

# Activate venv
if [ ! -d "gnn_env" ]; then
  echo "ERROR: gnn_env not found. Run ./setup.sh first."
  exit 1
fi
source gnn_env/bin/activate

# 1) Generate movie data
echo ""
echo "1/6 Generating movie data..."
python3 data/generate_movie_data.py
[ -f data/movie_data.json ] || { echo "ERROR: data/movie_data.json not created"; exit 1; }
echo "   OK: data/movie_data.json"

# 2) Start Neo4j
echo ""
echo "2/6 Starting Neo4j..."
docker-compose down 2>/dev/null || true
docker-compose up -d
echo "   Waiting for Neo4j..."
sleep 8
max=30
n=0
until python3 -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
with d.session() as s: s.run('RETURN 1')
d.close()
" 2>/dev/null; do
  n=$((n+1))
  [ $n -ge $max ] && { echo "   ERROR: Neo4j did not become ready"; docker-compose logs --tail 20 neo4j; exit 1; }
  sleep 2
  echo "   ... attempt $n/$max"
done
echo "   OK: Neo4j ready"

# 3) Load data into Neo4j
echo ""
echo "3/6 Loading data into Neo4j..."
python3 scripts/load_neo4j_data.py
echo "   OK: Data loaded"

# 4) Extract graph data (requires PyTorch + PyG)
echo ""
echo "4/6 Extracting graph data..."
if python3 -c "import torch_geometric" 2>/dev/null; then
  python3 scripts/extract_graph_data.py
  [ -f data/graph_data.pt ] || { echo "   ERROR: data/graph_data.pt not created"; exit 1; }
  echo "   OK: graph_data.pt and metadata.json"
else
  echo "   SKIP: torch_geometric not installed"
fi

# 5) Train GNN (optional)
echo ""
echo "5/6 Training GNN..."
if python3 -c "import torch_geometric; import torch_scatter" 2>/dev/null && [ -f data/graph_data.pt ]; then
  python3 scripts/train_gnn.py
  echo "   OK: Model trained"
else
  echo "   SKIP: PyG/torch_scatter or graph data not available"
fi

# 6) Start Flask app and verify
echo ""
echo "6/6 Starting Flask app and verifying..."
python3 -c "
import app
app.init_app()
print('   Neo4j:', 'Connected' if app.neo4j_driver else 'Disconnected')
print('   GNN:  ', 'Available' if app.gnn_available else 'Not available (OK for basic recommendations)')
" 
# Start server in background and hit it
python3 app.py &
APP_PID=$!
trap "kill $APP_PID 2>/dev/null || true" EXIT
sleep 3
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/ || echo "000")
if [ "$STATUS" = "200" ]; then
  echo "   GET / => $STATUS OK"
else
  echo "   GET / => $STATUS (expected 200)"
  exit 1
fi
MOVIES_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/api/movies || echo "000")
if [ "$MOVIES_STATUS" = "200" ]; then
  echo "   GET /api/movies => $MOVIES_STATUS OK"
else
  echo "   GET /api/movies => $MOVIES_STATUS (expected 200)"
  exit 1
fi

echo ""
echo "=== E2E test passed ==="
echo "Neo4j is still running. Stop with: docker-compose down"
