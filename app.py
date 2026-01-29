#!/usr/bin/env python3
"""
Flask web application for GNN Movie Recommendation Demo.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
from neo4j import GraphDatabase
import numpy as np

# Optional imports
try:
    from scripts.gnn_model import MovieRecommender
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - GNN features will be disabled")

app = Flask(__name__)
CORS(app)

# Global variables
neo4j_driver = None
recommender = None
gnn_available = False

def init_app():
    """Initialize connections and models."""
    global neo4j_driver, recommender, gnn_available

    # Initialize Neo4j connection
    try:
        neo4j_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        # Test the connection
        with neo4j_driver.session() as session:
            result = session.run("RETURN 'Neo4j connected' as message")
            record = result.single()
            print(f"Connected to Neo4j: {record['message']}")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        neo4j_driver = None

    # Initialize recommender (optional - will work without it)
    try:
        from scripts.gnn_model import MovieRecommender
        model_path = "models/gnn_gcn_2layers.pt"
        if os.path.exists(model_path):
            recommender = MovieRecommender(model_path)
            gnn_available = True
            print("Loaded GNN model")
        else:
            print("GNN model not found - using basic recommendations")
    except ImportError as e:
        print(f"GNN libraries not available: {e} - using basic recommendations")
    except Exception as e:
        print(f"Failed to load GNN model: {e} - using basic recommendations")

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/movies')
def get_movies():
    """Get list of movies."""
    if neo4j_driver is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        with neo4j_driver.session() as session:
            # First try a simple query to see if basic movie retrieval works
            result = session.run("""
                MATCH (m:Movie)
                RETURN m.id as id, m.title as title, m.year as year, m.rating as rating
                ORDER BY m.rating DESC
                LIMIT 50
            """)
            movies = []
            for record in result:
                movie_dict = dict(record)
                # Add genres separately to avoid collect issues
                genre_result = session.run("""
                    MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                    RETURN g.name as genre_name
                """, movie_id=record["id"])
                genres = [genre_record["genre_name"] for genre_record in genre_result]
                movie_dict["genres"] = genres
                movies.append(movie_dict)

        return jsonify(movies)
    except Exception as e:
        print(f"Database error in get_movies: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/movie/<int:movie_id>')
def get_movie_details(movie_id):
    """Get detailed information about a movie."""
    if neo4j_driver is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        with neo4j_driver.session() as session:
            # Get movie info
            movie_result = session.run("""
                MATCH (m:Movie {id: $movie_id})
                RETURN m.id as id, m.title as title, m.year as year,
                       m.rating as rating, m.budget as budget, m.box_office as box_office
            """, movie_id=movie_id)

            movie = movie_result.single()
            if not movie:
                return jsonify({"error": "Movie not found"}), 404

            movie_data = dict(movie)

            # Get genres
            genre_result = session.run("""
                MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                RETURN g.name as genre_name
            """, movie_id=movie_id)
            genres = [record["genre_name"] for record in genre_result]
            movie_data["genres"] = genres

            # Get actors
            actor_result = session.run("""
                MATCH (m:Movie {id: $movie_id})<-[:ACTED_IN]-(a:Person)
                RETURN a.name as actor_name
            """, movie_id=movie_id)
            actors = [record["actor_name"] for record in actor_result]
            movie_data["actors"] = actors

            # Get directors
            director_result = session.run("""
                MATCH (m:Movie {id: $movie_id})-[:DIRECTED_BY]->(d:Person)
                RETURN d.name as director_name
            """, movie_id=movie_id)
            directors = [record["director_name"] for record in director_result]
            movie_data["directors"] = directors

            # Get user ratings
            ratings_result = session.run("""
                MATCH (u:User)-[r:RATED]->(m:Movie {id: $movie_id})
                RETURN count(r) as num_ratings, avg(r.rating) as avg_rating
            """, movie_id=movie_id)

            ratings_data = ratings_result.single()
            if ratings_data:
                movie_data.update(dict(ratings_data))

        return jsonify(movie_data)
    except Exception as e:
        print(f"Database error in get_movie_details: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/recommend/<int:movie_id>')
def get_similar_movies(movie_id):
    """Get movies similar to the given movie."""
    if neo4j_driver is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        if gnn_available and recommender:
            # Use GNN-based recommendations
            try:
                similar_indices = recommender.get_similar_movies(movie_id, 'data/graph_data.pt', top_k=5)

                # Convert indices back to movie IDs
                with open('data/metadata.json', 'r') as f:
                    metadata = json.load(f)

                idx_to_movie_id = {v: k for k, v in metadata['movie_id_to_idx'].items()}
                similar_movie_ids = [idx_to_movie_id[idx] for idx in similar_indices if idx in idx_to_movie_id]

                # Get movie details
                if similar_movie_ids:
                    with neo4j_driver.session() as session:
                        result = session.run("""
                            MATCH (m:Movie)
                            WHERE m.id IN $movie_ids
                            RETURN m.id as id, m.title as title, m.year as year, m.rating as rating
                        """, movie_ids=similar_movie_ids)

                        movies = []
                        for record in result:
                            movie_dict = dict(record)
                            # Add genres separately
                            genre_result = session.run("""
                                MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                                RETURN g.name as genre_name
                            """, movie_id=record["id"])
                            genres = [genre_record["genre_name"] for genre_record in genre_result]
                            movie_dict["genres"] = genres
                            movies.append(movie_dict)
                else:
                    movies = []

                return jsonify({"method": "gnn", "movies": movies})
            except Exception as e:
                print(f"GNN recommendation failed: {e}")

        # Fallback to basic similarity (same genres)
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m1:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(m2:Movie)
                WHERE m1 <> m2
                WITH m2, count(g) as common_genres
                ORDER BY common_genres DESC, m2.rating DESC
                LIMIT 5
                RETURN m2.id as id, m2.title as title, m2.year as year, m2.rating as rating
            """, movie_id=movie_id)

            movies = []
            for record in result:
                movie_dict = dict(record)
                # Add genres separately
                genre_result = session.run("""
                    MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                    RETURN g.name as genre_name
                """, movie_id=record["id"])
                genres = [genre_record["genre_name"] for genre_record in genre_result]
                movie_dict["genres"] = genres
                movies.append(movie_dict)

        return jsonify({"method": "genre_similarity", "movies": movies})
    except Exception as e:
        print(f"Database error in get_similar_movies: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/user/<int:user_id>/recommendations')
def get_user_recommendations(user_id):
    """Get movie recommendations for a user."""
    if neo4j_driver is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        if gnn_available and recommender:
            # Use GNN-based recommendations when they return results
            try:
                recommended_ids = recommender.recommend_for_user(user_id, 'data/graph_data.pt', top_k=10)

                if recommended_ids:
                    with neo4j_driver.session() as session:
                        result = session.run("""
                            MATCH (m:Movie)
                            WHERE m.id IN $movie_ids
                            RETURN m.id as id, m.title as title, m.year as year, m.rating as rating
                        """, movie_ids=recommended_ids)

                        recommendations = []
                        for record in result:
                            movie_dict = dict(record)
                            # Add genres separately
                            genre_result = session.run("""
                                MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                                RETURN g.name as genre_name
                            """, movie_id=record["id"])
                            genres = [genre_record["genre_name"] for genre_record in genre_result]
                            movie_dict["genres"] = genres
                            recommendations.append(movie_dict)

                    if recommendations:
                        return jsonify({"method": "gnn", "recommendations": recommendations})
            except Exception as e:
                print(f"GNN recommendation failed: {e}")

        # Fallback to collaborative filtering (Neo4j) when GNN unavailable or returns empty
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r:RATED]->(m:Movie)
                WITH u, m, r.rating as user_rating
                ORDER BY user_rating DESC
                LIMIT 5

                // Find other users who rated similar movies highly
                MATCH (other:User)-[r2:RATED]->(m)
                WHERE other <> u AND r2.rating >= 4
                WITH u, other, count(*) as common_ratings
                ORDER BY common_ratings DESC
                LIMIT 10

                // Get movies that these similar users liked but our user hasn't rated
                MATCH (other)-[r3:RATED]->(rec:Movie)
                WHERE r3.rating >= 4 AND NOT exists((u)-[:RATED]->(rec))
                WITH rec, avg(r3.rating) as predicted_rating, count(*) as num_similar_users
                WHERE num_similar_users >= 2
                RETURN rec.id as id, rec.title as title, rec.year as year,
                       rec.rating as rating, predicted_rating, num_similar_users
                ORDER BY predicted_rating DESC
                LIMIT 10
            """, user_id=user_id)

            recommendations = []
            for record in result:
                movie_dict = dict(record)
                # Add genres separately
                genre_result = session.run("""
                    MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                    RETURN g.name as genre_name
                """, movie_id=record["id"])
                genres = [genre_record["genre_name"] for genre_record in genre_result]
                movie_dict["genres"] = genres
                recommendations.append(movie_dict)

        return jsonify({"method": "collaborative", "recommendations": recommendations})
    except Exception as e:
        print(f"Database error in get_user_recommendations: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/stats')
def get_stats():
    """Get database statistics."""
    if neo4j_driver is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        with neo4j_driver.session() as session:
            stats = {}

            # Basic counts
            queries = [
                ("movies", "MATCH (m:Movie) RETURN count(m) as count"),
                ("users", "MATCH (u:User) RETURN count(u) as count"),
                ("persons", "MATCH (p:Person) RETURN count(p) as count"),
                ("ratings", "MATCH ()-[r:RATED]-() RETURN count(r) as count"),
                ("genres", "MATCH (g:Genre) RETURN count(g) as count")
            ]

            for name, query in queries:
                result = session.run(query)
                record = result.single()
                stats[name] = record["count"] if record and record["count"] is not None else 0

            # Average ratings
            result = session.run("MATCH ()-[r:RATED]-() RETURN avg(r.rating) as avg_rating")
            record = result.single()
            avg_rating = record["avg_rating"] if record and record["avg_rating"] is not None else 0
            stats["avg_rating"] = round(float(avg_rating), 2) if avg_rating else 0

        return jsonify(stats)
    except Exception as e:
        print(f"Database error in get_stats: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/search')
def search_movies():
    """Search movies by title."""
    if neo4j_driver is None:
        return jsonify({"error": "Database connection not available"}), 500

    search_query = request.args.get('q', '').strip()
    if not search_query:
        return jsonify([])

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Movie)
                WHERE toLower(m.title) CONTAINS toLower($search_query)
                RETURN m.id as id, m.title as title, m.year as year, m.rating as rating
                ORDER BY m.rating DESC
                LIMIT 20
            """, search_query=search_query)

            movies = []
            for record in result:
                movie_dict = dict(record)
                # Add genres separately
                genre_result = session.run("""
                    MATCH (m:Movie {id: $movie_id})-[:BELONGS_TO]->(g:Genre)
                    RETURN g.name as genre_name
                """, movie_id=record["id"])
                genres = [genre_record["genre_name"] for genre_record in genre_result]
                movie_dict["genres"] = genres
                movies.append(movie_dict)

        return jsonify(movies)
    except Exception as e:
        print(f"Database error in search_movies: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/debug')
def debug():
    """Debug endpoint to check system status."""
    status = {
        "neo4j_connected": neo4j_driver is not None,
        "gnn_available": gnn_available,
        "torch_available": TORCH_AVAILABLE
    }

    if neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (m:Movie) RETURN count(m) as count")
                record = result.single()
                status["movie_count"] = record["count"] if record else 0
        except Exception as e:
            status["neo4j_error"] = str(e)
            status["movie_count"] = 0

    return jsonify(status)

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='127.0.0.1', port=5000)