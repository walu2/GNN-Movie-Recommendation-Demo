#!/usr/bin/env python3
"""
Load synthetic movie data into Neo4j database.
"""

import json
import sys
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class MovieDataLoader:
    def __init__(self, uri, user, password):
        """Initialize Neo4j driver."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Connected to Neo4j successfully!")
        except ServiceUnavailable:
            print("ERROR: Could not connect to Neo4j. Make sure Neo4j is running on bolt://localhost:7687")
            sys.exit(1)
        except AuthError:
            print("ERROR: Authentication failed. Check Neo4j username/password")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to connect to Neo4j: {e}")
            sys.exit(1)

    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()

    def clear_database(self):
        """Clear all data from the database."""
        print("Clearing existing data...")
        with self.driver.session() as session:
            # Delete all relationships first
            session.run("MATCH ()-[r]-() DELETE r")
            # Delete all nodes
            session.run("MATCH (n) DELETE n")
        print("Database cleared.")

    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        print("Creating constraints and indexes...")
        
        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT movie_id_unique IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT genre_name_unique IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            
            # Indexes for better query performance
            "CREATE INDEX movie_title_index IF NOT EXISTS FOR (m:Movie) ON (m.title)",
            "CREATE INDEX movie_year_index IF NOT EXISTS FOR (m:Movie) ON (m.year)",
            "CREATE INDEX movie_rating_index IF NOT EXISTS FOR (m:Movie) ON (m.rating)",
            "CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX user_name_index IF NOT EXISTS FOR (u:User) ON (u.name)"
        ]
        
        with self.driver.session() as session:
            for constraint_or_index in constraints_and_indexes:
                try:
                    session.run(constraint_or_index)
                except Exception as e:
                    # Constraint/index might already exist
                    print(f"Note: {e}")
        
        print("Constraints and indexes created.")

    def load_movies(self, movies):
        """Load movie nodes."""
        print(f"Loading {len(movies)} movies...")
        
        query = """
        UNWIND $movies AS movie
        CREATE (m:Movie {
            id: movie.id,
            title: movie.title,
            year: movie.year,
            rating: movie.rating,
            budget: movie.budget,
            box_office: movie.box_office
        })
        """
        
        with self.driver.session() as session:
            session.run(query, movies=movies)
        
        print(f"Loaded {len(movies)} movies.")

    def load_users(self, users):
        """Load user nodes."""
        print(f"Loading {len(users)} users...")
        
        query = """
        UNWIND $users AS user
        CREATE (u:User {
            id: user.id,
            name: user.name,
            age: user.age,
            favorite_genres: user.favorite_genres
        })
        """
        
        with self.driver.session() as session:
            session.run(query, users=users)
        
        print(f"Loaded {len(users)} users.")

    def load_persons(self, persons):
        """Load person nodes (actors and directors)."""
        print(f"Loading {len(persons)} persons...")
        
        query = """
        UNWIND $persons AS person
        CREATE (p:Person {
            id: person.id,
            name: person.name,
            birth_year: person.birth_year,
            is_actor: person.is_actor,
            is_director: person.is_director
        })
        """
        
        with self.driver.session() as session:
            session.run(query, persons=persons)
        
        print(f"Loaded {len(persons)} persons.")

    def load_genres(self, genres):
        """Load genre nodes."""
        print(f"Loading {len(genres)} genres...")
        
        query = """
        UNWIND $genres AS genre
        CREATE (g:Genre {
            name: genre.name
        })
        """
        
        with self.driver.session() as session:
            session.run(query, genres=genres)
        
        print(f"Loaded {len(genres)} genres.")

    def load_user_ratings(self, ratings):
        """Load user rating relationships."""
        print(f"Loading {len(ratings)} user ratings...")
        
        query = """
        UNWIND $ratings AS rating
        MATCH (u:User {id: rating.user_id})
        MATCH (m:Movie {id: rating.movie_id})
        CREATE (u)-[:RATED {rating: rating.rating}]->(m)
        """
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(ratings), batch_size):
                batch = ratings[i:i + batch_size]
                session.run(query, ratings=batch)
        
        print(f"Loaded {len(ratings)} user ratings.")

    def load_movie_genres(self, movie_genres):
        """Load movie-genre relationships."""
        print(f"Loading {len(movie_genres)} movie-genre relationships...")
        
        query = """
        UNWIND $movie_genres AS mg
        MATCH (m:Movie {id: mg.movie_id})
        MATCH (g:Genre {name: mg.genre})
        CREATE (m)-[:BELONGS_TO]->(g)
        """
        
        with self.driver.session() as session:
            session.run(query, movie_genres=movie_genres)
        
        print(f"Loaded {len(movie_genres)} movie-genre relationships.")

    def load_movie_cast(self, movie_cast):
        """Load movie-actor relationships."""
        print(f"Loading {len(movie_cast)} movie-cast relationships...")
        
        query = """
        UNWIND $movie_cast AS mc
        MATCH (m:Movie {id: mc.movie_id})
        MATCH (p:Person {id: mc.person_id})
        CREATE (p)-[:ACTED_IN]->(m)
        """
        
        # Process in batches
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(movie_cast), batch_size):
                batch = movie_cast[i:i + batch_size]
                session.run(query, movie_cast=batch)
        
        print(f"Loaded {len(movie_cast)} movie-cast relationships.")

    def load_movie_directors(self, movie_directors):
        """Load movie-director relationships."""
        print(f"Loading {len(movie_directors)} movie-director relationships...")
        
        query = """
        UNWIND $movie_directors AS md
        MATCH (m:Movie {id: md.movie_id})
        MATCH (p:Person {id: md.person_id})
        CREATE (m)-[:DIRECTED_BY]->(p)
        """
        
        with self.driver.session() as session:
            session.run(query, movie_directors=movie_directors)
        
        print(f"Loaded {len(movie_directors)} movie-director relationships.")

    def verify_data(self):
        """Verify that data was loaded correctly."""
        print("\nVerifying data...")
        
        queries = [
            ("Movies", "MATCH (m:Movie) RETURN count(m) as count"),
            ("Users", "MATCH (u:User) RETURN count(u) as count"),
            ("Persons", "MATCH (p:Person) RETURN count(p) as count"),
            ("Genres", "MATCH (g:Genre) RETURN count(g) as count"),
            ("User Ratings", "MATCH ()-[r:RATED]-() RETURN count(r) as count"),
            ("Movie-Genre Relations", "MATCH ()-[r:BELONGS_TO]-() RETURN count(r) as count"),
            ("Actor-Movie Relations", "MATCH ()-[r:ACTED_IN]-() RETURN count(r) as count"),
            ("Director-Movie Relations", "MATCH ()-[r:DIRECTED_BY]-() RETURN count(r) as count")
        ]
        
        with self.driver.session() as session:
            for name, query in queries:
                result = session.run(query)
                count = result.single()["count"]
                print(f"  {name}: {count}")

def load_data_file(file_path):
    """Load data from JSON file."""
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found: {file_path}")
        print("Please run 'python3 data/generate_movie_data.py' first to generate the data.")
        sys.exit(1)
    
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def main():
    """Main function to load all data."""
    print("=== Neo4j Movie Data Loader ===\n")
    
    # Load data from JSON file
    data_file = "data/movie_data.json"
    data = load_data_file(data_file)
    
    # Initialize loader
    loader = MovieDataLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Clear existing data
        loader.clear_database()
        
        # Create constraints and indexes
        loader.create_constraints_and_indexes()
        
        # Load nodes
        loader.load_movies(data["movies"])
        loader.load_users(data["users"])
        loader.load_persons(data["persons"])
        loader.load_genres(data["genres"])
        
        # Load relationships
        loader.load_user_ratings(data["relationships"]["user_ratings"])
        loader.load_movie_genres(data["relationships"]["movie_genres"])
        loader.load_movie_cast(data["relationships"]["movie_cast"])
        loader.load_movie_directors(data["relationships"]["movie_directors"])
        
        # Verify data
        loader.verify_data()
        
        print("\n✅ Data loaded successfully!")
        print(f"Neo4j Browser: http://localhost:7474")
        print(f"Database ready for GNN training and web app!")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)
    
    finally:
        loader.close()

if __name__ == "__main__":
    main()