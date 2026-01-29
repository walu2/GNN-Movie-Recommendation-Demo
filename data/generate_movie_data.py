#!/usr/bin/env python3
"""
Generate synthetic movie data for the GNN demo.
Creates movies, users, persons (actors/directors), genres, and relationships.
"""

import json
import random
from datetime import datetime, timedelta
import pandas as pd

# Configuration
NUM_MOVIES = 50
NUM_USERS = 100
NUM_PERSONS = 80  # actors and directors
GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", 
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", 
    "Music", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"
]

# Sample data
MOVIE_TITLES = [
    "The Last Stand", "Midnight Runner", "Ocean's Edge", "Silent Thunder", 
    "Crystal Dreams", "Neon Nights", "Shadow Hunter", "Golden Dawn",
    "Steel Rain", "Crimson Tide", "Electric Storm", "Frozen Hearts",
    "Desert Storm", "Urban Legend", "Wild Fire", "Dark Moon",
    "Silver Bullet", "Iron Will", "Brave Heart", "Quick Silver",
    "Storm Chaser", "Night Crawler", "Star Fighter", "Wind Walker",
    "Ghost Rider", "Fire Storm", "Ice Cold", "Thunder Road",
    "Lightning Strike", "Solar Flare", "Cosmic Ray", "Quantum Leap",
    "Time Warp", "Space Odyssey", "Galaxy Quest", "Star Trek",
    "Dark Knight", "Green Arrow", "Red Phoenix", "Blue Thunder",
    "White Lightning", "Black Widow", "Silver Fox", "Golden Eagle",
    "Iron Eagle", "Steel Magnolia", "Diamond Dogs", "Ruby Red",
    "Emerald City", "Sapphire Blue", "Crystal Clear", "Midnight Sun"
]

PERSON_NAMES = [
    "Alex Johnson", "Sarah Williams", "Michael Brown", "Emma Davis",
    "James Wilson", "Olivia Miller", "William Moore", "Ava Taylor",
    "Benjamin Anderson", "Sophia Thomas", "Lucas Jackson", "Isabella White",
    "Henry Harris", "Mia Martin", "Alexander Thompson", "Charlotte Garcia",
    "Sebastian Martinez", "Amelia Robinson", "Jack Clark", "Harper Rodriguez",
    "Owen Lewis", "Evelyn Lee", "Daniel Walker", "Abigail Hall",
    "Matthew Allen", "Emily Young", "Jacob Hernandez", "Madison King",
    "Michael Wright", "Elizabeth Lopez", "Ethan Hill", "Sofia Scott",
    "Noah Green", "Avery Adams", "Logan Baker", "Ella Gonzalez",
    "Samuel Nelson", "Scarlett Carter", "David Mitchell", "Grace Perez",
    "Joseph Roberts", "Chloe Turner", "John Phillips", "Victoria Campbell",
    "Andrew Parker", "Penelope Evans", "Ryan Edwards", "Riley Collins",
    "Nathan Stewart", "Layla Sanchez", "Caleb Morris", "Zoe Rogers",
    "Luke Reed", "Nora Cook", "Christian Morgan", "Lillian Bailey",
    "Hunter Rivera", "Brooklyn Cooper", "Isaiah Richardson", "Leah Cox",
    "Thomas Howard", "Audrey Ward", "Aaron Torres", "Maya Peterson",
    "Connor Gray", "Claire Ramirez", "Jeremiah James", "Skylar Watson",
    "Cameron Brooks", "Paisley Kelly", "Adrian Sanders", "Anna Price",
    "Wyatt Bennett", "Caroline Wood", "Angel Barnes", "Genesis Ross",
    "Miles Henderson", "Aaliyah Coleman", "Jason Jenkins", "Kinsley Perry",
    "Ian Powell", "Naomi Long", "Cooper Patterson", "Melanie Hughes",
    "Jaxon Flores", "Gabriella Washington", "Parker Butler", "Samantha Simmons"
]

def generate_movies():
    """Generate movie data."""
    movies = []
    used_titles = set()
    
    for movie_id in range(1, NUM_MOVIES + 1):
        # Ensure unique titles
        title = random.choice(MOVIE_TITLES)
        while title in used_titles:
            title = random.choice(MOVIE_TITLES)
        used_titles.add(title)
        
        # Generate movie attributes
        year = random.randint(1990, 2023)
        rating = round(random.uniform(3.5, 9.5), 1)  # IMDb-like rating
        budget = random.randint(5_000_000, 200_000_000)  # Budget in dollars
        box_office = budget * random.uniform(0.5, 5.0)  # Box office relative to budget
        
        movie = {
            "id": movie_id,
            "title": title,
            "year": year,
            "rating": rating,
            "budget": int(budget),
            "box_office": int(box_office)
        }
        
        movies.append(movie)
    
    return movies

def generate_users():
    """Generate user data."""
    users = []
    
    for user_id in range(1, NUM_USERS + 1):
        # Generate user preferences
        age = random.randint(18, 70)
        favorite_genres = random.sample(GENRES, k=random.randint(2, 5))
        
        user = {
            "id": user_id,
            "name": f"User_{user_id}",
            "age": age,
            "favorite_genres": favorite_genres
        }
        
        users.append(user)
    
    return users

def generate_persons():
    """Generate person data (actors and directors)."""
    persons = []
    used_names = set()
    
    for person_id in range(1, NUM_PERSONS + 1):
        # Ensure unique names
        name = random.choice(PERSON_NAMES)
        while name in used_names:
            name = random.choice(PERSON_NAMES)
        used_names.add(name)
        
        # Generate person attributes
        birth_year = random.randint(1950, 2000)
        is_actor = random.random() > 0.3  # 70% chance of being actor
        is_director = random.random() > 0.7  # 30% chance of being director
        
        # Ensure each person is at least one of actor or director
        if not is_actor and not is_director:
            is_actor = True
        
        person = {
            "id": person_id,
            "name": name,
            "birth_year": birth_year,
            "is_actor": is_actor,
            "is_director": is_director
        }
        
        persons.append(person)
    
    return persons

def generate_relationships(movies, users, persons):
    """Generate relationships between entities."""
    relationships = {
        "user_ratings": [],
        "movie_genres": [],
        "movie_cast": [],
        "movie_directors": []
    }
    
    # User ratings
    for user in users:
        # Each user rates 15-25 movies
        num_ratings = random.randint(15, 25)
        rated_movies = random.sample(movies, num_ratings)
        
        for movie in rated_movies:
            # Generate rating based on user preferences and movie attributes
            base_rating = movie["rating"] / 2  # Scale down from 10 to 5
            
            # Adjust rating based on user preferences
            movie_genres = [rel["genre"] for rel in relationships["movie_genres"] 
                           if rel["movie_id"] == movie["id"]]
            
            # If movie has user's favorite genres, increase rating
            genre_bonus = 0
            for genre in movie_genres:
                if genre in user["favorite_genres"]:
                    genre_bonus += 0.3
            
            # Add some randomness
            final_rating = base_rating + genre_bonus + random.uniform(-0.5, 0.5)
            final_rating = max(1, min(5, round(final_rating, 1)))  # Clamp to 1-5
            
            relationships["user_ratings"].append({
                "user_id": user["id"],
                "movie_id": movie["id"],
                "rating": final_rating
            })
    
    # Movie genres
    for movie in movies:
        # Each movie has 1-3 genres
        num_genres = random.randint(1, 3)
        movie_genres = random.sample(GENRES, num_genres)
        
        for genre in movie_genres:
            relationships["movie_genres"].append({
                "movie_id": movie["id"],
                "genre": genre
            })
    
    # Movie cast (actors)
    actors = [p for p in persons if p["is_actor"]]
    for movie in movies:
        # Each movie has 3-8 actors
        num_actors = random.randint(3, 8)
        movie_actors = random.sample(actors, min(num_actors, len(actors)))
        
        for actor in movie_actors:
            relationships["movie_cast"].append({
                "movie_id": movie["id"],
                "person_id": actor["id"],
                "person_name": actor["name"]
            })
    
    # Movie directors
    directors = [p for p in persons if p["is_director"]]
    for movie in movies:
        # Each movie has 1-2 directors
        num_directors = random.randint(1, 2)
        movie_directors = random.sample(directors, min(num_directors, len(directors)))
        
        for director in movie_directors:
            relationships["movie_directors"].append({
                "movie_id": movie["id"],
                "person_id": director["id"],
                "person_name": director["name"]
            })
    
    return relationships

def main():
    """Generate all synthetic data."""
    print("Generating synthetic movie data...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate entities
    print(f"Generating {NUM_MOVIES} movies...")
    movies = generate_movies()
    
    print(f"Generating {NUM_USERS} users...")
    users = generate_users()
    
    print(f"Generating {NUM_PERSONS} persons (actors/directors)...")
    persons = generate_persons()
    
    print("Generating relationships...")
    relationships = generate_relationships(movies, users, persons)
    
    # Create data structure
    data = {
        "movies": movies,
        "users": users,
        "persons": persons,
        "genres": [{"name": genre} for genre in GENRES],
        "relationships": relationships,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_movies": NUM_MOVIES,
            "num_users": NUM_USERS,
            "num_persons": NUM_PERSONS,
            "num_genres": len(GENRES),
            "num_ratings": len(relationships["user_ratings"])
        }
    }
    
    # Save to JSON file
    output_file = "data/movie_data.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nData generated successfully!")
    print(f"Saved to: {output_file}")
    print("\nSummary:")
    print(f"- Movies: {len(movies)}")
    print(f"- Users: {len(users)}")
    print(f"- Persons: {len(persons)}")
    print(f"- Genres: {len(GENRES)}")
    print(f"- User Ratings: {len(relationships['user_ratings'])}")
    print(f"- Movie-Genre Relations: {len(relationships['movie_genres'])}")
    print(f"- Movie-Cast Relations: {len(relationships['movie_cast'])}")
    print(f"- Movie-Director Relations: {len(relationships['movie_directors'])}")

if __name__ == "__main__":
    main()