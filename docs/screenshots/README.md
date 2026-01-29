# Screenshots for GNN Movie Recommendation Demo

Place screenshots of the running app here so the main README can reference them.  
**Base URL:** `http://localhost:5000` (run `python3 app.py` first).

## Files to add

| Filename | What to capture |
|----------|------------------|
| **01-homepage.png** | Top of the page: hero section (“Graph Neural Network Movie Recommendations”), status indicators (Neo4j Connected / GNN Model Ready), and the “Tech Stack” box (PyTorch & PyG, Neo4j, Flask). |
| **02-movies-stats.png** | “Database Statistics” cards (movies, users, persons, ratings, genres, avg rating) and the “Movies” grid showing several movie cards with title, year, rating, and genre badges. |
| **03-movie-detail.png** | Movie details modal open: title, year, rating, budget, box office, genres, cast, directors, and the “Find Similar Movies” button. |
| **04-recommendations.png** | “Recommendations” section with either similarity-based results (after “Find Similar Movies”) or user-based results (after entering a User ID and clicking Recommend). |

## How to capture

1. Start Neo4j and the app: `docker-compose up -d`, then `python3 app.py`.
2. Open `http://localhost:5000` in your browser.
3. For **01-homepage.png**: Scroll to the top and capture the hero + status + tech stack.
4. For **02-movies-stats.png**: Scroll to “Database Statistics” and “Movies” and capture both in one screenshot (or two if the page is long).
5. For **03-movie-detail.png**: Click any movie card to open the modal, then capture the modal.
6. For **04-recommendations.png**: Either click “Find Similar Movies” in the modal and capture the recommendations area, or enter a User ID (e.g. `1`), click “Recommend”, and capture the recommendations section.
