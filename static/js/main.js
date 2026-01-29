// Main JavaScript for GNN Movie Recommendation Demo

// Global variables
let currentMovieId = null;
let allMovies = [];

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    checkSystemStatus();
    loadStats();
    loadMovies();
    
    // Add enter key support for search
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchMovies();
        }
    });
    
    // Add enter key support for user recommendations
    document.getElementById('userIdInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getUserRecommendations();
        }
    });
});

// Check system status (Neo4j and GNN availability)
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/debug');
        const status = await response.json();
        
        // Update Neo4j status
        const neo4jDot = document.getElementById('neo4j-status');
        const neo4jText = document.getElementById('neo4j-text');
        
        if (status.neo4j_connected) {
            neo4jDot.className = 'status-dot connected';
            neo4jText.textContent = `Neo4j Connected (${status.movie_count || 0} movies)`;
        } else {
            neo4jDot.className = 'status-dot disconnected';
            neo4jText.textContent = 'Neo4j Disconnected - Start Neo4j database';
        }
        
        // Update GNN status
        const gnnDot = document.getElementById('gnn-status');
        const gnnText = document.getElementById('gnn-text');
        
        if (status.gnn_available) {
            gnnDot.className = 'status-dot connected';
            gnnText.textContent = 'GNN Model Ready';
        } else if (status.torch_available) {
            gnnDot.className = 'status-dot disconnected';
            gnnText.textContent = 'PyTorch Available - Train GNN model';
        } else {
            gnnDot.className = 'status-dot disconnected';
            gnnText.textContent = 'PyTorch Not Available - Install PyTorch';
        }
        
    } catch (error) {
        console.error('Failed to check system status:', error);
        document.getElementById('neo4j-status').className = 'status-dot disconnected';
        document.getElementById('neo4j-text').textContent = 'Connection Error';
        document.getElementById('gnn-status').className = 'status-dot disconnected';
        document.getElementById('gnn-text').textContent = 'Connection Error';
    }
}

// Load database statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        if (stats.error) {
            document.getElementById('statsRow').innerHTML = `
                <div class="col-12">
                    <div class="error-message">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${stats.error}
                    </div>
                </div>
            `;
            return;
        }
        
        const statsHtml = `
            <div class="col-md-2 mb-3">
                <div class="stat-card">
                    <div class="stat-number">${stats.movies || 0}</div>
                    <div class="stat-label">🎬 Movies</div>
                </div>
            </div>
            <div class="col-md-2 mb-3">
                <div class="stat-card">
                    <div class="stat-number">${stats.users || 0}</div>
                    <div class="stat-label">👥 Users</div>
                </div>
            </div>
            <div class="col-md-2 mb-3">
                <div class="stat-card">
                    <div class="stat-number">${stats.persons || 0}</div>
                    <div class="stat-label">⭐ Persons</div>
                </div>
            </div>
            <div class="col-md-2 mb-3">
                <div class="stat-card">
                    <div class="stat-number">${stats.genres || 0}</div>
                    <div class="stat-label">🏷️ Genres</div>
                </div>
            </div>
            <div class="col-md-2 mb-3">
                <div class="stat-card">
                    <div class="stat-number">${stats.ratings || 0}</div>
                    <div class="stat-label">📊 Ratings</div>
                </div>
            </div>
            <div class="col-md-2 mb-3">
                <div class="stat-card">
                    <div class="stat-number">${stats.avg_rating || 0}</div>
                    <div class="stat-label">★ Avg Rating</div>
                </div>
            </div>
        `;
        
        document.getElementById('statsRow').innerHTML = statsHtml;
        
    } catch (error) {
        console.error('Failed to load stats:', error);
        document.getElementById('statsRow').innerHTML = `
            <div class="col-12">
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Failed to load statistics. Please check your connection.
                </div>
            </div>
        `;
    }
}

// Load all movies
async function loadMovies() {
    try {
        document.getElementById('moviesList').innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading movies...</span>
                </div>
            </div>
        `;
        
        const response = await fetch('/api/movies');
        const movies = await response.json();
        
        if (movies.error) {
            document.getElementById('moviesList').innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${movies.error}
                </div>
            `;
            return;
        }
        
        allMovies = movies;
        displayMovies(movies);
        
    } catch (error) {
        console.error('Failed to load movies:', error);
        document.getElementById('moviesList').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Failed to load movies. Please check your connection.
            </div>
        `;
    }
}

// Display movies in the UI
function displayMovies(movies) {
    if (!movies || movies.length === 0) {
        document.getElementById('moviesList').innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-film fa-3x text-muted mb-3"></i>
                <p class="text-muted">No movies found.</p>
            </div>
        `;
        return;
    }
    
    const moviesHtml = movies.map(movie => {
        const genres = movie.genres ? movie.genres.map(genre => 
            `<span class="genre-badge">${genre}</span>`
        ).join(' ') : '';
        
        return `
            <div class="col-lg-4 col-md-6 mb-4">
                <div class="card movie-card h-100" onclick="showMovieDetails(${movie.id})">
                    <div class="card-body">
                        <h5 class="movie-title">${movie.title}</h5>
                        <p class="movie-year text-muted">${movie.year || 'Unknown Year'}</p>
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="movie-rating">
                                <i class="fas fa-star"></i>
                                ${movie.rating ? movie.rating.toFixed(1) : 'N/A'}
                            </span>
                        </div>
                        <div class="genres-container">
                            ${genres}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    document.getElementById('moviesList').innerHTML = `<div class="row">${moviesHtml}</div>`;
}

// Search movies
async function searchMovies() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        displayMovies(allMovies);
        document.getElementById('recommendationsList').innerHTML = `
            <div class="text-muted text-center py-5">
                <i class="fas fa-lightbulb fa-3x mb-3"></i>
                <p>Search for a movie or enter a user ID to get personalized recommendations!</p>
            </div>
        `;
        return;
    }
    
    try {
        document.getElementById('recommendationsList').innerHTML = `
            <div class="text-center py-3">
                <div class="spinner-border text-primary" role="status"></div>
                <p class="mt-2">Searching...</p>
            </div>
        `;
        document.getElementById('recommendations').scrollIntoView({ behavior: 'smooth' });
        
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        if (!response.ok) {
            throw new Error('Search request failed');
        }
        const data = await response.json();
        const movies = Array.isArray(data) ? data : (data.movies || data.results || []);
        const errMsg = data.error;
        
        if (errMsg) {
            document.getElementById('recommendationsList').innerHTML = `
                <div class="error-message"><i class="fas fa-exclamation-triangle me-2"></i>${errMsg}</div>
            `;
            return;
        }
        
        displayMovies(movies);
        displayRecommendations(movies, `Search results for "${query}"`);
    } catch (error) {
        console.error('Search failed:', error);
        document.getElementById('recommendationsList').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle me-2"></i>Search failed. Please try again.
            </div>
        `;
    }
}

// Show movie details in modal
async function showMovieDetails(movieId) {
    currentMovieId = movieId;
    
    try {
        // Show loading in modal
        const modal = new bootstrap.Modal(document.getElementById('movieModal'));
        document.getElementById('movieModalTitle').textContent = 'Loading...';
        document.getElementById('movieModalBody').innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading movie details...</span>
                </div>
            </div>
        `;
        modal.show();
        
        // Fetch movie details
        const response = await fetch(`/api/movie/${movieId}`);
        const movie = await response.json();
        
        if (movie.error) {
            document.getElementById('movieModalBody').innerHTML = `
                <div class="error-message">
                    ${movie.error}
                </div>
            `;
            return;
        }
        
        // Update modal content
        document.getElementById('movieModalTitle').textContent = movie.title;
        
        const genres = movie.genres ? movie.genres.map(genre => 
            `<span class="genre-badge">${genre}</span>`
        ).join(' ') : 'No genres available';
        
        const actors = movie.actors ? movie.actors.map(actor => 
            `<span class="person-badge">${actor}</span>`
        ).join(' ') : 'No cast information available';
        
        const directors = movie.directors ? movie.directors.map(director => 
            `<span class="person-badge">${director}</span>`
        ).join(' ') : 'No director information available';
        
        const modalBody = `
            <div class="row">
                <div class="col-md-4">
                    <div class="movie-poster">
                        <i class="fas fa-film"></i>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="movie-detail-section">
                        <h6>Basic Information</h6>
                        <p><strong>Year:</strong> ${movie.year || 'Unknown'}</p>
                        <p><strong>Rating:</strong> ${movie.rating ? movie.rating.toFixed(1) : 'N/A'}/10</p>
                        <p><strong>Budget:</strong> $${movie.budget ? movie.budget.toLocaleString() : 'Unknown'}</p>
                        <p><strong>Box Office:</strong> $${movie.box_office ? movie.box_office.toLocaleString() : 'Unknown'}</p>
                        <p><strong>User Ratings:</strong> ${movie.num_ratings || 0} ratings (avg: ${movie.avg_rating ? movie.avg_rating.toFixed(1) : 'N/A'})</p>
                    </div>
                    
                    <div class="movie-detail-section">
                        <h6>Genres</h6>
                        <div>${genres}</div>
                    </div>
                    
                    <div class="movie-detail-section">
                        <h6>Cast</h6>
                        <div class="cast-list">${actors}</div>
                    </div>
                    
                    <div class="movie-detail-section">
                        <h6>Directors</h6>
                        <div class="director-list">${directors}</div>
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('movieModalBody').innerHTML = modalBody;
        
    } catch (error) {
        console.error('Failed to load movie details:', error);
        document.getElementById('movieModalBody').innerHTML = `
            <div class="error-message">
                Failed to load movie details. Please try again.
            </div>
        `;
    }
}

// Get similar movies using GNN
async function getSimilarMovies() {
    if (!currentMovieId) return;
    
    try {
        document.getElementById('recommendationsList').innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Finding similar movies...</span>
                </div>
                <p class="mt-2">Using AI to find similar movies...</p>
            </div>
        `;
        
        // Close modal and scroll to recommendations
        bootstrap.Modal.getInstance(document.getElementById('movieModal')).hide();
        document.getElementById('recommendations').scrollIntoView({ behavior: 'smooth' });
        
        const response = await fetch(`/api/recommend/${currentMovieId}`);
        const result = await response.json();
        
        if (result.error) {
            document.getElementById('recommendationsList').innerHTML = `
                <div class="error-message">
                    ${result.error}
                </div>
            `;
            return;
        }
        
        displayRecommendations(result.movies, `Similar movies (using ${result.method})`);
        
    } catch (error) {
        console.error('Failed to get similar movies:', error);
        document.getElementById('recommendationsList').innerHTML = `
            <div class="error-message">
                Failed to get recommendations. Please try again.
            </div>
        `;
    }
}

// Get user recommendations
async function getUserRecommendations() {
    const userId = document.getElementById('userIdInput').value.trim();
    
    if (!userId) {
        document.getElementById('recommendationsList').innerHTML = `
            <div class="error-message"><i class="fas fa-exclamation-triangle me-2"></i>Please enter a User ID (1-100).</div>
        `;
        document.getElementById('recommendations').scrollIntoView({ behavior: 'smooth' });
        return;
    }
    const num = parseInt(userId, 10);
    if (num < 1 || num > 100) {
        document.getElementById('recommendationsList').innerHTML = `
            <div class="error-message"><i class="fas fa-exclamation-triangle me-2"></i>Please enter a valid User ID (1-100).</div>
        `;
        document.getElementById('recommendations').scrollIntoView({ behavior: 'smooth' });
        return;
    }
    
    try {
        document.getElementById('recommendationsList').innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Getting recommendations...</span>
                </div>
                <p class="mt-2">Analyzing user preferences...</p>
            </div>
        `;
        document.getElementById('recommendations').scrollIntoView({ behavior: 'smooth' });
        
        const response = await fetch(`/api/user/${userId}/recommendations`);
        if (!response.ok) {
            throw new Error('Request failed');
        }
        const result = await response.json();
        
        if (result.error) {
            document.getElementById('recommendationsList').innerHTML = `
                <div class="error-message"><i class="fas fa-exclamation-triangle me-2"></i>${result.error}</div>
            `;
            return;
        }
        
        const list = Array.isArray(result) ? result : (result.recommendations || []);
        const method = result.method || 'recommendations';
        displayRecommendations(list, `Recommendations for User ${userId} (${method})`);
    } catch (error) {
        console.error('Failed to get user recommendations:', error);
        document.getElementById('recommendationsList').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle me-2"></i>Failed to get user recommendations. Please try again.
            </div>
        `;
    }
}

// Display recommendations
function displayRecommendations(movies, title) {
    if (!movies || movies.length === 0) {
        document.getElementById('recommendationsList').innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-search fa-3x text-muted mb-3"></i>
                <h5>No recommendations found</h5>
                <p class="text-muted">Try a different movie or user ID.</p>
            </div>
        `;
        return;
    }
    
    const moviesHtml = movies.map(movie => {
        const genres = movie.genres ? movie.genres.map(genre => 
            `<span class="genre-badge">${genre}</span>`
        ).join(' ') : '';
        
        return `
            <div class="col-lg-3 col-md-4 col-sm-6 mb-4">
                <div class="card movie-card h-100" onclick="showMovieDetails(${movie.id})">
                    <div class="card-body">
                        <h6 class="movie-title">${movie.title}</h6>
                        <p class="movie-year text-muted small">${movie.year || 'Unknown Year'}</p>
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="movie-rating">
                                <i class="fas fa-star"></i>
                                ${movie.rating ? movie.rating.toFixed(1) : 'N/A'}
                            </span>
                        </div>
                        <div class="genres-container">
                            ${genres}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    document.getElementById('recommendationsList').innerHTML = `
        <div class="mb-3">
            <h5><i class="fas fa-magic me-2"></i>${title}</h5>
        </div>
        <div class="row">${moviesHtml}</div>
    `;
}

// Show error message
function showError(message) {
    // You can enhance this to show toast notifications or better error handling
    alert(message);
}

// Utility function to format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}