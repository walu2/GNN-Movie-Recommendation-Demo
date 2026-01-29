// GNN Movie Recommender JavaScript

let currentMovieId = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadMovies();

    // Search functionality
    const searchInput = document.getElementById('searchInput');
    let searchTimeout;

    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            const query = this.value.trim();
            if (query.length > 0) {
                searchMovies(query);
            } else {
                loadMovies();
            }
        }, 300);
    });

    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchMovies(this.value.trim());
        }
    });
});

// Load database statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const stats = await response.json();

        document.getElementById('moviesCount').textContent = stats.movies || 0;
        document.getElementById('usersCount').textContent = stats.users || 0;
        document.getElementById('personsCount').textContent = stats.persons || 0;
        document.getElementById('ratingsCount').textContent = stats.ratings || 0;
        document.getElementById('genresCount').textContent = stats.genres || 0;
        document.getElementById('avgRating').textContent = stats.avg_rating || '0.0';
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load movies list
async function loadMovies() {
    const moviesList = document.getElementById('moviesList');
    moviesList.innerHTML = '<div class="col-12 text-center"><div class="spinner-border" role="status"></div></div>';

    try {
        const response = await fetch('/api/movies');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const movies = await response.json();

        displayMovies(movies);
    } catch (error) {
        console.error('Error loading movies:', error);
        moviesList.innerHTML = '<div class="col-12"><div class="alert alert-danger">Error loading movies</div></div>';
    }
}

// Search movies
async function searchMovies(query) {
    if (!query) return;

    const moviesList = document.getElementById('moviesList');
    moviesList.innerHTML = '<div class="col-12 text-center"><div class="spinner-border" role="status"></div></div>';

    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const movies = await response.json();

        displayMovies(movies);
    } catch (error) {
        console.error('Error searching movies:', error);
        moviesList.innerHTML = '<div class="col-12"><div class="alert alert-danger">Error searching movies</div></div>';
    }
}

// Display movies in grid
function displayMovies(movies) {
    const moviesList = document.getElementById('moviesList');

    if (movies.length === 0) {
        moviesList.innerHTML = '<div class="col-12"><div class="alert alert-info">No movies found</div></div>';
        return;
    }

    moviesList.innerHTML = movies.map(movie => `
        <div class="col-md-6 col-lg-4 mb-3">
            <div class="card movie-card h-100" onclick="selectMovie(${movie.id})">
                <div class="card-body">
                    <h6 class="movie-title">${movie.title}</h6>
                    <div class="movie-meta">
                        <div><i class="fas fa-calendar"></i> ${movie.year}</div>
                        <div class="rating-stars">
                            ${generateStars(movie.rating)}
                            <span class="ms-1">${movie.rating}</span>
                        </div>
                    </div>
                    <div class="genres mt-2">
                        ${movie.genres ? movie.genres.slice(0, 3).map(genre =>
                            `<span class="badge bg-secondary genre-badge">${genre}</span>`
                        ).join('') : ''}
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Generate star rating display
function generateStars(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

    return (
        '<i class="fas fa-star"></i>'.repeat(fullStars) +
        (hasHalfStar ? '<i class="fas fa-star-half-alt"></i>' : '') +
        '<i class="far fa-star"></i>'.repeat(emptyStars)
    );
}

// Select a movie and show details
async function selectMovie(movieId) {
    currentMovieId = movieId;

    // Update UI to show selected movie
    document.querySelectorAll('.movie-card').forEach(card => {
        card.classList.remove('border-primary', 'bg-light');
    });

    event.currentTarget.classList.add('border-primary', 'bg-light');

    // Load movie details
    await loadMovieDetails(movieId);

    // Load recommendations
    await loadRecommendations(movieId);
}

// Load movie details
async function loadMovieDetails(movieId) {
    const detailsCard = document.getElementById('movieDetailsCard');
    const detailsDiv = document.getElementById('movieDetails');

    detailsDiv.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm" role="status"></div></div>';
    detailsCard.style.display = 'block';

    try {
        const response = await fetch(`/api/movie/${movieId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const movie = await response.json();

        detailsDiv.innerHTML = `
            <h5>${movie.title}</h5>
            <p class="text-muted mb-2">${movie.year} • ${generateStars(movie.rating)} ${movie.rating}</p>

            <div class="row mb-3">
                <div class="col-6">
                    <strong>Budget:</strong><br>
                    $${movie.budget ? movie.budget.toLocaleString() : 'N/A'}
                </div>
                <div class="col-6">
                    <strong>Box Office:</strong><br>
                    $${movie.box_office ? movie.box_office.toLocaleString() : 'N/A'}
                </div>
            </div>

            ${movie.genres && movie.genres.length > 0 ? `
                <div class="mb-3">
                    <strong>Genres:</strong><br>
                    ${movie.genres.map(genre => `<span class="badge bg-primary me-1">${genre}</span>`).join('')}
                </div>
            ` : ''}

            ${movie.directors && movie.directors.length > 0 ? `
                <div class="mb-3">
                    <strong>Director${movie.directors.length > 1 ? 's' : ''}:</strong><br>
                    ${movie.directors.join(', ')}
                </div>
            ` : ''}

            ${movie.actors && movie.actors.length > 0 ? `
                <div class="mb-3">
                    <strong>Cast:</strong><br>
                    ${movie.actors.slice(0, 5).join(', ')}${movie.actors.length > 5 ? '...' : ''}
                </div>
            ` : ''}

            ${movie.num_ratings ? `
                <div class="mb-3">
                    <strong>User Ratings:</strong><br>
                    ${movie.num_ratings} ratings, average: ${movie.avg_rating ? movie.avg_rating.toFixed(1) : 'N/A'}
                </div>
            ` : ''}
        `;
    } catch (error) {
        console.error('Error loading movie details:', error);
        detailsDiv.innerHTML = '<div class="alert alert-danger">Error loading movie details</div>';
    }
}

// Load movie recommendations
async function loadRecommendations(movieId) {
    const recCard = document.getElementById('recommendationsCard');
    const recDiv = document.getElementById('recommendations');

    recDiv.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm" role="status"></div></div>';
    recCard.style.display = 'block';

    try {
        const response = await fetch(`/api/recommend/${movieId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();

        // Handle both GNN and genre-based recommendations
        const recommendations = data.movies || data.recommendations || [];

        if (recommendations.length === 0) {
            recDiv.innerHTML = '<p class="text-muted">No recommendations available</p>';
            return;
        }

        recDiv.innerHTML = recommendations.map(movie => `
            <div class="recommendation-item" onclick="selectMovie(${movie.id})">
                <div class="recommendation-title">${movie.title}</div>
                <div class="recommendation-meta">
                    ${movie.year} • ${generateStars(movie.rating)} ${movie.rating}
                    ${movie.genres && movie.genres.length > 0 ? ` • ${movie.genres.slice(0, 2).join(', ')}` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading recommendations:', error);
        recDiv.innerHTML = '<div class="alert alert-warning">GNN recommendations not available</div>';
    }
}

// Utility functions
function showLoading() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoading() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) modal.hide();
}

// Error handling
function showError(message) {
    // You could implement a toast notification here
    console.error(message);
    alert(message);
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});