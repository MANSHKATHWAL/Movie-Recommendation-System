
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Sample movie dataset
movies = {
    "title": [
        "Inception", "Interstellar", "The Dark Knight",
        "Avengers", "Iron Man", "Titanic",
        "Avatar", "Joker", "The Matrix",
        "Spider-Man", "Frozen", "Finding Nemo",
        "The Lion King", "Gladiator", "The Martian"
    ],
    "genre": [
        "Sci-Fi Thriller", "Sci-Fi Drama",
        "Action Crime", "Action Sci-Fi",
        "Action Sci-Fi", "Romance Drama",
        "Sci-Fi Adventure", "Crime Drama",
        "Sci-Fi Action", "Action Adventure",
        "Animation Family", "Animation Adventure",
        "Animation Drama", "Action Drama",
        "Sci-Fi Adventure"
    ],
    "description": [
        "Dreams mind technology thriller",
        "Space science time travel adventure",
        "Batman superhero crime action",
        "Superhero team action space",
        "Technology superhero action",
        "Ship love romance tragedy",
        "Alien planet adventure nature",
        "Psychological crime villain drama",
        "Artificial intelligence virtual reality",
        "Spider superhero city action",
        "Princess ice magic family",
        "Fish ocean adventure family",
        "Lion kingdom family adventure",
        "Warrior roman empire battle",
        "Astronaut mars space survival"
    ],
    "year": [
        2010, 2014, 2008, 2012, 2008,
        1997, 2009, 2019, 1999, 2002,
        2013, 2003, 1994, 2000, 2015
    ],
    "rating": [
        8.8, 8.7, 9.0, 8.0, 7.9,
        7.9, 7.9, 8.3, 8.7, 7.4,
        7.4, 8.2, 8.5, 8.5, 8.0
    ]
}

df = pd.DataFrame(movies)

# Create combined features
df["combined"] = (
    df["genre"] + " " + df["description"]
)

# Convert text into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["combined"])

# Calculate cosine similarity
similarity = cosine_similarity(tfidf_matrix)


# Recommendation function
def recommend_movies(movie_name, number=5):

    matching_movies = df[
        df["title"] == movie_name
    ]

    if matching_movies.empty:
        return pd.DataFrame()

    movie_index = matching_movies.index[0]

    scores = list(
        enumerate(similarity[movie_index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    recommended = []

    for index, score in scores[:number]:

        recommended.append({
            "Movie": df.iloc[index]["title"],
            "Genre": df.iloc[index]["genre"],
            "Year": df.iloc[index]["year"],
            "Rating": df.iloc[index]["rating"],
            "Similarity (%)": round(score * 100, 2)
        })

    return pd.DataFrame(recommended)

# Dashboard UI
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies based on your interests "
    "and favorite movie choices."
)

# Dashboard statistics
col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", len(df))
col2.metric("Movie Genres", df["genre"].nunique())
col3.metric("Average Rating", round(df["rating"].mean(), 2))

st.divider()

# Sidebar
st.sidebar.header("Movie Preferences")

genre_list = ["All"] + sorted(
    set(
        genre
        for genres in df["genre"]
        for genre in genres.split()
    )
)

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    genre_list
)

# Filter movies by genre
if selected_genre == "All":
    filtered_movies = df
else:
    filtered_movies = df[
        df["genre"].str.contains(
            selected_genre,
            case=False
        )
    ]

st.subheader("🔍 Select Your Favorite Movie")

if filtered_movies.empty:
    st.warning("No movies found.")
else:
    selected_movie = st.selectbox(
        "Choose a movie",
        filtered_movies["title"].tolist()
    )

    movie_info = df[
        df["title"] == selected_movie
    ].iloc[0]

    # Selected movie details
    st.subheader("🎥 Movie Details")

    c1, c2, c3 = st.columns(3)

    c1.metric("Release Year", movie_info["year"])
    c2.metric("Rating", movie_info["rating"])
    c3.metric("Genre", movie_info["genre"])

    st.write(
        "**Movie Description:**",
        movie_info["description"]
    )

    # Recommendation button
    if st.button("Recommend Movies 🎯"):

        st.subheader("⭐ Recommended Movies")

        result = recommend_movies(
            selected_movie,
            5
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            "Recommendations generated successfully!"
        )

st.divider()

# Display complete dataset
with st.expander("View All Movies"):
    st.dataframe(
        df.drop(columns=["combined"]),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Movie Recommendation System | Python Mini Project"
)
