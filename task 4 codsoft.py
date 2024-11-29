# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create a sample movie dataset with more movies and additional details
data = {
    "movie_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "title": [
        "Baahubali:The Beginning", "RRR", "M.S.Dhoni-The Untold Story", "The Dark Knight", "Hanu-man", 
        "Avengers: Endgame", "Guardians of the Galaxy", "Star Wars", "Avatar", "Gravity"
    ],
    "genre": [
        "Action", "Action", "biographical", "Action", "Fantasy", 
        "Action", "Sci-Fi", "Sci-Fi", "Sci-Fi", "Sci-Fi"
    ],
    "description": [
        "Darkness befalls the Mahishmati kingdom after the assasination of beloved king Amarendra Baahubali.However,a deadly battle must be fought torestore justice.",
        "Under the British Raj, two revolutionaries with personal agendas forge a friendship, only to find each other on opposing sides.",
        "How did India get one of its favourite cricketers? Watch the life and struggles of MS Dhoni in his journey to become a respected sportsperson in this biopic.",
        "A vigilante fights crime in Gotham City.",
        "In the tranquil village of Anjanadri, a petty thief stumbles upon Hanuman-like abilities. With the impending threats, can he rise to become the hero they need?",
        "Superheroes unite to battle a cosmic threat.",
        "A group of misfits protect the galaxy from evil forces.",
        "An intergalactic saga about the fight between good and evil.",
        "A marine on an alien planet fights to save a native species.",
        "Astronauts struggle to survive in outer space."
    ],
    "rating": [8.0, 7.8, 8.0, 9.0, 7.8, 8.4, 8.1, 8.6, 7.9, 7.7],
    "release_year": [2015, 2022, 2016, 2008, 2024, 2019, 2014, 1977, 2009, 2013]
}

# Step 2: Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Step 3: Combine important features into a single string
df["combined_features"] = df["genre"] + " " + df["description"]

# Step 4: Transform text into numeric features using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")  # Remove common words (stop words)
tfidf_matrix = tfidf.fit_transform(df["combined_features"])  # Generate TF-IDF matrix

# Step 5: Compute pairwise cosine similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 6: Define a recommendation function
def recommend_content_based(movie_title, df, similarity_matrix, num_recommendations=3):
    """
    Recommend movies based on content similarity and display movie details.

    Args:
    - movie_title (str): Title of the movie for recommendations.
    - df (DataFrame): DataFrame containing movie details.
    - similarity_matrix (array): Precomputed cosine similarity matrix.
    - num_recommendations (int): Number of recommendations to return.

    Returns:
    - Details of the input movie and a list of recommended movie titles with details.
    """
    try:
        # Find the index of the movie in the DataFrame
        movie_index = df[df["title"].str.lower() == movie_title.lower()].index[0]

        # Extract details of the selected movie
        selected_movie_details = df.iloc[movie_index]

        # Get similarity scores for the target movie
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))

        # Sort movies by similarity score (excluding the input movie itself)
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

        # Extract details of recommended movies
        recommended_movies = [df.iloc[i[0]][["title", "rating", "release_year"]] for i in sorted_scores]

        return selected_movie_details, recommended_movies
    except IndexError:
        return None, ["Movie not found! Please check the title."]

# Step 7: Interactive user input
print("\n🎥 Welcome to the Movie Recommendation System!")
movie_to_search = input("Enter the name of a movie you like: ")

# Fetch recommendations
selected_movie, recommendations = recommend_content_based(movie_to_search, df, similarity_matrix)

# Step 8: Display the results
if selected_movie is not None:
    print(f"\n🎬 *Selected Movie*: {selected_movie['title']}")
    print(f"   - Genre: {selected_movie['genre']}")
    print(f"   - Rating: {selected_movie['rating']}")
    print(f"   - Release Year: {selected_movie['release_year']}")
    print(f"   - Description: {selected_movie['description']}\n")

    print(f"✨ *Top {len(recommendations)} Recommendations*:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['title']} (Rating: {rec['rating']}, Year: {rec['release_year']})")
else:
    print("❌ Movie not found! Please check your spelling or try another movie.")
