import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
import json

raw_path = 'data/raw/' # where the data is
clean_path = 'data/clean/' # to save the preprocessed dataset
model_path = 'data/model/' #to save the model dataset to training step

credits = pd.read_csv(raw_path + 'tmdb_5000_credits.csv')
movies = pd.read_csv(raw_path + 'tmdb_5000_movies.csv')

# Preprocess data
credits.columns = ['id', 'title', 'cast', 'crew']
movies_info = movies.merge(credits, on = "id")
movies_info["overview"] = movies_info["overview"].fillna("")

movies_info.to_csv(clean_path + "tmb_5000_complete.csv", sep = ";", header = True)

# Strings cannot be fed directly into any machine learning algorithm 
# so we will first compute Term Frequency- Inverse Document Frequency

#### THEORY ########################################################################
# Explain of (TI)-IDF: it is the relative frequency of any word in a document 
# and is given by dividing term instances with total instances.

# The other part of TF- (IDF) called Inverse Document Frequency is the relative 
# count of documents containing the term and is given as a log 
# (number of documents/documents with the term).

# the overall importance of each word in the document in which they appear would 
# be given by  TF * IDF

## FINAL RESULT: a matrix where each column represents a word in the overall 
# vocabulary (all the words that appear in at least one document) and each row 
# represents a movie, as before. TF-IDF is useful in reducing the importance of 
# words that occur frequently in our soup of movie description, genre, and 
# keyword and would, in turn, reduce their significance in computing the 
# final similarity score.
####################################################################################

# TF-IDF Vectorizer

# We have 'overview' column with a short description of the movie, the "genres" column
# and 'key_words' columns with some key information. 
# So we are going to get the column text to TF-IDF by the union of this two columns. 

def get_column_soup(df, column_name, key_name): 
    list_keywords = df[column_name].to_list()
    list_dict_keyw = [json.loads(lk_str) if lk_str else [] for lk_str in list_keywords]
    df[column_name + "_soup"] = [[kk[key_name] for kk in row] for row in list_dict_keyw]

# you can apply this functions to some other columns to make a better overview.

# Get words of 'keywords' soup column: 
get_column_soup(movies_info, "keywords", "name")

# Get words of genres column: 
get_column_soup(movies_info, "genres", "name")

def create_soup(row): 
    # Make the final soup column:
    return ' '.join(row["keywords_soup"]) + ' ' + ' '.join(row["genres_soup"]) + ' ' + ''.join(row["overview"])

movies_info["soup"] = movies_info.apply(create_soup, axis = 1)

tfidf = TfidfVectorizer(stop_words = "english")
tfidf_matrix = tfidf.fit_transform(movies_info["soup"])

# tfidf_matrix.shape == (4803, 23004). 
# The 4803 movies of our dataset are described by over 23004 words. 

# With this tf-idf matrix, we will now compute a similarity score. 
# There are several ways to compute similarity such as- using Euclidean distance  or 
# using the Pearson and the cosine similarity scores. 
# It is good to experiment with them as it cannot be said beforehand which 
# would be best- anyone of these can work based on the scenario.

# Calculate the cosine similarity scores by the dot product of normalized vectors, because 
# we are using the TD-IDF Vectorizer, we can calculate it by using the dot product directly. 

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
# Save the reverse map of indexs and movie title for searching the movies in our system recommender solution.
movie_indexs = pd.Series(movies_info.index, index = movies["title"]).drop_duplicates() 

#### MOVIE RECOMMENDER SYSTEM ############################################################

# It can be created by following the steps: 
# 1. Identify the index of the movie in the dataset to work with it. 
# 2. Calculate the cosine similarity of the objective movie with all movies in the dataset.
#   2.1. Generate a list of tuples where the first element is the movie index and its similarity score. 
# 3. Sort the tuple list to get the most similar movies. 
# 4. Get top 10 similar movies to show, but delete the first element because it refers to the target movie itself.
# 5. Return the titles that correspond to the indexs of the top elements.

def movie_recommender(target_movie):
    idx = movie_indexs[target_movie]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    # With enumerate we obtain the index of the movie to compare and the score of similarity with the target_movie. 
    sim_scores = sorted(sim_scores, 
        key = lambda x: x[1], # sort by the 2nd element (python start on 0 position.)
        reverse = True
    )

    top10 = sim_scores[1:11] # ignore the first element (0)
    movies_idx_recommended = [i[0] for i in top10] # index of movies that will be present by the recommender
    return movies["title"].iloc[movies_idx_recommended]

# Test the recommender system: 
movie_recommender("The Avengers")
# 7                  Avengers: Age of Ultron
# 511                                  X-Men
# 242                         Fantastic Four
# 26              Captain America: Civil War
# 64                       X-Men: Apocalypse
# 79                              Iron Man 2
# 85     Captain America: The Winter Soldier
# 169     Captain America: The First Avenger
# 182                                Ant-Man
# 68                                Iron Man
# Name: title, dtype: object