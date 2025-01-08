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