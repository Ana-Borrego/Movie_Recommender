import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

credits = pd.read_csv('data/raw/tmdb_5000_credits.csv')
movies = pd.read_csv('data/raw/tmdb_5000_movies.csv')

#
credits.columns = ['id', 'title', 'cast', 'crew']
movies_info = movies.merge(credits, on = "id")
movies_info["overview"] = movies_info["overview"].fillna("")

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

