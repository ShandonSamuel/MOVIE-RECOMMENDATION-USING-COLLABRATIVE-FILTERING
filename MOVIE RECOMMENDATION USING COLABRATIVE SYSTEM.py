# -*- coding: utf-8 -*-
"""SHANDON MOVIE RECOMMENDATION SYSTEM.ipynb


# MOVIE RECOMMENDATION SYSTEM

IMPORTING PACKAGES
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

column_names = ['MovieID', 'YearOfRelease', 'MovieName']

movie = pd.read_excel('/content/movie_info.xlsx',
                      header=None,
                      names=column_names,
                      usecols="A:C"

                       )
user = pd.read_csv('/content/user_info.txt',
                   names=['MovieID', 'Rating', 'Date'])

num_users = 50
user['CustomerID'] = ['U' + str(i % num_users + 1) for i in range(len(user))]

movie

user

"""DATA CLEANING"""

movie.drop_duplicates(inplace=True)
movie
user.drop_duplicates(inplace=True)
user

print(movie.isnull().sum());print(" ")
print(user.isnull().sum())

user['MovieID'].dropna(inplace=True)
user['Rating'].dropna(inplace=True)

user.isnull().sum()
user['Date']=pd.to_datetime(user['Date'],errors='coerce')
user

user.isnull().sum()

movie.isnull().sum()

movie['YearOfRelease'].fillna(movie['YearOfRelease'].mean(), inplace=True)
movie.isnull().sum()

user['MovieID'] = pd.to_numeric(user['MovieID'], errors='coerce')
user['Rating'] = pd.to_numeric(user['Rating'], errors='coerce')
user = user.dropna(subset=['MovieID', 'Rating'])


user['MovieID'] = user['MovieID'].astype(int)
user['Rating'] = user['Rating'].astype(int)

"""DATA VIEWING"""

user.info()

movie.info()

print(movie.head())
print(user.head())
print(movie.tail())
print(user.tail())

"""DATA SLICING"""

print(user[user['Rating']>=4])
print(user[user['Rating']<=2])

"""DATA MERGING"""

MOV_USER = pd.merge(user, movie, on="MovieID")
MOV_USER['Date'] = pd.to_datetime(MOV_USER['Date'], errors='coerce')
MOV_USER

pivot_table = MOV_USER.pivot_table(index='CustomerID',
                                    columns='MovieID',
                                    values='Rating').fillna(0)


sparse_matrix = csr_matrix(pivot_table.values)
similarity = cosine_similarity(sparse_matrix)
similarity_df = pd.DataFrame(similarity,
                             index=pivot_table.index,
                             columns=pivot_table.index)

def get_recom(user_id, top_n=15):

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:top_n+1].index

    recom_movies = (
        MOV_USER[MOV_USER['CustomerID'].isin(similar_users)]
        .groupby('MovieID')['Rating']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    recom = movie[movie['MovieID'].isin(recom_movies)][
        ['MovieID', 'MovieName', 'YearOfRelease']
    ]
    return recom.to_dict(orient='records')

output = {
    "RecommendationsForU1": get_recom("U1", top_n=15)
}

with open("RA2512052010042.json", "w") as f:
    json.dump(output, f, indent=4)

print("Collaborative Filtering completed.")


# CODE BY SHANDON SAMUEL S






