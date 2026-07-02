import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recom(books_user_likes):

    # Read datasets
    books = pd.read_csv("Bookz.csv")
    img = pd.read_csv("Imagez.csv")

    # Use first 5000 books
    books = books[:5000].reset_index(drop=True)
    img = img.reset_index(drop=True)

    df = books.copy()

    # Fill missing values
    features = ['Title', 'Author', 'Publisher']
    for feature in features:
        df[feature] = df[feature].fillna('')

    # Combine features
    def combine_features(row):
        return (
            str(row['Title']) + " " +
            str(row['Author']) + " " +
            str(row['Publisher'])
        )

    df["combined_features"] = df.apply(combine_features, axis=1)

    # Create count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    # Cosine similarity
    cosine_sim = cosine_similarity(count_matrix)

    # Find book index
    result = df[df["Title"] == books_user_likes]

    if result.empty:
        return [["Book not found in dataset", "", "", ""]]

    books_index = result.index[0]

    # Similar books
    similar_books = list(enumerate(cosine_sim[books_index]))
    sorted_similar_books = sorted(
        similar_books,
        key=lambda x: x[1],
        reverse=True
    )

    final_list = []

    # Skip first result (it is the same book)
    for book_index, score in sorted_similar_books[1:11]:

        temp = []

        temp.append(df.iloc[book_index]["Title"])

        if book_index < len(img):
            temp.append(img.iloc[book_index]["Image-URL-M"])
        else:
            temp.append("")

        temp.append(df.iloc[book_index]["Year"])
        temp.append(df.iloc[book_index]["Author"])

        final_list.append(temp)

    return final_list


def bookdisp():

    books = pd.read_csv("Bookz.csv")
    img = pd.read_csv("Imagez.csv")

    books = books.reset_index(drop=True)
    img = img.reset_index(drop=True)

    finallist = []

    random_books = np.random.randint(
        0,
        min(len(books), len(img)),
        10
    )

    for i in random_books:
        temp = []

        temp.append(books.iloc[i]["Title"])
        temp.append(img.iloc[i]["Image-URL-M"])
        temp.append(books.iloc[i]["Year"])
        temp.append(books.iloc[i]["Author"])

        finallist.append(temp)

    return finallist