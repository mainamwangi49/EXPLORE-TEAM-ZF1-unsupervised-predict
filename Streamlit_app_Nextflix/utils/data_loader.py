"""

    Helper functions for data loading and manipulation.

    Author: Explore Data Science Academy.

"""
# Data handling dependencies
import pandas as pd
import numpy as np

def load_movie_titles(path_to_movies):
    """Load movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie titles.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_list = df['title'].to_list()
    return movie_list

def load_most_recent_movies(path_to_movies):
    """Load most recent movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Most recent movie titles.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    #most_recent_movie = df['url'].to_list()
    return df

def load_year_data(path_to_movies):
    """Load most recent movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Most recent movie titles.

    """
    df = pd.read_csv(path_to_movies)
    # Fill NaN values with 0 and convert the years to integers.
    df['year'] = df['year'].fillna(0).astype(int)

    # Fish out the odd years
    odd_ones = [0, 2, 3, 5, 6, 25, 101, 261]

    # Slice the merged data frame by removing the records with the odd year.
    df = df[~df['year'].isin(odd_ones)]
    return pd.DataFrame(df['year'].unique(), columns=['year']).sort_values('year', ascending=False)['year']

def load_genre_data(path_to_movies):
    """Load most recent movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Most recent movie titles.

    """
    df = pd.read_csv(path_to_movies)
    # Fill NaN values with 0 and convert the years to integers.
    df = df['genre']

    return pd.DataFrame(df.unique(), columns=['genre']).sort_values('genre', ascending=True)['genre']

def load_director_data(path_to_movies):
    """Load most recent movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Most recent movie titles.

    """
    df = pd.read_csv(path_to_movies)
    # Fill NaN values with 0 and convert the years to integers.
    df = df['director']

    return pd.DataFrame(df.unique(), columns=['director']).sort_values('director', ascending=True)

def load_merged_data(path_to_movies):
    """Load most recent movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Most recent movie titles.

    """
    df = pd.read_csv(path_to_movies)
    return df

def load_ratings_data(path_to_movies):
    """Load most recent movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Most recent movie titles.

    """
    df = pd.read_csv(path_to_movies)
    df = df.sort_values('rating', ascending=False)#.head(100)
    return df


    # def load_highest_rated_data(path_to_movies):
    #     """Load most rated movie titles from database records.

    #     Parameters
    #     ----------
    #     path_to_movies : str
    #         Relative or absolute path to movie database stored
    #         in .csv format.

    #     Returns
    #     -------
    #     list[str]
    #         Most recent movie titles.

    #     """
    #     df = pd.read_csv(path_to_movies)
    #     #df = df.sort_values('rating', ascending=False)#.head(100)
    #     return df