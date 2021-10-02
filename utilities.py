import pandas as pd
import numpy as np

# calculate weigted ratings for each book
def weighted_rating(x,m,C):
    v = x['work_ratings_count']
    R = x['average_rating']
    return (v / (v + m) * R) + (m / (m + v) * C)


def build_chart(tag, tag_md, book_df, percentile=0.60):
    book_id = tag_md[tag_md['tag_id'] == tag]['goodreads_book_id'].unique()
    df = book_df[book_df['goodreads_book_id'] == book_id]
    vote_counts = df[df['work_ratings_count'].notnull()]['work_ratings_count'].astype('int')
    vote_averages = df[df['average_rating'].notnull()]['average_rating'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['work_ratings_count'] >= m) & (df['work_ratings_count'].notnull()) & (df['average_rating'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['work_ratings_count'] = qualified['work_ratings_count'].astype('int')
    qualified.rename(columns={"work_ratings_count": "ratings_count"}, inplace=True)
    qualified['average_rating'] = qualified['average_rating'].astype('int')

    qualified['weighted_rating'] = qualified.apply(
        lambda x: (x['ratings_count'] / (x['ratings_count'] + m) * x['average_rating']) + (m / (m + x['ratings_count']) * C),
        axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False)
    return qualified

def filter_keywords(x,tag_bag):
    words = []
    for i in x:
        if i in tag_bag:
            words.append(i)
    return words

def stem_language_code(x):
    if x.startswith('en'):
        return 'eng'
    else:
        return x

def get_recommendations(title, indices, titles, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

def scale_array(unscale_data):
    scale_data = (unscale_data - np.min(unscale_data)) / np.ptp(unscale_data)
    return scale_data

def split_dataframe(self, df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
      df: a dataframe.
      holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
      train: dataframe for training
      test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test






