import pandas as pd
import torch
import numpy as np
import math
import os

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

def build_rating_sparse_tensor(self, ratings_df):
    # build sparse matrix
    """
    Args:
        ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
    Returns:
      a torch.sparse representing the ratings matrix
    """
    indices = ratings_df[['user_id', 'book_id']].values
    values = ratings_df['rating'].values
    return torch.sparse_coo_tensor(list(zip(*indices)), values,
                                   size=(self.user_num, self.book_num))

def sparse_mean_square_error(self, sparse_ratings, user_embeddings, book_embeddings):
    """ calculate mean square errors for embeddings
    Args:
      sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
      user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
        dimension, such that U_i is the embedding of user i.
      book_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
        dimension, such that V_j is the embedding of book j.
    Returns:
      A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    predictions = torch.matmul(user_embeddings, book_embeddings.transpose(0, 1)).index_select(0,
                  sparse_ratings.indices[0, :]).index_select(1,sparse_ratings.indices[1,:])
    loss = torch.nn.MSELoss(predictions, sparse_ratings.values)
    return loss


class ReviewsIterator:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]

def batches(X, y, bs=32, shuffle=True):
    for xb, yb in ReviewsIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)

def retrieve_lastest_file(folder):
    files = [os.path.join(folder, fname) for fname in os.listdir(folder)]
    if len(files) == 0:
        return None
    latest_filename = max(files, key=os.path.getmtime)
    return latest_filename