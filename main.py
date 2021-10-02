# recommendation system for goodbooks-10k dataset
# author: Fei Wu
# created time: 09/12/2021

import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, BaselineOnly
from surprise.model_selection import cross_validate
import pickle
import os
import numpy as np
from utilities import *


class BookRecommender:
    def __init__(self, quantile_rate=0.8, mode='train', save_path='./model'):
        self.save_path = save_path
        if mode == 'train':
            self._load_data()
            self._data_process()
            self._weighted_rating(quantile_rate)
            self.content_based_recommender()
            self.collaborative_filtering(save_model=True)
            self.user_num = self.ratings.user_id.max()
            self.book_num = self.ratings.book_id.max()
            self._save()
        elif mode == 'eval':
            self.load()
        else:
            return

    def _save(self):
        # save trained model, qualified books form and similarity matrix
        # save path is path_to_code/model/
        pickle.dump(self.algo, open(os.path.join(self.save_path, 'finalized_model.sav'), 'wb'))
        self.qualified.to_csv(os.path.join(self.save_path, 'qualified.csv'), index=False)
        np.save(os.path.join(self.save_path, 'similarity.npy'), self.cosine_sim)

    def load(self):
        # load tained model, qualified books form, similarity matrix and to_read.csv ratings.csv
        self.algo = pickle.load(open(os.path.join(self.save_path, 'finalized_model.sav'), 'rb'))
        self.qualified = pd.read_csv(os.path.join(self.save_path, 'qualified.csv'))
        self.cosine_sim = np.load(os.path.join(self.save_path, 'similarity.npy'))
        self.to_read = pd.read_csv('./goodbooks-10k/to_read.csv')
        self.ratings = pd.read_csv('./goodbooks-10k/ratings.csv')

    def _load_data(self):
        # 1. load data from all csv files in the dataset
        self.ratings = pd.read_csv('./goodbooks-10k/ratings.csv')
        self.to_read = pd.read_csv('./goodbooks-10k/to_read.csv')
        self.books = pd.read_csv('./goodbooks-10k/books.csv')

        self.tags = pd.read_csv('./goodbooks-10k/tags.csv')
        self.book_tags = pd.read_csv('./goodbooks-10k/book_tags.csv')

    def _data_process(self):
        # 2. data prepare
        # 2.1 merge tags with book_tags and remove negative tags count, then merge book_id from books.csv into book_tags
        self.book_tags = self.book_tags.merge(self.tags, on='tag_id')
        self.book_tags.loc[self.book_tags['count'] < 0, 'count'] = 0
        self.book_tags = self.book_tags.merge(self.books[['goodreads_book_id', 'book_id']], on='goodreads_book_id')
        # 2.2 remove the tags which is lease than 1 and the "to_read" tage
        self.book_tags_select = self.book_tags.drop(
            self.book_tags.loc[(self.book_tags['count'] <= 1) |
                               (self.book_tags['tag_name'].apply(
                                   lambda x: x.startswith('to-read')))].index).reset_index(drop=True)
        # 2.3 stack tags for each book and merge book_tags into books
        d = {'tag_id': list, 'tag_name': list, 'count': 'first'}
        self.book_tags_organize = self.book_tags_select.groupby(['goodreads_book_id', 'book_id'], sort=False,
                                                                as_index=False).agg(d).reindex(
            columns=self.book_tags_select.columns)
        self.books = self.books.merge(self.book_tags_organize[['book_id', 'tag_id', 'tag_name']], on='book_id')

    def _weighted_rating(self, quantile_rate=0.8):
        # 3. calculate weighted ratings for each book
        # the weighted ratings is defined by v/(v+m) * R + m/(v+m) * C
        # v: number of ratings for the book
        # m: minimum ratings required, suppose we need at least 80% ratings
        # R: average ratings for the book
        # C: mean rating across the whole chart
        ratings_count = self.books[self.books['work_ratings_count'].notnull()]['work_ratings_count'].astype('int')
        ratings_average = self.books[self.books['average_rating'].notnull()]['average_rating'].astype('int')
        m = ratings_count.quantile(quantile_rate)
        C = ratings_average.mean()
        # extract qualified books
        self.qualified = self.books[
            (self.books['work_ratings_count'] >= m) & (self.books['work_ratings_count'].notnull()) & (
                self.books['average_rating'].notnull())][
            ['book_id', 'goodreads_book_id', 'best_book_id', 'work_id',
             'books_count', 'isbn', 'isbn13', 'authors', 'original_publication_year',
             'original_title', 'title', 'language_code', 'average_rating',
             'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
             'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
             'image_url', 'small_image_url', 'tag_id', 'tag_name']]
        self.qualified['work_ratings_count'] = self.qualified['work_ratings_count'].astype('int')
        self.qualified['average_rating'] = self.qualified['average_rating'].astype('int')
        print(self.qualified.shape)
        # calculate weigted ratings for selected books and sort in descending order
        self.qualified['weighted_rating'] = weighted_rating(self.qualified, m, C)
        self.qualified = self.qualified.sort_values('weighted_rating', ascending=False)

    def content_based_recommender(self):
        # 4. build content based recommender
        # query a book title, return similar books
        # 4.1 select features: tags, authors and language_code
        self.books = self.books.reset_index()
        titles = self.books['title']
        indices = pd.Series(self.books.index, index=self.books['title'])

        # 4.2 process each features
        self.qualified['tag_name'] = self.qualified['tag_name'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x])
        # merge some similar language code, such as en, eng, en-CA, etc,, and remove nan value
        self.qualified['language_code'] = self.qualified['language_code'].fillna("").apply(
            lambda x: [stem_language_code(x)])
        # make sure the system will not confuse with authors who have the same first name
        self.qualified['authors'] = self.qualified['authors'].fillna("").astype('str').apply(
            lambda x: [str.lower(x.replace(" ", ""))])

        # 4.3 join all features
        self.qualified['soup'] = self.qualified['tag_name'] + self.qualified['language_code'] + self.qualified[
            'authors']
        self.qualified['soup'] = self.qualified['soup'].apply(lambda x: ' '.join(x))

        # 4.4 vectorization
        count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        count_matrix = count.fit_transform(self.qualified['soup'])

        # 4.5 calculate similarity
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)

    def collaborative_filtering(self, algorithm='baseline', save_model=False):
        # 5. build collaborative filtering
        # 5.1 query a user id, return similar favorite
        selected_dataset = self.ratings[self.ratings['book_id'].isin(list(self.qualified['book_id'].unique()))][
            ['user_id', 'book_id', 'rating']]
        reader = Reader()
        data = Dataset.load_from_df(selected_dataset, reader)
        # 5.2 split dataset into 5 fold and use cross_validation
        if algorithm == 'SVD':
            self.algo = SVD()
        elif algorithm == 'baseline':
            self.algo = BaselineOnly()
        cross_validate(self.algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    def predict_collaborative_iltering(self, user_id, book_id):
        prediction = self.algo.predict(user_id, book_id, r_ui=3)
        return prediction

    def hybrid(self, userId, category='title', item=None):
        # 6. combine content-based filer with collaborative filter
        # @param:
        # input: userId: query user id, [1, userNum]
        #        category: query category, one of ['tag_name', 'book_id', 'title', 'popularity', 'rating']
        #        item: query content, e.g., if category is 'title', item is one of the titles in the dataset;
        #              if category is 'tag_name', item is one of the tags in the dataset
        # output: top 10 recommended books in dataframe, the columns are as follows:
        #  ['book_id', 'work_id', 'isbn', 'isbn13', 'authors', 'original_publication_year','title', 'language_code',
        #  'average_rating','work_ratings_count', 'work_text_reviews_count', 'image_url', 'small_image_url']
        if category == 'tag_name':
            idx_list = self.qualified.loc[self.qualified['tag_name'].apply(lambda x: item in x)].index
        elif category == 'popularity':
            self.qualified = self.qualified.sort_values('work_ratings_count', ascending=False).reset_index()
            return self.qualified[['book_id', 'work_id',
                                   'isbn', 'isbn13', 'authors', 'original_publication_year',
                                   'title', 'language_code', 'average_rating', 'work_ratings_count',
                                   'work_text_reviews_count',
                                   'image_url', 'small_image_url']].head(10)
        elif category == 'rating':
            self.qualified = self.qualified.sort_values('weighted_rating', ascending=False).reset_index()
            return self.qualified[['book_id', 'work_id',
                                   'isbn', 'isbn13', 'authors', 'original_publication_year',
                                   'title', 'language_code', 'average_rating', 'work_ratings_count',
                                   'work_text_reviews_count',
                                   'image_url', 'small_image_url']].head(10)
        else:
            indices = pd.Series(self.qualified.index, index=self.qualified[category])
            idx_list = [indices[item]]

        # get top 50 similar books
        book_indices = []
        for idx in idx_list:
            similarity_scores = list(enumerate(self.cosine_sim[int(idx)]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similarity_scores = similarity_scores[1:51]
            book_indices += [i[0] for i in similarity_scores]
        book_indices = list(set(book_indices))

        # predict ratings for the 50 books
        recommend_books = self.qualified.iloc[book_indices][['book_id', 'work_id',
                                                             'isbn', 'isbn13', 'authors', 'original_publication_year',
                                                             'title', 'language_code', 'average_rating',
                                                             'work_ratings_count', 'work_text_reviews_count',
                                                             'image_url', 'small_image_url']]
        recommend_books['est'] = recommend_books['book_id'].apply(lambda x: self.algo.predict(userId, x, r_ui=3).est)
        recommend_books = recommend_books.sort_values('est', ascending=False)
        # get the to-read list for the user
        to_read_df = self.to_read.loc[self.to_read['user_id'] == userId]
        # get the already-read list for the user
        read_df = self.ratings.loc[self.ratings['user_id'] == userId]
        # find the book list that the user not already-read but to-read
        recommen_list_index = recommend_books.loc[recommend_books['book_id'].apply(
            lambda x: x in list(to_read_df['book_id']) and x not in list(read_df['book_id']))].index
        recommend_list = recommend_books.loc[recommen_list_index]
        # return the top 10 selected books
        if len(recommend_list) >= 10:
            return recommend_list.head(10)
        else:
            rest_recommend = recommend_books.drop(recommen_list_index).head(10 - len(recommend_list))
            recommend_list = recommend_list.append(rest_recommend, ignore_index=True)
            recommend_list.drop(columns=['est'], inplace=True)
            return recommend_list.head(10)


def main(argv):
    # 1. Train model
    # the folder includes pre-trained model, located in ./model/ folder
    # @param:
    # quantile_rate: Due to the limitations of my machine, I selected a portion of the books with a high number of ratings for training.
    #                0.8 means that the selected books need to have more than 80% ratings than rest of the books. There are about 2000 books selected.
    # mode: 'train' for training or 'eval' for evaluation
    # save_path: path to save trained models
    if argv[0] == 'train':
        quantile_rate = float(argv[1])
        recommender = BookRecommender(quantile_rate=quantile_rate, mode='train')
        # recommender = BookRecommender(quantile_rate=0.8, mode='train')
    # 2. Test model
    # The quantile rate is only used when training the model. if you prefer to change the quantile rate, please re-train the model.
    elif argv[0] == 'eval' and len(argv) >= 3:
        user_id = int(argv[1])
        category = argv[2]
        if len(argv) >= 4:
            item = ' '.join(argv[3:])
        else:
            item = None
        if category == 'book_id':
            item = int(item)

        recommender = BookRecommender(mode='eval')
        result = recommender.hybrid(user_id, category=category, item=item)
        # query list contains 3 parts, user id, query category and query item
        # user_id = 1
        # query category can be one of the following: [tag_name, book_id, author, title, popularity, rating]
        # e.g,. query based on tag name
        # result = recommender.hybrid(user_id, category='tag_name', item='fantasy')

        # e.g,. query based on book_id
        # book_id = 258
        # result = recommender.hybrid(user_id, category='book_id', item = book_id)

        # e.g,. query based on title
        # result = recommender.hybrid(user_id, category='title', item = 'To Kill a Mockingbird')

        # e.g,. query based on top 10 popularity
        # result = recommender.hybrid(user_id, category='popularity')

        # e.g,. query based on top 10 rated
        # result = recommender.hybrid(user_id, category='rating')

        # the result includes information ['book_id', 'work_id',
        #              'isbn', 'isbn13', 'authors', 'original_publication_year',
        #              'title', 'language_code', 'average_rating','work_ratings_count', 'work_text_reviews_count',
        #              'image_url', 'small_image_url']
        print(result)
    else:
        print("Please check your input parameters.")

if __name__ == "__main__":
    main(sys.argv[1:])
