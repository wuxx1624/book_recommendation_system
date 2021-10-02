import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from sklearn.model_selection import train_test_split, cross_validate
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utilities_fordummy import *
from CFModel import *
from torch.autograd import Variable


class BookRecommender:
    def __init__(self, quantile_rate, save_path=None):
        self._load_data()
        self._data_process()
        self._weighted_rating(quantile_rate)
        self.user_num = self.ratings.user_id.max()
        self.book_num = self.ratings.book_id.max()
        self.save_path = save_path

    def _load_data(self):
        # 1. load data
        self.ratings = pd.read_csv('./goodbooks-10k/ratings.csv')
        self.to_read = pd.read_csv('./goodbooks-10k/to_read.csv')
        self.books = pd.read_csv('./goodbooks-10k/books.csv')

        self.tags = pd.read_csv('./goodbooks-10k/tags.csv')
        self.book_tags = pd.read_csv('./goodbooks-10k/book_tags.csv')

    def load_model(self):
        latest_checkpoint_file = retrieve_lastest_file(self.save_path)
        if latest_checkpoint_file is not None:
            checkpoint = torch.load(latest_checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.train_loss = checkpoint['train_loss']
            self.val_loss = checkpoint['val_loss']

    def _save(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }, os.path.join(self.save_path, 'epoch_%s.model' % (self.epoch,)))

    def _data_process(self):
        # 2. data prepare
        # 2.1 merge tags with book_tags and remove negative tags count, then merge into books
        self.book_tags = self.book_tags.merge(self.tags, on='tag_id')
        self.book_tags.loc[self.book_tags['count'] < 0, 'count'] = 0
        self.book_tags = self.book_tags.merge(self.books[['goodreads_book_id', 'book_id']], on='goodreads_book_id')
        # remove the tags which is lease than 1
        self.book_tags_select = self.book_tags.drop(self.book_tags.loc[self.book_tags['count'] <= 1].index).reset_index(
            drop=True)
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

    def _build_weighted_rating_char_per_tag(self):
        s = self.book_tags.apply(lambda x: pd.Series(x['tag_id']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'tag_id'
        tag = ' '  # input tag
        top_book_per_tag = build_chart(tag, self.book_tags)

    def content_based_recommender(self):
        # build content based recommender
        # query a book title, return similar books
        # 1. select features: tags, authors and language_code
        self.books = self.books.reset_index()
        titles = self.books['title']
        indices = pd.Series(self.books.index, index=self.books['title'])

        # 2. process each features
        self.qualified['tag_name'] = self.qualified['tag_name'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x])
        # merge some similar language code, such as en, eng, en-CA, etc,, and remove nan value
        self.qualified['language_code'] = self.qualified['language_code'].fillna("").apply(
            lambda x: [stem_language_code(x)])
        # make sure the system will not confuse with authors who have the same first name
        self.qualified['authors'] = self.qualified['authors'].fillna("").astype('str').apply(
            lambda x: [str.lower(x.replace(" ", ""))])

        # 3. join all features
        self.qualified['soup'] = self.qualified['tag_name'] + self.qualified['language_code'] + self.qualified[
            'authors']
        self.qualified['soup'] = self.qualified['soup'].apply(lambda x: ' '.join(x))

        # 4. vectorization
        count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        count_matrix = count.fit_transform(self.qualified['soup'])

        # 5. calculate similarity
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        title = 'To Kill a Mockingbird'
        content_based_reconmender = get_recommendations(title, indices, titles, self.cosine_sim)
        return content_based_reconmender

    def collaborative_iltering_1(self, user_id, book_id, algorithm='SVD'):
        # build collaborative filtering
        # query a user id, return similar favorite
        selected_dataset = self.ratings[self.ratings['book_id'].isin(list(self.qualified['book_id'].unique()))][
            ['user_id', 'book_id', 'rating']]
        reader = Reader()
        data = Dataset.load_from_df(selected_dataset, reader)
        # split dataset into 5 fold and use cross_validation
        if algorithm == 'SVD':
            self.algo = SVD()
        else:
            self.algo = None
        cross_validate(self.algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        prediction = self.algo.predict(user_id, book_id, r_ui=3)
        return prediction



    def build_model(self):
        """
        Args:
          ratings: a DataFrame of the ratings
          embedding_dim: the dimension of the embedding vectors.
          init_stddev: float, the standard deviation of the random initial embeddings.
        Returns:
          model: a CFModel.
        """
        # Split the ratings DataFrame into train and test.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = EmbeddingNet(self.user_num, self.book_num, n_factors=20).to(device)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=1e-6, weight_decay=1e-5)
        self._train()



    def _train(self):
        n_epochs = 100
        no_improvements = 0
        best_loss = np.inf
        for epoch in range(n_epochs):
            stats = {'epoch': epoch + 1, 'total': n_epochs}
            for phase in ('train', 'val'):
                training = phase == 'train'
                running_loss = 0.0
                n_batches = 0
                iterator = batches(*self.datasets[phase], shuffle=training, bs=256)
                for batch in iterator:
                    x_batch, y_batch = [b.to(device) for b in batch]
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(training):
                        prediction = self.model(x_batch[:, 0], x_batch[:, 1])
                        loss = self.loss(prediction, y_batch)
                        if training:
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item()
                epoch_loss = running_loss / self.dataset_sizes[phase]
                stats[phase] = epoch_loss
                if phase == 'train':
                    print('epoch %d loss: %f' % (epoch + 1, epoch_loss))
                if phase == 'val':
                    if epoch_loss < best_loss:
                        print('loss improvement on epoch: %d' % (epoch + 1))
                        best_loss = epoch_loss
                        no_improvements = 0
                    else:
                        no_improvements += 1
                if no_improvements >= 10:
                    break



    def collaborative_iltering(self):
        selected_dataset = self.ratings[self.ratings['book_id'].isin(list(self.qualified['book_id'].unique()))][
            ['user_id', 'book_id', 'rating']]
        selected_dataset['user_id'] = selected_dataset['user_id'].apply(lambda x: (x-1))
        selected_dataset['book_id'] = selected_dataset['book_id'].apply(lambda x: (x - 1))
        train_sets_x, test_sets_x, train_sets_y, test_sets_y = train_test_split(selected_dataset[['user_id','book_id']].values, selected_dataset[['rating']].values, test_size=0.2)
        # scale ratings value to [0 ,1] to help with convergence
        self.datasets = {'train': (train_sets_x, scale_array(train_sets_y)), 'val': (test_sets_x, scale_array(test_sets_y))}
        self.dataset_sizes = {'train': len(train_sets_x), 'val': len(test_sets_x)}
        self.build_model()
        # train, test = self.split_dataframe(df)

    def hybrid(self, userId, title):
        indices = pd.Series(self.books.index, index=self.books['title'])
        idx = indices[title]
        book_id = self.books.loc[idx]['book_id']
        # print(idx)

        sim_scores = list(enumerate(self.cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        book_indices = [i[0] for i in sim_scores]

        movies = self.qualified.iloc[book_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
        movies['est'] = movies['id'].apply(lambda x: self.algo.predict(userId, indices_map.loc[x]['book_id']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)


recommender = BookRecommender(quantile_rate=0.8)
user_id = 1
book_id = 258
prediction = recommender.collaborative_iltering_1(user_id, book_id)
print(prediction)
