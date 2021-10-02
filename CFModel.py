import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_items, n_factors, H1, D_out):
        """
        Simple Feedforward with Embeddings
        """
        super().__init__()
        # user and item embedding layers
        self.user_factors = nn.Embedding(n_users, n_factors,
                                               sparse=True)
        self.item_factors = nn.Embedding(n_items, n_factors,
                                               sparse=True)
        # linear layers
        self.linear1 = nn.Linear(n_factors*2, H1)
        self.linear2 = nn.Linear(H1, D_out)

    def forward(self, users, items):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, items_embedding], 1)
        h1_relu = nn.ReLU(self.linear1(x))
        output_scores = self.linear2(h1_relu)
        return output_scores

    def predict(self, users, items):
        # return the score
        output_scores = self.forward(users, items)
        return output_scores


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
    # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
    # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
        # matrix multiplication
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)