import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)  # полезно для retrieval

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64,
                 user_feat_dim=0, item_feat_dim=0):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.use_user_feats = user_feat_dim > 0
        self.use_item_feats = item_feat_dim > 0

        user_in = emb_dim + user_feat_dim
        item_in = emb_dim + item_feat_dim

        self.user_tower = MLP(user_in, [128, 64], emb_dim)
        self.item_tower = MLP(item_in, [128, 64], emb_dim)

    def encode_user(self, user_ids, user_feats=None):
        u = self.user_emb(user_ids)
        if self.use_user_feats:
            u = torch.cat([u, user_feats], dim=-1)
        return self.user_tower(u)

    def encode_item(self, item_ids, item_feats=None):
        i = self.item_emb(item_ids)
        if self.use_item_feats:
            i = torch.cat([i, item_feats], dim=-1)
        return self.item_tower(i)

    def score(self, user_vecs, item_vecs):
        return (user_vecs * item_vecs).sum(dim=-1)