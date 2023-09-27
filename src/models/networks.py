import torch.nn as nn
import torch

class TripletModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.ssl_features, emb_dim)
        )
    
    def forward(self, wav):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x_tr = torch.mean(x, 1)
        x = self.embedding_layer(x_tr)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

class Origw2v(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(Origw2v, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
    
    def forward(self, wav):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x_tr = torch.mean(x, 1)
        return x_tr