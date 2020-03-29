import numpy as np
import torch
import torch.nn.functional as F

from typing import List

from torch import nn


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.normal_(std=0.001)


class DAEEncoder(nn.Module):
    def __init__(self, q_dims: List[int], dropout: List[float]):
        super().__init__()

        self.q_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(p), nn.Linear(inp, out))
                for p, inp, out in zip(dropout, q_dims[:-1], q_dims[1:])
            ]
        )
        init_weights(self)

    def forward(self, X):
        h = F.normalize(X, p=2, dim=1)
        for layer in self.q_layers:
            h = torch.tanh(layer(h))
        return h


class VAEEncoder(nn.Module):
    def __init__(self, q_dims: List[int], dropout: List[float]):
        super().__init__()

        self.q_dims = q_dims
        q_dims_ = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(p), nn.Linear(inp, out))
                for p, inp, out in zip(dropout, q_dims_[:-1], q_dims_[1:])
            ]
        )
        init_weights(self)

    def forward(self, X):
        h = F.normalize(X, p=2, dim=1)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, p_dims: List[int], dropout: List[float]):
        super().__init__()

        self.p_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(p), nn.Linear(inp, out))
                for p, inp, out in zip(dropout, p_dims[:-1], p_dims[1:])
            ]
        )

    def forward(self, X):
        h = X
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h


class MultiDAE(nn.Module):
    def __init__(
        self,
        p_dims: List[int],
        dropout_enc: List[float],
        dropout_dec: List[float],
        q_dims: List[int] = None,
    ):
        super().__init__()

        self.encode = DAEEncoder(q_dims, dropout_enc)
        self.decode = Decoder(p_dims, dropout_dec)

    def forward(self, X):
        return self.decode(self.encode(X))


class MultiVAE(nn.Module):
    def __init__(
        self,
        p_dims: List[int],
        dropout_enc: List[float],
        dropout_dec: List[float],
        q_dims: List[int] = None,
    ):
        super().__init__()

        self.encode = VAEEncoder(q_dims, dropout_enc)
        self.decode = Decoder(p_dims, dropout_dec)

    def forward(self, X):
        mu, logvar = self.encode(X)
        sampled_z = self.sample_z(mu, logvar)
        return self.decode(sampled_z), mu, logvar

    def sample_z(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
