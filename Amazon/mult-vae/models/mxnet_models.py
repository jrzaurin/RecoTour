import numpy as np
import mxnet as mx

from typing import List

from mxnet import nd, init, autograd
from mxnet.gluon import nn, Block, HybridBlock


class DAEEncoder(HybridBlock):
    def __init__(self, q_dims: List[int], dropout: List[float]):
        super().__init__()

        with self.name_scope():
            self.q_layers = nn.HybridSequential(prefix="q_net")
            for p, inp, out in zip(dropout, q_dims[:-1], q_dims[1:]):
                self.q_layers.add(nn.Dropout(p))
                self.q_layers.add(nn.Dense(in_units=inp, units=out))

    def hybrid_forward(self, F, X):
        h = F.L2Normalization(X)
        for layer in self.q_layers:
            h = F.tanh(layer(h))
        return h


class VAEEncoder(HybridBlock):
    def __init__(self, q_dims: List[int], dropout: List[float]):
        super().__init__()

        q_dims_ = q_dims[:-1] + [q_dims[-1] * 2]
        with self.name_scope():
            self.q_layers = nn.HybridSequential(prefix="q_net")
            for p, inp, out in zip(dropout, q_dims_[:-1], q_dims_[1:]):
                self.q_layers.add(nn.Dropout(p))
                self.q_layers.add(nn.Dense(in_units=inp, units=out))

    def hybrid_forward(self, F, X):
        h = F.L2Normalization(X)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu, logvar = F.split(h, axis=1, num_outputs=2)
        return mu, logvar


class Decoder(HybridBlock):
    def __init__(self, p_dims: List[int], dropout: List[float]):
        super().__init__()

        with self.name_scope():
            self.p_layers = nn.HybridSequential(prefix="p_net")
            for p, inp, out in zip(dropout, p_dims[:-1], p_dims[1:]):
                self.p_layers.add(nn.Dropout(p))
                self.p_layers.add(nn.Dense(in_units=inp, units=out))

    def hybrid_forward(self, F, X):
        h = X
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h


class MultiDAE(HybridBlock):
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

    def hybrid_forward(self, F, X):
        return self.decode(self.encode(X))


class MultiVAE(HybridBlock):
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

    def hybrid_forward(self, F, X):
        mu, logvar = self.encode(X)
        if autograd.is_training():
            std = F.exp(0.5 * logvar)
            eps = F.random.normal_like(std)
            mu = (eps * std) + mu
        return self.decode(mu), mu, logvar
