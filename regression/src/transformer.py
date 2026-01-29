###########################################
# This file is folked from https://github.com/chengxiang/LinearTransformer
#
# B: batch-size of prompts
# N: number of context examples (excluding query)
# d: dimension of covariates
# P,Q : (d+2) x (d+2)
# Z, Output : B x (N+1) + (d+2)
###########################################

import math

import torch
from torch import nn


def attention(V, Q, Z):
    """
    Compute attention with a combined key-query matrix Q and value matrix V.
    """
    _, N, D = Z.shape
    Attn = torch.einsum('BNi, ij, BMj -> BNM', (Z, Q, Z))
    values = torch.einsum('ij, BNj -> BNi', (V, Z))
    Output = torch.einsum('BNM, BMi -> BNi', (Attn, values))
    return Output / N


class Transformer_F(nn.Module):
    """The Transformer module
    n_layer : the number of layers
    n_head : the number of heads
    var : the variance of initialization.
    v_matrix: contains the value matrices, has dimension n_layer x n_head x 1 x (d+2) x (d+2)
    q_matrix: contains the product of key and query matrices, has dimension n_layer x n_head x 1 x (d+2) x (d+2)
    """
    def __init__(self, n_layer, n_head, d, params_std=0.05):
        super().__init__()
        self.register_parameter('v_matrix', nn.Parameter(torch.zeros(n_layer, n_head, 1, d+2, d+2)))
        self.register_parameter('q_matrix', nn.Parameter(torch.zeros(n_layer, n_head, 1, d+2, d+2)))

        with torch.no_grad():
            self.v_matrix.normal_(0, params_std)
            self.q_matrix.normal_(0, params_std)
        self.n_layer = n_layer
        self.n_head = n_head
        self.d = d

    def forward(self, Z):
        for i in range(self.n_layer):
            Zi = Z
            residues = 0
            for j in range(self.n_head):
                Vij = self.v_matrix[i, j, 0, :, :]
                Qij = self.q_matrix[i, j, 0, :, :]
                residues = residues + attention(Vij, Qij, Zi)
            Z = Zi + residues / self.n_head
        return Z

    @property
    def allparam(self):
        "Stacked view of value and key-query parameters for logging."
        return torch.cat((self.v_matrix, self.q_matrix), dim=2)

    @allparam.setter
    def allparam(self, value):
        with torch.no_grad():
            self.v_matrix.copy_(value[:, :, 0:1, :, :])
            self.q_matrix.copy_(value[:, :, 1:2, :, :])


def in_context_loss(model, Z, y):
    "evaluate the loss of model, given data (Z,y)"
    output = model(Z)
    return nn.functional.mse_loss(output[:, -1, -1], y, reduction='mean')


# one-step update of (non-)clipping algotirthm
def clip_and_step(model, optimizer, toclip, threshold=1.0):
    grad_v = model.v_matrix.grad
    grad_q = model.q_matrix.grad

    grads = [g for g in (grad_v, grad_q) if g is not None]
    if len(grads) == 0:
        norm_p = torch.tensor(0.0, device=model.v_matrix.device)
    elif len(grads) == 1:
        norm_p = grads[0].norm()
    else:
        norm_p = torch.cat(grads, dim=2).norm()

    if toclip and norm_p > threshold:
        scale = threshold / norm_p
        if grad_v is not None:
            grad_v.mul_(scale)
        if grad_q is not None:
            grad_q.mul_(scale)
    optimizer.step()
    return norm_p.item()
