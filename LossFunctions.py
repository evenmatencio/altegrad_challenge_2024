"Loss functions used for training and evaluating the model."

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

CE = torch.nn.CrossEntropyLoss()

def Contrastive_Loss(v1, v2):
    """ 
    Computes a kind of Contrastive Loss, measuring the similarity between corresponding vectors
    from the matrices v1 and v2. More precisely, it penalizes the distance between the identity
    matrix of size [n_batch, n_batch] and the dot product <v1, v2^T>. 

    In other words we want to maximise the dot product between the representation of graph i v1_i
    and the representation of text i v2_i, for all i in {1, ..., n_batch}, while minimizing the
    dot product between one graph representation v1_i and all other texts representation v2_j,
    j!=i, and reciprocaly.

    Note: both argument can be switched.
    Args:
        v1 (torch.Tensor): the graph representation in the feature space created by the graph model.
            shape: [n_batch, n_features_out]
        v2 (torch.Tensor): the text representation in the feature space created by the text model.
            shape: [n_batch, n_features_out]
    Returns:
        The resulting error.
    """
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

###############################################################
### Tools functions for the computation of the InfoNCE loss

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def InfoNCE(v1, v2, temperature=0.1, reduction='mean'):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        v1 (torch.Tensor): the graph representation in the feature space created by the graph model.
            shape: [n_batch, n_features_out]
        v2 (torch.Tensor): the text representation in the feature space created by the text model.
            shape: [n_batch, n_features_out]

        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.

  Input shape:
        v1: (N, D) Tensor with query samples (e.g. embeddings of the input).
        v2: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).

    Returns:
         Value of the InfoNCE Loss.
    """

    # Normalize to unit vectors
    v1, v2 = normalize(v1, v2)

    # Cosine between all combinations
    logits = v1 @ transpose(v2)

    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(v1), device=v1.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)




class NTXent(torch.nn.Module):
    """
    Code from https://github.com/yuyangw/MolCLR/blob/master/utils/nt_xent.py
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    

