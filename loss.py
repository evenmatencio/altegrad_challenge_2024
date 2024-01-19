import torch

CE = torch.nn.CrossEntropyLoss()

def original_contrastive_loss(v1, v2):
    """ 
    Computes a kind of Contrastive Loss, measuring the similarity between corresponding vectors
    from the matrices v1 and V2. More precisely, it penalizes the distance between the identity matrix
    of size [n_batch, n_batch] and the dot product <v1, v2^T>. 

    In other words we want to maximise the dot product between the representation of graph i v1_i and
    the representation of text i v2_i, for all i in {1, ..., n_batch}, while minimizing the dot product
    between one graph representation v1_i and all other texts representation v2_j, j!=i, and reciprocaly.

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