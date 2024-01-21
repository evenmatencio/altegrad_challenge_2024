"Wrap the training, validation and testing steps into functions."

import os
import json
import time
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score as lrap

from loss import original_contrastive_loss
from dataloader import GraphDataset, TextDataset
from plot_utils import plot_losses, plot_lrap


def compute_LRAP_metric(text_embeddings: list, graph_embeddings: list):
    "See label_ranking_average_precision_score from scikit learn."
    y_computed = cosine_similarity(text_embeddings, graph_embeddings)
    y_true = np.eye(N=y_computed.shape[0])
    return lrap(y_true, y_computed)

def train(
    nb_epochs: int,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    train_loader: torch_geometric.data.DataLoader,
    val_loader: torch_geometric.data.DataLoader,
    save_path: str,
    device,
    hyper_param_dict,
    print_every: int = 50,
):
    """
    WARNING:
    -------
    The loss that is used should be adapted to the one we use to compute the LRAP metric.

    For instance, the original_contrastive_loss is based on dot product,
    exactly the same as cosine similarity.
    In this case, both the loss and the LRAP metric rely on the same operation.
    """

    # Saving hyper-parameters
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/hyper_parameters.json", "w+", encoding="utf-8") as json_f:
        json.dump(hyper_param_dict, json_f, indent="    ")

    loss = 0
    losses = []
    val_losses = []
    val_lraps = []
    count_iter = 0
    time1 = time.time()
    best_validation_loss = 1000000

    # Per batch training
    for i in range(nb_epochs):
        print(f'-----EPOCH{i+1}-----')
        model.train()
        for batch in train_loader:
            # Forward step
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            # Metric computation
            current_loss = original_contrastive_loss(x_graph, x_text)
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            loss += current_loss.item()
            count_iter += 1
            if count_iter % print_every == 0:
                time2 = time.time()
                print(f"Iteration: {count_iter}, Time: {time2-time1:.4f} s, training loss: {loss/print_every:.4f}")
                losses.append(loss)
                loss = 0
        model.eval()
        val_loss = 0
        val_graph_embeddings = []
        val_text_embeddings = []
        for batch in val_loader:
            # Forward setp
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            current_loss = original_contrastive_loss(x_graph, x_text)
            val_loss += current_loss.item()
            for x_graph_emb in x_graph.tolist():
                val_graph_embeddings.append(x_graph_emb)
            for x_text_emb in x_text.tolist():
                val_text_embeddings.append(x_text_emb)

        val_lrap = compute_LRAP_metric(val_text_embeddings, val_graph_embeddings)
        val_losses.append(val_loss/len(val_loader))
        val_lraps.append(val_lrap/len(val_loader))

        # Plotting
        if i == 0:
            losses_arr = np.array(losses).reshape([1, len(losses)])
        else:
            losses_arr = np.concatenate((losses_arr, [losses]), axis=0)
            loss_fig, _ =  plot_losses(losses_arr, np.array(val_losses))
            loss_fig.savefig(f"{save_path}/losses.png")
            lrap_fig, _ = plot_lrap(val_lraps)
            lrap_fig.savefig(f"{save_path}/val_lrap.png")
        losses = []
        count_iter = 0

        # Saving best model
        best_validation_loss = min(best_validation_loss, val_loss)
        print(f'-----EPOCH +{i+1}+ ----- done.  Validation loss: {val_loss/len(val_loader)}. Validation LRAP: {val_lrap}')
        if best_validation_loss==val_loss:
            print('validation loss improoved saving checkpoint...')
            save_path_model = os.path.join(save_path, 'model.pt')
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, save_path_model)
            print(f'checkpoint saved to: {save_path_model}')


def test(
    checkpoint_path: str,
    model: torch.nn.Module,
    test_cids_dataset: GraphDataset,
    test_text_dataset: TextDataset,
    device,
    batch_size = 32,
):
    "Test loop."

    # Loading the model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    # Generating representation of the graph test set
    graph_test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)
    graph_embeddings = []
    for batch in graph_test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    # Generating representation of the text test set
    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())

    return text_embeddings, graph_embeddings
