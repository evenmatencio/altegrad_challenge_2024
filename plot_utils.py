"Plot useful graphs."

import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_metric, val_metric):
    "Plot the loss at the end of each epoch (train+val). The training loss inside a batch is in --."
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    assert train_metric.shape[0] == val_metric.shape[0]
    nb_epochs = train_metric.shape[0]
    ax.plot(range(1, nb_epochs+1), val_metric, label='Validation', c='r', linewidth=5)
    ax.plot(range(1, nb_epochs+1), train_metric[:, -1], label='Training', c='b', linewidth=5)
    for i in range(nb_epochs):
        iterations = np.linspace(i, i+1, train_metric.shape[1])
        ax.plot(iterations, train_metric[i, :], c='b', linewidth=3, linestyle= '--')
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss comparison")
    ax.set_xticks(list(range(nb_epochs+1)), list(range(nb_epochs+1)))
    return fig, ax

def plot_lrap(values):
    "Plot the LRAP metric value at the end of each epoch for validation set."
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(list(range(len(values))), values, c='r')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("LRAP")
    ax.set_title("Validation LRAP")
    ax.set_xticks(list(range(len(values))), list(range(1, len(values)+1)))
    return fig, ax

def plot_lrs(values):
    "Plot the learning rates at the end of each epoch"
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(list(range(len(values))), values, c='r')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("learning rate")
    ax.set_title("LR")
    ax.set_xticks(list(range(len(values))), list(range(1, len(values)+1)))
    return fig, ax
