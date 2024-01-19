import matplotlib.pyplot as plt
import numpy as np

def plot_metric(train_metric, val_metric):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    assert train_metric.shape[0] == val_metric.shape[0]
    nb_epochs = train_metric.shape[0]
    ax.plot(range(1, nb_epochs+1), val_metric, label='Validation', c='r', linewidth=5)
    ax.plot(range(1, nb_epochs+1), train_metric[:, -1], label='Training', c='b', linewidth=5)
    for i in range(nb_epochs):
        iterations = np.linspace(i, i+1, train_metric.shape[1])
        ax.plot(iterations, train_metric[i, :], c='b', linewidth=3, linestyle= '--')
    ax.legend()
    return fig, ax

