"Similar to run_training_from_github_scripts.ipynb, training pipeline."

import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
from torch_geometric.loader import DataLoader

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from Model import Model
from train_val_test import train, test
import LossFunctions


##################################################
## TRAINING


# Select text model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300, graph_gnnlayers=3, text_head=False, use_aggregation_class=True) # nout = bert model hidden dim
model.to(device)

# Load data
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer, nrows=9)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer, nrows=9)

# Hyper-parameters
nb_epochs = 2
batch_size = 2
learning_rate = 2e-5
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
hyper_param = {
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "loss": "original_loss", 
    "LRAP": "using cosine similarity" ,
    "optimizer": optimizer.__str__(),
    "model": model.__str__()
    }
loss = LossFunctions.NTXent('cpu', batch_size, 0.1, True)

# Save path
save_path = './model_checkpoints/test'

loss_func = LossFunctions.NTXent(device, batch_size, 0.1, True)
train(nb_epochs, optimizer, loss_func, model, train_loader, val_loader, save_path, device, hyper_param, print_every=2)



# ##################################################
# ## TESTING

# model_path = os.path.join(save_path, 'model0.pt')

# test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
# test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

# text_embeddings, graph_embeddings = test(model_path, model, test_cids_dataset, test_text_dataset, device)



# # ##################################################
# # ## GENERATE OUTPUT

# from sklearn.metrics.pairwise import cosine_similarity

# similarity = cosine_similarity(text_embeddings, graph_embeddings)

# solution = pd.DataFrame(similarity)
# solution['ID'] = solution.index
# solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
# solution.to_csv('submission.csv', index=False)
