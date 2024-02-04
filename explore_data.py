#%%

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def build_graph_from_cid(cid: int, data_root = './data/raw'):
    # Parse graph file
    edge_list = []
    with open(f"{data_root}/{cid}.graph", 'r', encoding="utf-8") as graph_f:
        next(graph_f)
        for line in graph_f:
            if line != "\n":
                edge = (int(line.split()[0]), int(line.split()[1]) )
                edge_list.append(edge)
            else:
                break
    # Plot graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G


def explore_graph_dataset(data_root = './data/raw'):
    nodes_nb = 0
    edges_nb = 0
    for filename in tqdm(os.listdir(data_root)[1:], desc='Looping over the cid.graph files...'):
        cid = filename.split('.')[0]
        G = build_graph_from_cid(cid, data_root)
        nodes_nb += G.number_of_nodes()
        edges_nb += G.number_of_edges()
    return len(os.listdir(data_root)), nodes_nb, edges_nb

def explore_train_graph_dataset():
    # Reading the cids of the train dataset
    train_cids = []
    train_tsv_path = './data/train.tsv'
    with open(train_tsv_path, 'r', encoding="utf-8") as train_tsv_file:
        for line in train_tsv_file:
            train_cids.append(line.split('\t')[0])
    # Collecting metadata on training graphs
    nodes_nb = 0
    edges_nb = 0
    for cid in tqdm(train_cids, desc='Looping over training graphs only'):
        G = build_graph_from_cid(cid)
        nodes_nb += G.number_of_nodes()
        edges_nb += G.number_of_edges()
    return len(train_cids), nodes_nb, edges_nb
    

def explore_tokenizer(text_dataset, tokenizer, min_relevant_token_id=998):
    prop_known_tokens = 0
    tot_nb_tokens = 0
    for sentence in text_dataset.sentences:
        # We count the number of identified tokens and we divide by 
        # the total number of tojens (minus 2 for start and end tokens)
        tokens = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='np')
        nb_known_tokens = np.sum(tokens > min_relevant_token_id)
        prop_known_tokens += nb_known_tokens / (tokens.shape[1]-2)
        tot_nb_tokens += tokens.shape[1]-2
    return tot_nb_tokens, prop_known_tokens / len(text_dataset)

# %%
