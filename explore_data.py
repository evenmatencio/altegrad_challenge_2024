import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

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
    return G


# def explore_graph_dataset(data_root = './data/raw'):
#     for filename in os.listdir(data_root):
#         cid = filename.split('.')[0]
#         graph = 
    

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
