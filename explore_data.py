import networkx as nx
import matplotlib.pyplot as plt

def draw_graph_from_cid(cid: int, data_root = './data/raw'):
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
    nx.draw(G)
    plt.show()
    return G

draw_graph_from_cid(69)
    