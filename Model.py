"Gather the encoders for text and graph, and the model."

from torch import nn
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from transformers import AutoModel


class GCNGraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, n_gnn_layers, use_aggregation_class):
        super(GCNGraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.n_layers = n_gnn_layers
        self.conv_0 = GCNConv(num_node_features, graph_hidden_channels)
        if self.n_layers < 2:
            raise ValueError("Number layers must be greater than 1.")
        self.hidden_gnn_layers = nn.ModuleList([GINConv(graph_hidden_channels, graph_hidden_channels) for i in range(n_gnn_layers-1)])
        if use_aggregation_class:
            gate_nn = nn.Linear(graph_hidden_channels, graph_hidden_channels)
            to_attention_nn = nn.Linear(graph_hidden_channels, graph_hidden_channels)
            self.aggregator = AttentionalAggregation(gate_nn, to_attention_nn)
            self.aggragation_func = lambda x, index: self.aggregator.forward(x, index, dim=0)
        else:
            self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
            self.aggragation_func = lambda x, index: self.mol_hidden1(global_mean_pool(x, index)).relu()
        self.mol_hidden2 = nn.Linear(nhid, nout)
            

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv_0(x, edge_index)
        x = x.relu()
        for layer in self.hidden_gnn_layers:
            x = layer(x, edge_index)
            x = x.relu()
        x = self.aggragation_func(x, index=batch)
        x = self.mol_hidden2(x)
        return x

class TextEncoderWithHead(nn.Module):
    def __init__(self, model_name, n_out, bert_out=768):
        super(TextEncoderWithHead, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(bert_out, n_out)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        encoded_text = self.dropout(encoded_text.last_hidden_state[:,0,:])
        return self.head(encoded_text)
    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:,0,:]


class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, graph_gnnlayers, text_head=False, use_aggregation_class=False):
        super(Model, self).__init__()
        self.graph_encoder = GCNGraphEncoder(num_node_features, nout, nhid, graph_hidden_channels, graph_gnnlayers, use_aggregation_class)
        if text_head:
            self.text_encoder = TextEncoderWithHead(model_name, nout)
        else:
            self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder
