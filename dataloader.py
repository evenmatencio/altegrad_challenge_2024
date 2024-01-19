import os
import os.path as osp
import torch
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
import pandas as pd

class GraphTextDataset(Dataset):
    """
    A torch_geometric dataset class, providing "on the fly" data loading methods. 
    Each of its items contains both text and graph-related attributes.
    
    It is intended to be sampled in a torch_geometric.data.DataLoader class. 
    When doig so, the resulting batch objects are torch_geometric.data.batch.Batch object.
    This kind of batch models "a batch of graphs as one big (dicconnected) graph" (as done in TPs). 
    It has the following attributes:
        ## TEXT RELATED
        - input_ids (shape = [n_batch, 256]): the tokens ids corresponding to the chemical description of the graph.
        They are computed using the provided tokenizer, that should be specific to the TextEncoder we want to use. 
        - attention_mask (shape = [n_batch, 256]): the attention mask to be used by the TextEncoder model (DistillBert for instance)
        ## GRAPH RELATED
        - batch (shape=[tot_n_nodes]): the index of the graph to which each node belongs within the batch 
        - x (shape=[tot_n_nodes, 300]): the node features of all the nodes of the big disconnected graph modelling the batch
        (or equivalently of all the nodes of the different graphs in the batch)
        - edge_index (shape)=[2, tot_n_edges]): the edges index of the big disconnected graph modelling the batch
    """
    def __init__(self, root, gt, split, tokenizer=None, transform=None, pre_transform=None, nrows=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None, nrows=nrows)   
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
        edge_index  = []
        x = []
        with open(raw_path, 'r') as f:
            next(f)
            for line in f: 
                if line != "\n":
                    edge = *map(int, line.split()), 
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f: #get mol2vec features:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            text_input = self.tokenizer([self.description[1][cid]],
                                    return_tensors="pt", 
                                    truncation=True, 
                                    max_length=256,
                                    padding="max_length",
                                    add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    
class GraphDataset(Dataset):
    def __init__(self, root, gt, split, transform=None, pre_transform=None, nrows=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None, nrows=nrows)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
        edge_index  = []
        x = []
        with open(raw_path, 'r') as f:
            next(f)
            for line in f: 
                if line != "\n":
                    edge = *map(int, line.split()), 
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        sep = "/"
        if sep not in self.raw_paths[0]:
            sep = '\\'
        for raw_path in self.raw_paths:
            cid = int(raw_path.split(sep)[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256, nrows=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.nrows=nrows
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            if self.nrows != None:
                head = [next(file) for _ in range(self.nrows)]
            else:
                lines = file.readlines()
                head = [line.strip() for line in lines]
        return head
    
    

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
