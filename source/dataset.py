import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch_geometric.data import Dataset
import json
import torch
from torch_geometric.data import Data
import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import random
from source.utils import add_zeros
import gzip

#This dataset class is used to load graph data from a JSONL file. Useful when you don't have enough memory to load the entire dataset into memory at once.
# class GraphJSONDataset(Dataset):
#     def __init__(self, path, transform=None, pre_transform=None, has_labels=True):
#         self.path = path
#         self.has_labels = has_labels
#         self.offsets = []
#         with open(self.path, 'r', encoding='utf-8') as f:
#             offset = 0
#             for line in f:
#                 self.offsets.append(offset)
#                 offset += len(line.encode('utf-8'))
#         super().__init__(None, transform, pre_transform)

#     def len(self):
#         return len(self.offsets)

#     def get(self, idx):
#         offset = self.offsets[idx]
#         with open(self.path, 'r', encoding='utf-8') as f:
#             f.seek(offset)
#             line = f.readline()
#         return self.parse_graph(line)

    # def parse_graph(self, line):
    #     item = json.loads(line)
    #     edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
    #     edge_attr = torch.tensor(item["edge_attr"], dtype=torch.float) if "edge_attr" in item else None
    #     num_nodes = item['num_nodes']
    #     y = torch.tensor(item['y'][0], dtype=torch.long) if self.has_labels else None
    #     return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.num_graphs, self.graphs_dicts = self._count_graphs() 
        super().__init__(None, transform, pre_transform)

    def len(self):
        return self.num_graphs  
    
    def get(self, idx):
        return dictToGraphObject(self.graphs_dicts[idx])

    def _count_graphs(self):
        with gzip.open(self.raw, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)  # Load full JSON array without keeping references
            return len(graphs_dicts),graphs_dicts  # Return number of graphs

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_split, test_path = None, train_path = None):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.train_path is None:
                raise ValueError("Training path is None during setup for 'fit'")
            
            full_train_dataset = GraphDataset(self.train_path, transform = add_zeros)
            total_size = len(full_train_dataset)
            print(f"Dataset size: {total_size}")
            indices = list(range(total_size))
            random.shuffle(indices)

            val_size = int(self.val_split * total_size)
            self.train_dataset = Subset(full_train_dataset, indices[val_size:])
            print(f"Train dataset size: {len(self.train_dataset)}")
            self.val_dataset = Subset(full_train_dataset, indices[:val_size])
            print(f"Validation dataset size: {len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:
            if self.test_path is None:
                raise ValueError("Test path is None during setup for 'test'")
            self.test_dataset = GraphDataset(self.test_path, transform = add_zeros)
            print(f"Test dataset size: {len(self.test_dataset)}")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)