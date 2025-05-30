import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import random
import numpy as np
import ijson
import gzip
import json
from decimal import Decimal

def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def convert_decimal(obj):
    if isinstance(obj, list):
        return [convert_decimal(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj
    
#This function is not used now, but it can help to reduce the RAM usage when loading large datasets. It is used for GraphJSONDataset.
def create_jsonl_datasets(old_path):
    base_output_dir = "/home/palu001/Github/datasets_jsonl"
    rel_path = os.path.relpath(old_path, start='/home/palu001/Github/dataset')
    rel_path_jsonl = rel_path.replace(".json.gz", ".jsonl")  # rimuovo .gz
    output_path = os.path.join(base_output_dir, rel_path_jsonl)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with gzip.open(old_path, "rt", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:  # open normale per output
        parser = ijson.items(fin, "item")
        for graph in parser:
            clean_graph = convert_decimal(graph)
            json.dump(clean_graph, fout)
            fout.write("\n")