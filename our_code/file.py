import os
import ijson
import gzip
import json
from decimal import Decimal

def convert_decimal(obj):
    if isinstance(obj, list):
        return [convert_decimal(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj

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

create_jsonl_datasets("/home/palu001/Github/dataset/data/A/train.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/A/test.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/B/train.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/B/test.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/C/train.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/C/test.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/D/train.json.gz")
create_jsonl_datasets("/home/palu001/Github/dataset/data/D/test.json.gz")