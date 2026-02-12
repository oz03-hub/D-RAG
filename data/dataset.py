import datasets
import json

def load_dataset(addr, cache_dir):
    def gen():
        with open(addr, 'r') as f:
            dataset = json.load(f)
            for data in dataset:
                yield data
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)