import torch_geometric
import torch_geometric.datasets as tgset
from data import policy2transform, preprocess, SubgraphData, TUDataset, PTCDataset, Sampler

policy = "node_large_degree_deleted"
for policy in ['node_deleted', 'node_large_degree_deleted', 'node_small_degree_deleted', 'node_in_bcc_deleted', 'node_in_tree_deleted']:
    dataset = tgset.ZINC(root=f'dataset/{policy}', subset=True)
    stats = [
        g.subgraph_batch.max().item()+1
        for g in dataset
    ]
    import numpy as np
    print(policy, np.mean(stats))