import torch
from torch.utils.data import Sampler
import numpy as np

class TimeAdjustedSampler(Sampler):
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.p_t = dataset.p_t
        self.cumulative_lengths = dataset.cumulative_lengths

    def __iter__(self):
        num_batches = len(self)
        for _ in range(num_batches):
            batch_indices = []
            while len(batch_indices) < self.batch_size:
                # sample a simulation length
                sample = np.random.choice(len(self.p_t), p=self.p_t)
                
                # find idxs of simulation of length t
                idx_start = self.cumulative_lengths[sample]
                idx_end = self.cumulative_lengths[sample + 1]
                
                # uniformly choose idx to sample
                idx = np.random.randint(idx_start, idx_end)
                batch_indices.append(idx)
            
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size