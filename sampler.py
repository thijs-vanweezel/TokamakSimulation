import torch
from torch.utils.data import Sampler
import numpy as np

class TimeAdjustedSampler(Sampler):
    def __init__(self, dataset, batch_size: int, omega: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.omega = omega

        self.timesteps = dataset.timesteps
        self.p_t = p_t = dataset.calculate_p_t(omega=omega)

    def __iter__(self) -> [[np.array, np.array, np.array]]:
        # Gather data untill batch size is reached
        batches = []
        while len(batches) < self.batch_size:

            # sample a simulation length
            sample = np.random.choice(len(self.p_t), p=self.p_t)
            
            # find idxs of simulation of length t
            idx_start = self.dataset.cumulative_lengths[sample]
            idx_end = self.dataset.cumulative_lengths[sample + 1]

            # uniformly choose idx to sample and get its simulation
            idx = np.random.randint(idx_start, idx_end)
            output, forcing = self.dataset[idx]

            # Get all time blocks from sampled simulation
            t_start = 0
            while len(batches) < self.batch_size:
                t_middle = t_start + self.omega
                t_end = t_start + 2*self.omega

                # check if there is enough room for, if not, break loop
                if t_end > output.shape[0]:
                    break

                # fetch X data
                X = output[t_start:t_middle,:,:]
                F = forcing[t_start:t_end,:,:]
                Y = output[t_middle:t_end,:,:]
                
                # append to batch
                batches.append([X, F, Y])

                # go to next batch
                t_start += 2*self.omega

        yield batches[:self.batch_size+1]

    def __len__(self):
        return len(self.dataset) // batch_size