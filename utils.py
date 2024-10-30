import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, json, numpy as np, torch
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset, Sampler

class FusionDataset(IterableDataset):
    """
    A dataset that efficiently loads chunks of a trajectory (x_t and x_{t+1}), in order. 
    The trajectories are trimmed to a random length for pushforward-training, and they are always shuffled.
    Args:
    \t- `data_dir` (str): Directory containing npz files and optionally other directories.
    \t- `device` (str): Torch device, either "cuda" or "cpu".
    \t- `omega` (int): Size of trajectory chunks.
    """
    def __init__(self, data_dir:str="./data/preprocessed/", device:str=None, omega:int=20):
        super().__init__()
        self.get_filepaths = lambda: [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
        self.filepaths = self.get_filepaths()
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.omega = omega
        self.std = self.mean = None
    def scale(self, x):
        # Calculate parameters once
        if self.std is None:
            self.std, self.mean = torch.std_mean(x)
        # Scale
        return (x - self.mean) / self.std
    def unscale(self, x):
        return x * self.std + self.mean
    def __iter__(self):
        while len(self.filepaths)>0:
            # Load random trajectory and forcing variables
            idx = torch.randint(0, len(self.filepaths), (1,)).item()
            x, f = np.load(self.filepaths.pop(idx)).values()
            x, f = torch.tensor(x, device=self.device), torch.tensor(f, device=self.device)
            trajectory = torch.concat([x,f], dim=-1)
            # Scale
            trajectory = self.scale(trajectory)
            # Trim to random lenth for push-forward training
            length = torch.randint(low=2, high=trajectory.size(0)//self.omega+1, size=(1,)).item()*self.omega
            start_idx = torch.randint(0, trajectory.size(0)-length+1, (1,)).item()
            trajectory = trajectory[start_idx:start_idx+length]
            # Return entire trajectory
            yield torch.split(trajectory, self.omega)
        self.filepaths = self.get_filepaths()

class TrajectoryPreservingSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = iter(dataset)
        self.batch_size = batch_size
        self.storage = [[] for _ in range(batch_size)]
        self.empty_indices = lambda: [index for index, trajectory in enumerate(self.storage) if len(trajectory)<=1]

    def fill_storage(self):
        # Mask for where training should inject true starting point
        self.mask = [True]*self.batch_size
        # While storage contains trajectories with unavailable tplus1
        empty_indices = self.empty_indices()
        while empty_indices:
            try:
                # Fill new trajectory
                idx = empty_indices.pop(0)
                self.storage[idx] = list(next(self.dataset))
                self.mask[idx] = False
        # Signal that dataset is empty or not
            except StopIteration:
                return False
        return True
    
    def __iter__(self):
        # while dataset is not empty
        while self.fill_storage():
            # Get x_t and x_tplus1 (without force) and remove the former
            batch_t = [self.storage[i].pop(0) for i in range(self.batch_size)]
            batch_tplus1 = [self.storage[i][0] for i in range(self.batch_size)]
            yield torch.stack(batch_t), torch.stack(batch_tplus1)[...,:-2]

def plot_1d_statistic_over_time(data, statistic_idx, title):
    """
    Plot a single statistic over time.

    Args:
        data: np.array, shape (Time, Coordinate, Statistic), data to plot
        statistic_idx: int, index of the statistic to plot
        title: str, title of the plot
    """
    print(data.shape)
    print(data[:, :, statistic_idx].shape)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(data[:, :, statistic_idx].T, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Position on magnetic line")
    plt.show()
    return fig

def generate(true_trajectory, forward_t, prior, decoder):
    """
    Generate a trajectory using the starting point and forcing variables provided.
    Args:
        true_trajectory: list of tensors of shape [omega, 500, 8]
    """
    # Generate trajectory by iteration
    gen_trajectory = [true_trajectory[0][...,:-2]]
    for i in range(len(true_trajectory)):
        # Concatenate forcing variables
        x_t_hat = keras.ops.concatenate([
            gen_trajectory[i], 
            true_trajectory[i][...,-2:]
        ], axis=-1)
        # Generate
        h_t = forward_t(keras.ops.expand_dims(x_t_hat, axis=0))
        z, *_ = prior(h_t)
        x_tplus1_hat = decoder(z, h_t)[0]
        # Save
        gen_trajectory.append(x_tplus1_hat)
    return gen_trajectory

def plot_loss(filepath="./results/basic0/history.json"):
    # Load training history
    with open(filepath, "r") as f:
        train_loss_history = json.load(f)
    fig, ax1 = plt.subplots();
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("Küllback-Leibler divergence");
    ax1.set_title("Training Loss");
    ax1.plot(
        train_loss_history["kl_loss"], 
        color="black", 
        label="Küllback-Leibler divergence",
        linestyle="--"
    );
    fig.legend();
    ax2 = ax1.twinx();
    ax2.set_ylabel("Reconstruction loss");
    ax2.plot(
        train_loss_history["rec_loss"], 
        color="black", 
        label="Reconstruction loss",
        linestyle=":"
    );
    fig.legend();