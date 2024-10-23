import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, json, numpy as np, torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, IterableDataset

class FusionDataset(IterableDataset):
    """
    A dataset that efficiently loads chunks of a trajectory (x_t and x_{t+1}), in order.
    Args:
    \t- `data_dir` (str): Directory containing npz files and optionally other directories.
    \t- `device` (str): Torch device, either "cuda" or "cpu".
    \t- `omega` (int): Size of trajectory chunks.
    """
    def __init__(self, data_dir:str="./data/preprocessed/", device:str=None, omega:int=20):
        super().__init__()
        self.filepaths = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.trajectory = []
        self.idx = 0
        self.omega = omega
    def reset(self):
        self.trajectory = []
    def __iter__(self):
        while self.idx<len(self.filepaths):
            # If trajectory is empty or x_{t+1} is not available
            if len(self.trajectory)<=1:
                # Load trajectory and forcing variables
                x, f = np.load(self.filepaths[self.idx]).values()
                x, f = torch.tensor(x, device=self.device), torch.tensor(f, device=self.device)
                trajectory = torch.concat([x,f], dim=-1)
                # Split by omega, and drop remainder (which is randomly either first or last chunk)
                drop_last = bool(torch.randint(0, 2, (1,)).item())
                split_sizes = (not drop_last)*[trajectory.size(0)%self.omega] \
                    + [self.omega]*(trajectory.size(0)//self.omega) + drop_last*[trajectory.size(0)%self.omega]
                trajectory = torch.split(trajectory, split_sizes)[0+(not drop_last):len(split_sizes)-drop_last]
                # Save for quick access
                self.trajectory = list(trajectory)
                # Update file index
                self.idx += 1
            # Return x_t and x_{t+1} (without forcing) and remove the former
            yield self.trajectory.pop(0), self.trajectory[0][..., :-2]
        self.idx = 0

fusion_dataset = DataLoader(FusionDataset(), batch_size=16)

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
    plt.figure(figsize=(12, 6))
    plt.imshow(data[:, :, statistic_idx].T, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Position on magnetic line")
    plt.show()

def generate(true_trajectory, forward_t, prior, decoder):
    # Generate trajectory by iteration
    gen_trajectory = [true_trajectory[0]]
    for i in range(len(true_trajectory)):
        # Generate
        h_t = forward_t(gen_trajectory[i])
        z, *_ = prior(h_t)
        gen_trajectory[i+1] = decoder(z, h_t)
        # Concatenate forcing variables
        gen_trajectory[i+1] = keras.ops.concatenate([
            gen_trajectory[i+1], 
            true_trajectory[i+1][...,-2:]
        ], axis=-1)
        # Save
        gen_trajectory.append(gen_trajectory[i+1])
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
        list(map(lambda epoch: epoch["kl_loss"], train_loss_history)), 
        color="black", 
        label="Küllback-Leibler divergence",
        linestyle="--"
    );
    plt.legend();
    ax2 = ax1.twinx();
    ax2.set_ylabel("Reconstruction loss");
    ax2.plot(
        list(map(lambda epoch: epoch["rec_loss"], train_loss_history)), 
        color="black", 
        label="Reconstruction loss",
        linestyle=":"
    );
    plt.legend();