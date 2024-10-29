import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, json, numpy as np, torch
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset

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
        self.reset()
    def reset(self):
        """
        Start from the beginning of a random trajectory with random length.
        """
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
        # Save for quick access
        self.trajectory = trajectory
        self.i = 0
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
            # Reset if trajectory is empty
            if ((self.i+2)*self.omega)>self.trajectory.size(0):
                self.reset() # HOW TO LET TRAINER KNOW THAT A NEW TRAJECTORY STARTS?
            # Return x_t and x_{t+1} (without forcing)
            x_t = self.trajectory[self.i*self.omega:(self.i+1)*self.omega]
            x_tplus1 = self.trajectory[(self.i+1)*self.omega:(self.i+2)*self.omega,...,:-2]
            self.i += 1
            yield x_t, x_tplus1
        self.filepaths = self.get_filepaths()

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