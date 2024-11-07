import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, json, numpy as np, torch
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset, DataLoader

class FusionDataset(IterableDataset):
    """
    A dataset that efficiently loads chunks of a trajectory (x_t and x_{t+1}), in order. 
    The trajectories are trimmed to a random length for pushforward-training, and they are always shuffled.

    Args:
        `data_dir` (str): Directory containing npz files and optionally other directories.
        `device` (str): Torch device, either "cuda" or "cpu".
        `omega` (int): Size of trajectory chunks.
        `batch_size` (int): As used in the `DataLoader` wrapper. Defaults to 1.
    """
    def __init__(self, data_dir:str, batch_size:int=1, device:str=None, omega:int=20):
        super().__init__()
        self.get_filepaths = lambda: [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files if file.endswith(".npz")]
        self.filepaths = self.get_filepaths()
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.omega = omega
        self.trajectories = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
    def scale(self, x):
        # These parameters have been precalculated on the training set
        self.maximum = torch.tensor([9.29983400e+20, 3.95554500e+04, 2.87982800e+04, 4.26110400e+01, 1.83180800e+21, 6.19892000e+20, 4.79527254e+03, 6.71215625e+19], device=self.device)
        self.minimum = torch.tensor([ 4.5370e+18, -5.7290e+02, -7.3316e+00,  1.4339e-01,  1.8328e+15, 2.9860e+14,  0.0000e+00,  0.0000e+00], device=self.device)
        self.sigma = torch.tensor([[6.07735822e+19, 9.96008140e+03, 5.33818657e+03, 1.03287412e+01, 1.43263560e+19, 1.05223976e+19, 1.15240628e+03, 1.72603208e+19]], device=self.device)
        self.mu = torch.tensor([[7.07982534e+19, 1.70274903e+04, 8.30559879e+03, 1.24850983e+01, 3.18866547e+18, 3.36626442e+18, 5.39373277e+02, 8.06264466e+18]], device=self.device)
        # Scale
        return (x - self.minimum) / (self.maximum - self.minimum)
    def unscale(self, x):
        return x * (self.maximum - self.minimum) + self.minimum
    def get_trajectory(self, pushforward=True):
        # Load random trajectory and forcing variables
        idx = torch.randint(0, len(self.filepaths), (1,)).item()
        x, f = np.load(self.filepaths.pop(idx)).values()
        x, f = torch.tensor(x, device=self.device), torch.tensor(f, device=self.device)
        trajectory = torch.concat([x,f], dim=-1)
        # Scale
        trajectory = self.scale(trajectory)
        if pushforward:
            # Trim to random lenth for push-forward training
            length = torch.randint(low=2, high=trajectory.size(0)//self.omega+1, size=(1,)).item()*self.omega
            start_idx = torch.randint(0, trajectory.size(0)-length+1, (1,)).item()
            trajectory = trajectory[start_idx:start_idx+length]
        return list(torch.split(trajectory, self.omega))
    def fill_trajectories(self):
        # Mask for where training should inject true starting point
        self.mask = [True]*len(self.trajectories)
        # Loop if self.trajectories contains a trajectory where tplus1 is unavailable
        empty_indices = [index for (index, trajectory) in enumerate(self.trajectories) if len(trajectory)<=1]
        while empty_indices:
            # If we can load a new trajectory
            if len(self.filepaths)<=0:
                return False
            # Inject new trajectory
            idx = empty_indices.pop()
            self.trajectories[idx] = self.get_trajectory()
            self.mask[idx] = False
        return True
    def __iter__(self):
        while self.fill_trajectories():
            for i in range(self.batch_size):
                # Get (unbatched) x_t and x_tplus1 (without force) and remove the former
                x_t = self.trajectories[i].pop(0)
                x_tplus1 = self.trajectories[i][0]
                yield x_t, x_tplus1[...,:-2]
        self.filepaths = self.get_filepaths()

# Really a very simple utility function
dataloader = lambda data_dir, batch_size=16, device=None, omega=20: \
    DataLoader(FusionDataset(data_dir, batch_size, device, omega), batch_size)

def plot_1d_statistic_over_time(data, statistic_idx, title):
    """
    Plot a single statistic over time.

    Args:
        data: np.array, shape (Time, Coordinate, Statistic), data to plot
        statistic_idx: int, index of the statistic to plot
        title: str, title of the plot
    """
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

def plot_loss(filepath):
    """
    Plots the training loss with separate axes from a JSON file.
    Args:
        filepath (str): The path to the JSON file containing the training loss history, 
        consisting of Küllback-Leibler divergence, reconstruction loss, and validation loss.  
    """
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
    ax2.plot(
        train_loss_history["rec_loss"], 
        color="black", 
        label="Validation reconstruction loss",
        linestyle="-."
    );
    fig.legend();