import os
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch

os.chdir("../results")

def safe_load_pickle(path):
    # Try loading in chunks to handle large integers
    with open(path, 'rb') as f:
        # First try with torch's built-in loading
        try:
            # Explicitly specify CPU and use default pickle protocol
            return torch.load(
                f, 
                map_location=lambda storage, loc: storage.cpu(),
            )
        except OverflowError:
            # If that fails, try reopening the file and using pickle directly
            f.seek(0)
            import pickle
            return pickle.load(f)
        

def estimate_derivative(x):
    """
    Estimate derivative of 2D array using forward differences along spatial dimension.
    Input shape: (time_steps, spatial_points)
    Returns array of same shape with padding of zeros at the beginning of spatial dimension
    
    Parameters:
    x: array of shape (time_steps, spatial_points)
    
    Returns:
    array of shape (time_steps, spatial_points) with spatial derivatives
    """
    time_steps, spatial_points = x.shape
    
    # Calculate differences along spatial dimension for all time steps
    derivative = x[:, 1:] - x[:, :-1]  # Shape: (time_steps, spatial_points-1)
    
    # Pad with zeros at the beginning of spatial dimension for each time step
    padded_derivative = np.zeros_like(x)  # Shape: (time_steps, spatial_points)
    padded_derivative[:, 1:] = derivative
    
    return padded_derivative

def PDE_plasma_density(n_pred, v_pred, n_true, v_true, B):
    """
        n: plasma density
        v: ion speed parallel to magnetic field
        B: strength of magnetic field
    """
    pred = estimate_derivative(n_pred * v_pred / B)
    true = estimate_derivative(n_true * v_true / B)
    return B**2 * (true - pred)**2

def PDE_plasma_momentum(n_pred, v_pred, temp_pred, n_true, v_true, temp_true, B, m):
    """
        n: plasma density
        v: ion speed parallel to magnetic field
        T: plasma temperature
        B: strength of magnetic field
    """
    pred = - B * estimate_derivative(n_pred * m * v_pred**2 / B) - estimate_derivative(2 * n_pred * np.e * temp_pred)
    true = - B * estimate_derivative(n_true * m * v_true**2 / B) - estimate_derivative(2 * n_true * np.e * temp_true)
    return (true - pred)**2

def PDE_plasma_energy(n_pred, v_pred, temp_pred, n_true, v_true, temp_true, B):
    """
        n: plasma density
        v: ion speed parallel to magnetic field
        T: plasma temperature
        B: strength of magnetic field
    """
    q = lambda n, v, T: 5 * n * np.e * T * v - np.exp( 3 * np.log(10) + np.log(T)*(5/2) ) * estimate_derivative(T)
    pred = -B * estimate_derivative(q(n_pred, v_pred, temp_pred) / B) + v_pred * estimate_derivative(2 * n_pred * temp_pred)
    true = -B * estimate_derivative(q(n_true, v_true, temp_true) / B) + v_true * estimate_derivative(2 * n_true * temp_true)
    return (true - pred)**2
 
def PDE_atom_momentum(n_a_pred, v_a_pred, n_a_true, v_a_true, m, T_a):
    """
        n_a: neutral atom density
        v_a: neutral atom speed parallel to magnetic field
        T_a: neutral temperature
    """
    pred = -1 * estimate_derivative(n_a_pred * m * v_a_pred**2) - estimate_derivative(np.e * n_a_pred * T_a)
    true = -1 * estimate_derivative(n_a_true * m * v_a_true**2) - estimate_derivative(np.e * n_a_true * T_a)
    return (true - pred)**2

def scale_x(x):
    # These parameters have been precalculated on the training set
    sigma = np.array([6.07735822e+19, 9.96008140e+03, 5.33818657e+03, 1.03287412e+01, 1.43263560e+19, 1.05223976e+19])
    mu = np.array([7.07982534e+19, 1.70274903e+04, 8.30559879e+03, 1.24850983e+01, 3.18866547e+18, 3.36626442e+18])

    # Reshape mu and sigma to match broadcasting dimensions [1, 1, 6]
    mu = mu.reshape(1, 1, -1)
    sigma = sigma.reshape(1, 1, -1)
    
    return (x - mu) / sigma

def calculate_PDE_loss_of_trajectory(x_pred, x_true, pde_vars):
    """Calculates the PDE losses for a trajectory loss
    """
    # scale 
    x_true = scale_x(x_true)

    # trim prediction to maintain consistency in timesteps
    x_pred = x_pred[:x_true.shape[0], ...]
    x_pred = scale_x(x_pred)
    
    # DIV1D values
    T_a         = pde_vars[..., 0]
    m           = pde_vars[..., 1]
    B           = pde_vars[..., 2]

    # predicted values
    n_pred      = x_pred[..., 0]
    v_pred      = x_pred[..., 1]
    temp_pred   = x_pred[..., 3]
    n_a_pred    = x_pred[..., 4]
    v_a_pred    = x_pred[..., 2]

    # predicted values
    n_true      = x_true[..., 0]
    v_true      = x_true[..., 1]
    temp_true   = x_true[..., 3]
    n_a_true    = x_pred[..., 4]
    v_a_true    = x_pred[..., 2]

    plasma_density = PDE_plasma_density(n_pred, v_pred, n_true, v_true, B)
    plasma_momentum = PDE_plasma_momentum(n_pred, v_pred, temp_pred, n_true, v_true, temp_true, B, m)
    plasma_energy = PDE_plasma_energy(n_pred, v_pred, temp_pred, n_true, v_true, temp_true, B)
    atom_momentum = PDE_atom_momentum(n_a_pred, v_a_pred, n_a_true, v_a_true, m, T_a)

    return plasma_density, plasma_momentum, plasma_energy, atom_momentum 

from typing import List, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_latent_space(
    latent_samples: Union[np.ndarray, List[np.ndarray]],
    method: str = 'pca',
    n_neighbors: int = 15,  # Only used for t-SNE
    perplexity: float = 30.0,  # Only used for t-SNE
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce dimensionality of trajectory latent samples to 2D using PCA or t-SNE.
    
    Parameters:
    -----------
    latent_samples : np.ndarray
        The latent samples to reduce. Shape should be (n_trajectories, *feature_dims)
    method : str, optional (default='pca')
        Reduction method to use: 'pca' or 'tsne'
    n_neighbors : int, optional (default=15)
        Number of neighbors (only used for t-SNE)
    perplexity : float, optional (default=30.0)
        Perplexity parameter for t-SNE
    random_state : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        2D reduced representations of shape (n_trajectories, 2)
    """
    # Convert list to numpy array if necessary
    if isinstance(latent_samples, list):
        latent_samples = np.array(latent_samples)
    
    # Flatten the feature dimensions
    flattened_samples = latent_samples.reshape(latent_samples.shape[0], -1)
    
    # Choose reduction method
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    # Fit and transform the data
    reduced_samples = reducer.fit_transform(flattened_samples)
    
    return reduced_samples

import matplotlib.pyplot as plt 

def save_plot(filename, data, title):
    """
    Plot a single statistic over time.

    Args:
        data: np.array, shape (Time, Coordinate, Statistic), data to plot
        statistic_idx: int, index of the statistic to plot
        title: str, title of the plot
    """
    fig = plt.figure(figsize=(12, 6))
    plt.ioff()
    plt.imshow(data.T, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Position on magnetic line")
    plt.savefig(filename)
    plt.close(fig)

def save_plots(file_dir, true, pred, plasma_density, plasma_momentum, plasma_energy, atom_density):
    
    vars = [
        "plasma_density", "plasma_velocity", "plasma_temperature",
        "atom_density", "atom_velocity", "molecule_density"
    ]

    for idx, name in enumerate(vars):
        save_plot(file_dir + f"true_trajectory_{name}.png", true[..., idx], "Ground truth")
        save_plot(file_dir + f"predicted_trajectory{name}.png", pred[..., idx], "Predicted trajectory")


    save_plot(file_dir + "PDE_plasma_density.png", plasma_density, "Plasma density PDE loss")
    save_plot(file_dir + "PDE_plasma_momentum.png", plasma_momentum, "Plasma momentum PDE loss")
    save_plot(file_dir + "PDE_plasma_energy.png", plasma_energy, "Plasma energy PDE loss")
    save_plot(file_dir + "PDE_atom_density.png", atom_density, "Atom density PDE loss")


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_latent_losses(data_points: List[Tuple[float, float, float, float, float, float]], 
                      loss_names: List[str] = ['Loss 1', 'Loss 2', 'Loss 3', 'Loss 4'],
                      n_bins: int = 50,
                      cmap: str = 'viridis',
                      interpolation: str = 'gaussian'):
    """
    Create heatmaps of losses over latent space coordinates.
    
    Parameters:
    -----------
    data_points : List[Tuple]
        List of tuples containing (x, y, loss1, loss2, loss3, loss4)
    loss_names : List[str]
        Names of the losses for plot titles
    n_bins : int
        Number of bins for the 2D histogram
    cmap : str
        Colormap to use for the heatmaps
    interpolation : str
        Interpolation method for imshow
    """
    # Convert list of tuples to numpy array
    data = np.array(data_points)
    x, y = data[:, 0], data[:, 1]
    losses = data[:, 2:]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Loss Distribution over Latent Space', fontsize=16, y=0.95)
    
    # Calculate common x and y bin edges for all plots
    x_bins = np.linspace(x.min(), x.max(), n_bins)
    y_bins = np.linspace(y.min(), y.max(), n_bins)
    
    # Create heatmaps for each loss
    for idx, (ax, loss, name) in enumerate(zip(axes.flat, losses.T, loss_names)):
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], 
                                            weights=loss,
                                            density=False)
        
        # Normalize by count to get average loss per bin
        counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        # Avoid division by zero
        mask = counts > 0
        hist[mask] = hist[mask] / counts[mask]
        
        # Plot
        im = ax.imshow(hist.T, origin='lower', aspect='auto',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      interpolation=interpolation,
                      cmap=cmap)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="PDE loss")
        
        # Set labels and title
        ax.set_xlabel('reduced dimension 1')
        ax.set_ylabel('reduced dimension 2')
        ax.set_title(name)
        
        # Add scatter plot of actual points with low opacity
        ax.scatter(x, y, c='white', alpha=0.1, s=1)

    plt.savefig("latent_analysis.png")


indexes = [
    168,
    223,
    190,
    107,
    99,
    94,
    191,
    83,
    138,
    89,
    133,
    268,
    82,
    167,
    30,
    152,
    26,
    199,
    162,
    253
]
trajectory_files = [f"data/files/data/test/div1d_output_{idx}.npz" for idx in indexes]


# calculate PDE losses
latent_samples = []
PDES = []

for idx in range(20):
    # load data
    preds = safe_load_pickle(f"data/files/generated_trajectories{idx}.pkl")
    zs = safe_load_pickle(f"data/files/generated_zs{idx}.pkl")
    file = trajectory_files[idx]

    print("trajectory: ", file)

    # create directory to save images
    file_name = ".".join(file.split(".")[:-1]).split("/")[-1]
    file_dir = f"plots/{file_name}/"

    # fetch ground truth and pde vars
    x_true, _, pde_vars = np.load(file).values()

    # for each sample for a given trajectory ground truth
    for idx in range(preds.shape[0]):

        print("\t sample: ", idx)

        sample_traj = preds[idx]
        sample_z = zs[idx]

        sample_traj = sample_traj.detach().numpy()
        
        # calculate PDE losses
        plasma_density, plasma_momentum, plasma_energy, atom_density = calculate_PDE_loss_of_trajectory(sample_traj, x_true, pde_vars)

        # plot curves
        # file_name = file_dir + f"sample_{idx}/"
        # os.makedirs(file_name, exist_ok=True)
        # save_plots(file_name, x_true, sample_traj, plasma_density, plasma_momentum, plasma_energy, atom_density)
           
        # save results
        latent_samples.append(sample_z)
        latent_time_size = sample_z.shape[0]
        PDES.append(
            torch.tensor(np.stack(
                # stack four numpy arrays of size (timesteps, ) into torch tensor with size (timesteps, 4)
                # makes sure it matches dimension of latent space for time
                [np.sum(plasma_density, axis=1)[:latent_time_size], 
                 np.sum(plasma_momentum, axis=1)[:latent_time_size], 
                 np.sum(plasma_energy, axis=1)[:latent_time_size], 
                 np.sum(atom_density, axis=1)[:latent_time_size]], 
                axis=1))
            )
        
# dimensionality reduction

# preprocess
latent_samples_comb = torch.cat(latent_samples, dim=0)
PDES_comb = torch.cat(PDES, dim=0).numpy()

print("starting dim reduction")
# reduce dims
latent_space_2D = reduce_latent_space(latent_samples_comb, method="tsne")
print("dim reduction finished")

# combine with pde losses
latent_analysis = np.concatenate([latent_space_2D, PDES_comb], axis=1)

# plot
plot_latent_losses(
    latent_analysis,
    ["Plasma Density", "Plasma Momentum", "Plasma Energy", "Atom Density"]
)