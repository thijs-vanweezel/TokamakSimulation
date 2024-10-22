import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, json
from matplotlib import pyplot as plt

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