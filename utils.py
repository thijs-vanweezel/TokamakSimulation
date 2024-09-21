import matplotlib.pyplot as plt

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