# util file for plotting
import matplotlib.pyplot as plt


def plot_corr_matrix(df, filename="corr_plot.png"):
    """
    Plot a correlation matrix of the columns in the DataFrame df
    """
    corr = df.corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=-45)
    plt.yticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.colorbar()

    # add value to square on plot
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(i, j, round(corr.iloc[i, j], 1), ha="center", va="center", size=6)

    plt.savefig("corr_plot.png")
    plt.close()


def plot_scatter(df, x, y, xlabel, ylabel, title, filename="scatter_plot.png"):
    """
    Plot a scatter plot of the columns x and y in the DataFrame df
    """
    plt.scatter(df[x], df[y], c=df.index.month)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    # color by month

    plt.savefig(filename)
    plt.close()


def plot_3d_scatter(
    df,
    x,
    y,
    z,
    xlabel,
    ylabel,
    zlabel,
    title,
    filename="3d_scatter_plot.png",
    color_by=None,
    flip_x_axes=False,
    flip_y_axes=False,
    flip_z_axes=False,
):
    """
    Plot a 3D scatter plot of the columns x, y, and z in the DataFrame df
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if color_by:
        ax.scatter(df[x], df[y], df[z], c=df[color_by])
    else:
        ax.scatter(df[x], df[y], df[z], c=df.index.month)
    if flip_x_axes:
        ax.invert_xaxis()
    if flip_y_axes:
        ax.invert_yaxis()
    if flip_z_axes:
        ax.invert_zaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_3d_scatter_no_index(
    df,
    x,
    y,
    z,
    xlabel,
    ylabel,
    zlabel,
    title,
    filename="3d_scatter_plot.png",
    color_by=None,
    flip_x_axes=False,
    flip_y_axes=False,
    flip_z_axes=False,
):
    """
    Plot a 3D scatter plot of the columns x, y, and z in the DataFrame df
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if color_by:
        ax.scatter(df[x], df[y], df[z], c=df[color_by])
    else:
        ax.scatter(df[x], df[y], df[z])
    if flip_x_axes:
        ax.invert_xaxis()
    if flip_y_axes:
        ax.invert_yaxis()
    if flip_z_axes:
        ax.invert_zaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_line(df, x, y, xlabel, ylabel, title, filename="line_plot.png"):
    """
    Plot a line plot of the columns x and y in the DataFrame df
    """
    plt.plot(df[x], df[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
