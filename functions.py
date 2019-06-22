"""Game of life functions."""
import numpy as np
from numba import jit, int32, int8


@jit((int32[:, :], int32, int32, int32), nopython=True)
def update_neighbours(neighbours, i, j, grid_length):
    """Update neighbours with nearest neighbours of site (i, j)."""
    neighbours[0, 0] = i
    neighbours[1, 0] = i
    neighbours[2, 0] = i + 1
    neighbours[3, 0] = i - 1
    neighbours[4, 0] = i - 1
    neighbours[5, 0] = i - 1
    neighbours[6, 0] = i + 1
    neighbours[7, 0] = i + 1

    neighbours[0, 1] = j + 1
    neighbours[1, 1] = j - 1
    neighbours[2, 1] = j
    neighbours[3, 1] = j
    neighbours[4, 1] = j + 1
    neighbours[5, 1] = j - 1
    neighbours[6, 1] = j + 1
    neighbours[7, 1] = j - 1

    neighbours %= grid_length
    return()


@jit(int32(int8[:, :], int32[:, :]), nopython=True)
def count_neighbours(in_grid, neighbours):
    """Count the number of live neighbours of the site."""
    count = np.int32(0)
    for n in range(neighbours.shape[0]):
        if in_grid[neighbours[n, 0], neighbours[n, 1]] == 1:
            count += 1
    return(count)


@jit((int32, int32[:, :], int8[:, :], int8[:, :]), nopython=True)
def grid_sweep(grid_length, neighbours, in_grid, out_grid):
    """Sweep the grid once with game of life rules."""
    for i in range(grid_length):
        for j in range(grid_length):

            update_neighbours(neighbours, i, j, grid_length)
            n_count = count_neighbours(in_grid, neighbours)

            if in_grid[i, j] == 1:

                if n_count < 2:
                    out_grid[i, j] = 0
                if n_count == 2 or n_count == 3:
                    out_grid[i, j] = 1
                if n_count > 3:
                    out_grid[i, j] = 0

            if in_grid[i, j] == 0 and n_count == 3:
                out_grid[i, j] = 1
    return()


@jit((int32, int32, int32, int32[:, :], int8[:, :], int8[:, :], int8[:, :, :]),
     nopython=True)
def game_of_life(n_frames, interval, grid_length, neighbours, in_grid,
                 out_grid, solutions):
    """Simulate the game of life."""
    for sweeps in range(n_frames):

        for i in range(interval):

            grid_sweep(grid_length, neighbours, in_grid, out_grid)

        in_grid[:, :] = out_grid[:, :]
        solutions[:, :, sweeps + 1] = out_grid[:, :]
    return()


def make_movie(solutions, file_name, fps):
    """Make a movie of the game of life."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Game of Life',
                    artist='James Denholm',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig, ax = plt.subplots(1, figsize=(2, 2))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.88)

    ax.set_xticks([])
    ax.set_yticks([])

    with writer.saving(fig, file_name + '.mp4', dpi=500):

        for count in range(solutions.shape[2]):

            print("Movie progress = %i / %i" % (count + 1, solutions.shape[2]))
            a = ax.imshow(solutions[:, :, count], vmin=0, vmax=1,
                          cmap="inferno_r")
            ax.set_title("t = %.3e" % count, fontsize=10)

            writer.grab_frame()
            a.remove()
    return()
