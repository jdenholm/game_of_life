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


@jit(int32(int8[:, :], int32[:, :]), nopython=True)
def count_neighbours(in_grid, neighbours):
    """Count the number of live neighbours of the site."""
    count = np.int32(0)
    for n_count in range(neighbours.shape[0]):
        if in_grid[neighbours[n_count, 0], neighbours[n_count, 1]] == 1:
            count += 1
    return count


@jit((int32[:, :], int8[:, :], int8[:, :]), nopython=True)
def grid_sweep(neighbours, in_grid, out_grid):
    """Sweep the grid once with game of life rules."""
    for i in range(in_grid.shape[0]):
        for j in range(in_grid.shape[1]):

            update_neighbours(neighbours, i, j, in_grid.shape[0])
            n_count = count_neighbours(in_grid, neighbours)

            if in_grid[i, j] == 1:

                if n_count < 2:
                    out_grid[i, j] = 0
                if n_count in (2, 3):
                    out_grid[i, j] = 1
                if n_count > 3:
                    out_grid[i, j] = 0

            if in_grid[i, j] == 0 and n_count == 3:
                out_grid[i, j] = 1


@jit((int32, int32, int32[:, :], int8[:, :], int8[:, :, :]), nopython=True)
def game_of_life(n_frames, interval, neighbours, in_grid,
                 solutions):
    """Simulate the game of life."""
    out_grid = np.zeros((in_grid.shape[0], in_grid.shape[1]), dtype=np.int8)
    out_grid[:, :] = in_grid
    solutions[:, :, 0] = out_grid[:, :]
    for sweeps in range(n_frames):

        for _ in range(interval):

            grid_sweep(neighbours, in_grid, out_grid)

        in_grid[:, :] = out_grid[:, :]
        solutions[:, :, sweeps + 1] = out_grid[:, :]
    return()


def make_movie(solutions, file_name, fps):
    """Make a movie of the game of life."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ffmpeg_writer = manimation.writers['ffmpeg']
    metadata = dict(title='Game of Life',
                    artist='James Denholm',
                    comment='Movie support!')
    writer = ffmpeg_writer(fps=fps, metadata=metadata)

    fig, axis = plt.subplots(1, figsize=(2, 2))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.88)

    axis.set_xticks([])
    axis.set_yticks([])

    print("Movie progress")

    with writer.saving(fig, file_name + '.mp4', dpi=500):

        for count in range(solutions.shape[2]):

            progress_bar(count + 1, solutions.shape[2], decimals=3)
            heat_map = axis.imshow(solutions[:, :, count], vmin=0, vmax=1,
                                   cmap="YlGnBu")
            axis.set_title("t = %.3e" % count, fontsize=10)

            writer.grab_frame()
            heat_map.remove()
    return()


def progress_bar(iteration, total, prefix=' ', suffix=' ', decimals=1,
                 length=50, fill=''):
    """Call in a loop to create terminal progress bar.
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    "https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console"
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration /
                                                            float(total)))
    filled_length = int(length * iteration // total)
    prog_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, prog_bar, percent, suffix), end='\r')
    the_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, the_bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
    return()
