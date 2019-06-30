#!/usr/bin/env python3
"""Code to simulate Conway's Game of Life."""
import time
import numpy as np
import functions as f


def main(grid_length, n_sweeps, n_frames, make_movie, save_data):
    """Run the game of life."""
    in_grid = np.random.randint(2, size=(grid_length, grid_length),
                                dtype=np.int8)
    out_grid = in_grid.copy()
    interval = np.max([1, n_sweeps // n_frames])

    fps = 10
    file_name = "game_of_life"

    solutions = np.zeros((grid_length, grid_length, n_frames + 1),
                         dtype=np.int8)
    solutions[:, :, 0] = out_grid[:, :]

    neighbours = np.zeros((8, 2), dtype=np.int32)

    tic = time.time()

    f.game_of_life(n_frames, interval, grid_length, neighbours, in_grid,
                   out_grid, solutions)

    toc = time.time()

    print("game of life time = %f seconds" % (toc - tic))

    if save_data:
        np.save("%s" % file_name, solutions)
    if make_movie:
        f.make_movie(solutions, file_name, fps)


main(2 ** 6, 10 ** 3, 10 ** 3, True, True)
