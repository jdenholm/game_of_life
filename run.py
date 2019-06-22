#!/usr/bin/env python3
"""Code to simulate Conway's Game of Life."""
import numpy as np
import time
import functions as f


grid_length = 2 ** 6
in_grid = np.random.randint(2, size=(grid_length, grid_length), dtype=np.int8)
out_grid = in_grid.copy()
n_sweeps = 10 ** 2
n_frames = 10 ** 2
interval = np.max([1, n_sweeps // n_frames])

make_movie = True
fps = 10
save_evolution = True
file_name = "test"


solutions = np.zeros((grid_length, grid_length, n_frames + 1), dtype=np.int8)
solutions[:, :, 0] = out_grid[:, :]

neighbours = np.zeros((8, 2), dtype=np.int32)

tic = time.time()

f.game_of_life(n_frames, interval, grid_length, neighbours, in_grid,
               out_grid, solutions)

toc = time.time()

print("game of life time = %f seconds" % (toc - tic))


if save_evolution:
    np.save("%s" % file_name, solutions)
if make_movie:
    f.make_movie(solutions, file_name, fps)
