#!/usr/bin/env python3
"""Code to write movies of the evolution of the Potts model."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import time
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Zero-temperature coarsening in the Potts ferromagnet',
                artist='James Denholm', comment='Movie support!')
writer = FFMpegWriter(fps=5, metadata=metadata)

fig, ax = plt.subplots(1, figsize=(2, 2))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.88)

ax.set_xticks([])
ax.set_yticks([])

tic = time.time()

solutions = np.load("test.npy")
file_name = "test"

with writer.saving(fig, file_name + '.mp4', dpi=500):

    for count in range(solutions.shape[2]):

        print("%i / %i" % (count + 1, solutions.shape[2]))
        a = ax.imshow(solutions[:, :, count], vmin=0, vmax=1, cmap="YlGnBu")

        writer.grab_frame()
        a.remove()

toc = time.time()

print(toc - tic)
