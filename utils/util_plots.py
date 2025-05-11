import os
import sys
import logging
import json
import datetime
import random
import re
import math
import itertools
import functools
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation


def plot_diffusion_steps_gif(steps,batch, save_path="diffusion_process.gif"):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    
    #random number generator
    random_index = np.random.randint(0, batch.shape[0])
    random_channel = np.random.randint(0, batch.shape[1])
    ax.set_ylim(-0.05 + batch[random_index, random_channel].min(), batch[random_index, random_channel].max() + 0.2)
    ax.set_xlim(0, len(steps[0][0, 0]))  # Assuming each step has the same length

    # Plot the static background line from batch in shadowy blue

    ax.plot(np.arange(len(batch[random_index,random_channel])), batch[random_index,random_channel], color="blue", alpha=0.6, lw=2)
    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(np.arange(len(steps[i][0, 0])), steps[i][random_index, random_channel])
        ax.set_title(f"Sample: {random_index}, Channel: {random_channel} , Step: {i}")
        return (line,)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(steps), blit=True)
    ani.save(save_path, writer=PillowWriter(fps=10))
    plt.close()

def load_files(base_path, groups, file_types):
    """
    Load files dynamically for multiple groups and file types.
    
    Args:
        base_path (str): The base path to the files.
        groups (list of int): The groups to load files for (e.g., [0, 1, 2, 3]).
        file_types (list of str): The file types to load (e.g., ['imputation', 'mask', 'original']).
    
    Returns:
        dict: A nested dictionary with loaded data for each group and file type.
    """
    data = {}

    for group in groups:
        group_data = {}
        for file_type in file_types:
            file_path = f"{base_path}/{file_type}{group}.npy"
            group_data[file_type] = np.load(file_path)
        data[group] = group_data

    return data