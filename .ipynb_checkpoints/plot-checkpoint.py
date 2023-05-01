import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(xs,ys):
    plt.figure()
    plt.grid()
    plt.plot(xs,ys, 'b*-')
