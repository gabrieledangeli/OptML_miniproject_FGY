import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(xs, ys, method):
    """
    Plot the trajectory of a method.

    Parameters:
    xs (list): The x-coordinates of the trajectory.
    ys (list): The y-coordinates of the trajectory.
    method (str): The name of the method used to generate the trajectory.

    Returns:
    None
    """
    plt.figure()
    plt.grid()
    plt.plot(xs, ys, 'b--', linewidth=1, label=method)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title("Trajectory of {}".format(method))
    plt.legend()
    # Add markers for the starting and end points
    plt.scatter(xs[0], ys[0], c='r', marker='o', label='Start')
    plt.scatter(xs[-1], ys[-1], c='g', marker='o', label='End')
    plt.legend()

    plt.show()
    
    
def plot_trajectories(methods, xs_list, ys_list, line_styles):
    """
    Plot the trajectories of multiple methods.

    Parameters:
    methods (list): The names of the methods used to generate the trajectories.
    xs_list (list): A list of x-coordinates for each trajectory.
    ys_list (list): A list of y-coordinates for each trajectory.
    line_styles (list): A list of line styles for each method.

    Returns:
    None
    """
    plt.figure()
    plt.grid()

    # Plot the trajectories for each method
    for i in range(len(methods)):
        plt.plot(xs_list[i], ys_list[i], line_styles[i], linewidth=1, label=methods[i])

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title("Trajectories")
    plt.legend()

    # Add markers for the starting and end points
    starts_x = [xs[0] for xs in xs_list]
    starts_y = [ys[0] for ys in ys_list]
    ends_x = [xs[-1] for xs in xs_list]
    ends_y = [ys[-1] for ys in ys_list]

    plt.scatter(starts_x, starts_y, c='r', marker='o', label='Start')
    plt.scatter(ends_x, ends_y, c='g', marker='o', label='End')
    plt.legend()

    plt.show()