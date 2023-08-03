import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def create_logarithmic_heatmap_plot(plot_title, x_label, y_label, heatmap, x_axis_values, y_axis_values, value_text_in_cells = True):
    """
    Creating a matplotlib plot of a heatmap. Will colors in the logarithmic scale.
    Parameters
    ----------
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    x_label: str
        The x axis label in the plot.
    y_label: str
        The y axis label in the plot.
    heatmap: np.ndarray
        A numpy matrix holding the values for the heatmap. Should be a 2 dimentional ndarray.
    x_axis_values: np.array
        A numpy array holding each value for each iteration on the x axis.
    y_axis_values: np.array
        A numpy array holding eeach value of each iteration on the y axis.
    value_text_in_cells: boolean
        Defaults to true. Determiting if each cell in the heatmap image should have it's value printed as text in the cell.
    """

    # Inner function for converting an array with numerical elements to an array with strings
    def array_elements_to_string(arr):
        new_arr = []

        for element in arr:
            new_arr.append(str(element))
        
        return new_arr
    
    labels_x = array_elements_to_string(x_axis_values)
    labels_y = array_elements_to_string(y_axis_values)

    plt.figure()

    # Create value text in each cell in the heatmap
    if value_text_in_cells:
        for i in range(len(heatmap)):
            for j in range(len(heatmap[0])):
                plt.gca().text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center", color="white")

    plt.title(plot_title)

    # Setting the correct values at the axes.
    plt.xticks(np.arange(0, len(x_axis_values)), labels_x)
    plt.yticks(np.arange(0, len(y_axis_values)), labels_y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.imshow(heatmap, norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"../figures/{plot_title}")