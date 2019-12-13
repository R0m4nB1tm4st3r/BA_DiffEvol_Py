#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#############################################################################################################################
#####################################--Functions--###########################################################################
#############################################################################################################################
def Plot_SingleGraph(dataHorizontal, dataVertical, horizontalLabel, verticalLabel, title, graphLabel):
    """
    create a figure with a single graph containing the given data

    - dataHorizontal: data for the horizontal axis of the graph
    - dataVertical: data for the vertical axis of the graph
    - horizontalLabel: label for the horizontal axis
    - verticalLabel: label for the vertical axes
    - title: title of the graph
    - graphLabel: legend entry for the graph
    """

    diagram, ax = plt.subplots()
    ax.plot(dataHorizontal, dataVertical, label=graphLabel)

    plt.xlabel(horizontalLabel)
    plt.ylabel(verticalLabel)
    plt.title(title)
#############################################################################################################################
def PlotInSameFigure(dataHorizontal, dataVerticalArray, graphLabelArray):
    """
    plot several graphs in one figure

    - dataHorizontal: data for the horizontal axis of the graph
    - dataVerticalArray: data for more than one graph for the vertical axis
    - graphLabelArray: legend entries for each graph
    """

    for i in range(len(dataVerticalArray)):
        plt.plot(dataHorizontal, dataVerticalArray[i], label=graphLabelArray[i])

#############################################################################################################################
def PlotWithSubPlots(dataHorizontal, dataVerticalArray, graphLabelArray, yDim, xDim):
    """
    plot several graphs in one figure creating subplots for each graph

    - dataHorizontal: data for the horizontal axis of the graph
    - dataVerticalArray: data for more than one graph for the vertical axis
    - graphLabelArray: legend entries for each graph
    - yDim: number of lines
    - xDim: number of columns
    """

    fig, ax_list = plt.subplots(xDim, yDim)
    for i in range(len(dataVerticalArray)):
        ax_list[i].plot(dataHorizontal, dataVerticalArray[i])
        ax_list[i].legend(( graphLabelArray[i], ))
#############################################################################################################################
def CreateSubplotGrid(rows, columns, shareX):
    """
    creates a subplot grid object 

    - rows: number of rows
    - columns: number of columns
    - shareX: determines whether all subplots share the x-axis or not
    """
    fig, ax = plt.subplots(rows, columns, shareX)
    return fig, ax
##############################################################################################################################
def PlotExample_3D():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='b')

    plt.show()
##########################################################################################################################
def PlotExample_3D_2():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
##########################################################################################################################
def LinePlot3D(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the line.
    surf = ax.plot(X, Y, Z)

    plt.show()