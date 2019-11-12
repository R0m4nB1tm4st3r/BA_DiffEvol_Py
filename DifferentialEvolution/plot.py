#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import matplotlib.pyplot as plt

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