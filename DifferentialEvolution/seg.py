#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
import math
import random as rand
import time
import diff_evol as de
import os

from tesserocr import PyTessBaseAPI, RIL
from PIL import Image
from itertools import repeat
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
#############################################################################################################################
#####################################--Functions--###########################################################################
#############################################################################################################################
def GetImage(path, mode):
    """
    read out an image from given file path
    """
    return cv2.imread(path, mode) # cv2.IMREAD_GRAYSCALE)
#############################################################################################################################
def ShowImage(image):
    """
    display image in separate window that waits for any key input to close
    """
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#############################################################################################################################
def SaveImage(image, path):
    cv2.imwrite(path, image)
#############################################################################################################################
def CalculateRawHistogram(image):
    """
    calculate histogram for image that displays the absolute numbers of gray levels

    - image: input image to calculate the histogram for
    """
    h = np.zeros(256, np.float_)
    for i in np.nditer(image):
        h[i - 1] = h[i - 1] + 1

    return h
#############################################################################################################################
def CalculateNormalizedHistogram(image):
    """
    calculate histogram for image that displays the relative numbers of gray levels

    - image: input image to calculate the histogram for
    """

    raw = CalculateRawHistogram(image)
    norm = np.zeros(256, np.float_)

    for i in range(256):
        norm[i] = raw[i] / image.size

    return norm
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
#############################################################################################################################
def SumOfGauss(param_list, classNum, g_lvls):
    """
    calculate histogram approximation as sum of gaussian probability density distribution functions

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    - classNum: number of classes
    - g_lvls: list of graylevels
    """
    return sum([param_list[i] * norm.pdf(g_lvls, loc=param_list[i + classNum], scale=param_list[i + classNum * 2]) for i in range(classNum)])
#############################################################################################################################
def CalcErrorEstimation(param_list):
    """
    calculate mean sqare error between histogram and gaussian approximation including penalty in case sum(Pi) != 1

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    """
    classNum = K
    g_lvls = graylevels
    histogram = h 
    o = 1.5

    result = sum(( SumOfGauss(param_list, classNum, g_lvls) - histogram ) ** 2) / g_lvls.size + abs(sum(param_list[:classNum]) - 1) * o
    return result
#############################################################################################################################
def TestFunction_SimpleParabolic(x):
    """
    calculate the square of input x

    - x: value for the function to be evaluated
    """
    return x ** 2
#############################################################################################################################
def InitializePopulation(Np, paramNum):
    """
    initialize population with shape given by input parameters

    - Np: 
    - paramNum: 
    """
    population = np.random.rand(Np, paramNum)
    return population
#############################################################################################################################
def CalculateThresholdValues(param_list, classNum):
    """
    calculate threshold values for image segmentation

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    - classNum: number of classes
    """
    thresholdValues = np.arange(classNum - 1, dtype=np.uint8)
    indexOrder = np.argsort(param_list[classNum:classNum * 2])

    P = [param_list[indexOrder[i]] for i in range(classNum)]
    my = np.sort(param_list[classNum:classNum * 2])
    sigma = [param_list[classNum * 2 + indexOrder[i]] for i in range(classNum)]

    for i in range(classNum - 1):
        a = sigma[i] ** 2 - sigma[i + 1] ** 2
        b = 2 * ( my[i] * ( sigma[i + 1] ** 2 ) - my[i + 1] * ( sigma[i] ** 2 ) )
        c = ( sigma[i] * my[i + 1] ) ** 2 - ( sigma[i + 1] * my[i] ) ** 2 + 2 * ( ( sigma[i] * sigma[i + 1] ) ** 2 ) * math.log(( ( sigma[i + 1] * P[i] ) / ( sigma[i] * P[i + 1] ) ))

        p = np.poly1d([a, b, c], False, "T")
        p_roots = np.roots(p)
        if p_roots.size == 1:
            thresholdValues[i] = p_roots[0]
        elif p_roots[0] == p_roots[1]:
            thresholdValues[i] = p_roots[0]
        elif p_roots[0] < 0:
            thresholdValues[i] = p_roots[1]
        elif p_roots[1] < 0:
            thresholdValues[i] = p_roots[0]
        elif p_roots[0] > 255:
            thresholdValues[i] = p_roots[1]
        elif p_roots[1] > 255:
            thresholdValues[i] = p_roots[0]
        else:
            if i > 0:
                if not(np.amin(p_roots) <= thresholdValues[i-1]):
                    thresholdValues[i] = np.amin(p_roots)
                else:
                    thresholdValues[i] = np.amax(p_roots)
            else:
                if np.amin(p_roots) <= my[i]:
                    thresholdValues[i] = np.amax(p_roots)
                else:
                    thresholdValues[i] = np.amin(p_roots)

    return thresholdValues
#############################################################################################################################
def DoImageSegmentation(image, thresholdValues):
    """
    examine segmentation on input image based on input threshold values

    - image: image to be segmentated
    - thresholdValues: limit graylevels defining the classes
    """
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #newImage = np.copy(image)
    #values = np.arange(1, dtype=np.uint8)
    #values = np.arange(3*(K-1), dtype=np.uint8).reshape(K-1, 3)
    #values[0:K-1] = rgbColorList[0:K-1]
    #values = np.append(values, thresholdValues[0:thresholdValues.size - 1])
    #condList = [image < thresholdValues[i] for i in range(thresholdValues.size)]

    #newImage = np.select(condList, values, thresholdValues[thresholdValues.size - 1])
    #color_image = np.select(condList, values, rgbColorList[(rgbColorList.shape[0]) - 1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(K-2):
                if image[i, j] <= thresholdValues[k]:
                    color_image[i, j] = rgbColorList[k]
                elif image[i, j] > thresholdValues[k] and image[i, j] <= thresholdValues[k+1]:
                    color_image[i, j] = rgbColorList[k+1]
                elif image[i, j] > thresholdValues[thresholdValues.size-1]:
                    color_image[i, j] = rgbColorList[thresholdValues.size]
                    break
    #color_image = np.array([], dtype=np.uint8)
    return color_image
#############################################################################################################################
#####################################--Globals--#############################################################################
#############################################################################################################################
K = 4
G = 750
img1 = GetImage('Y:\MobileOcr\Bilder\TestSet_DivBauteile\All_Images\good\IMG_1255_DLXFFNICKLT_c.jpg', cv2.IMREAD_GRAYSCALE)
#tesseract_testImage = Image.open("opencv-logo2.png")
#img1 = cv2.fastNlMeansDenoising(img1, None, 5, 5, 77)
#img1 = cv2.medianBlur(img1, 3)
#img1 = gaussian_filter(img1, sigma=0.73, order=2)
#ShowImage(img1)
h = CalculateNormalizedHistogram(img1)
graylevels = np.arange(256)
rgbColorList = np.array([[0, 0, 0], [255, 102, 102], [255, 255, 102], [102, 255, 102], [255, 255, 255]], dtype=np.uint8)
#############################################################################################################################
#######################################--Main--##############################################################################
#############################################################################################################################
if __name__ == "__main__":
    print('Hello World')

    t_min = np.array([])
    t_max = np.array([])
    t_min = np.append(t_min, [list(repeat(0., K)), list(repeat(0., K)), list(repeat(0., K))]) 
    t_max = np.append(t_max, [list(repeat(1., K)), list(repeat(255., K)), list(repeat(12., K))]) 

    print("Press Enter to start")

    while input() != "n":

        print("Optimizing...")
        np.random.seed(123)
        plotFigure, plotAxes = CreateSubplotGrid(2, 1, False)

        test_population = InitializePopulation(3*10*K, 3*K)
        de_handle = de.DE_Handler(0.3, 0.70, G, 3*10*K, test_population, CalcErrorEstimation, True, t_min, t_max)

        bestParams, bestValueHistory = de_handle.DE_GetBestParameters()
        bestMember = t_min + bestParams[0] * ( t_max - t_min )
        thresholdValues = CalculateThresholdValues(bestMember, K)

        newImage = DoImageSegmentation(img1, thresholdValues)

        plotFigure.set_dpi(200)

        currentTime = time.strftime("%H:%M:%S").replace(":", "_")
        currentDate = time.strftime("%d/%m/%Y").replace("/", "_")
        fileName = "IMG_1255_DLXFFNICKLT_c-" + "SEG_Test-" + currentDate + "-" + currentTime + "-" + str(G) + "-" + str(K) + ".jpg"

        plotAxes[0].plot(graylevels, h)
        plotAxes[0].plot(graylevels, SumOfGauss(bestMember, K, graylevels))
        plotAxes[0].legend(("Histogram of the original Image", "Gaussian Approximation of the Histogram", ))
        plotAxes[0].vlines(thresholdValues, 0, np.amax(h), label="Threshold Values")
        for i in range(thresholdValues.size):
            plotAxes[0].annotate("T" + str(i+1), xy=(thresholdValues[i], np.amax(h)), )
        plotAxes[0].set_xlabel("Graylevel g")
        plotAxes[0].set_ylabel("n_Pixel_relative")
        plotAxes[0].set_title("Mean Square Error: " + str(bestParams[1]))
        plotAxes[1] = plt.imshow(newImage, cmap='gray')
        plotAxes[1].axes.set_title("Result of Image Segmentation")
        plt.tight_layout()
        plt.savefig("IMG_1255_DLXFFNICKLT_c-" + "SEG_Plot-" + currentDate + "-" + currentTime + "-" + str(G) + "-" + str(K) + ".jpg", dpi=200)
        plt.show()

        SaveImage(newImage, fileName)
        seg_image = Image.open(fileName)

        with PyTessBaseAPI() as api:
            api.SetImage(seg_image)
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)

            print("Found {} textline image components.".format(len(boxes)))
            for i, (im, box, _, _) in enumerate(boxes):
                api.SetRectangle(box["x"], box["y"], box["w"], box["h"])
                ocrResult = api.GetUTF8Text()
                conf = api.MeanTextConf()
                print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                  "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))

        print("Do you wish to make another plot? (y/n)")
