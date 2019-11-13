import numpy as np
import math
import cv2

from scipy.stats import norm

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
def SumOfGauss(param_list, classNum, g_lvls):
    """
    calculate histogram approximation as sum of gaussian probability density distribution functions

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    - classNum: number of classes
    - g_lvls: list of graylevels
    """
    return sum([param_list[i] * norm.pdf(g_lvls, loc=param_list[i + classNum], \
        scale=param_list[i + classNum * 2]) \
        for i in range(classNum)])
#############################################################################################################################
def CalcErrorEstimation(param_list, classNum, g_lvls, histogram, o):
    """
    calculate mean sqare error between histogram and gaussian approximation including penalty in case sum(Pi) != 1

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    """
    return (sum( \
        ( SumOfGauss(param_list, classNum, g_lvls) - histogram ) ** 2) / g_lvls.size) + \
        (abs(sum(param_list[:classNum]) - 1) * o)
    #return result
#############################################################################################################################
def CalculateThresholdValues(param_list, classNum):
    """
    calculate threshold values for image segmentation

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    - classNum: number of classes
    """
    thresholdValues = np.arange(classNum - 1, dtype=np.uint8)
    #numRow = sp.math.factorial(classNum-1)
    #numCol = classNum-1
    #thresholdValues = np.arange(numCol*numRow).reshape(numRow, numCol)
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
        p.dtype = np.uint8
        r1 = np.real(p_roots[0])
        if p_roots.size == 1:
            thresholdValues[i] = r1
        else:
            r2 = np.real(p_roots[1])
            if r1 == r2:
                thresholdValues[i] = r1
            elif r1 < 0:
                thresholdValues[i] = r2
            elif r2 < 0:
                thresholdValues[i] = r1
            elif r1 > 255:
                thresholdValues[i] = r2
            elif r2 > 255:
                thresholdValues[i] = r1
            else:
                r1 = np.amin(p_roots)
                r2 = np.amax(p_roots)
                if i > 0:
                    if r1 >= thresholdValues[i-1]:
                        thresholdValues[i] = r1
                    else:
                        thresholdValues[i] = r2
                else:
                    if (r1 >= my[i]) and (r1 < my[i+1]):
                        thresholdValues[i] = r1
                    else:
                        thresholdValues[i] = r2

    return thresholdValues
#############################################################################################################################
def DoImageSegmentation(image, thresholdValues, K, rgbColorList):
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
            for k in range(K-2, -1, -1):
                if image[i, j] > thresholdValues[k]:
                    color_image[i, j] = rgbColorList[k+1]
                    break
                else:
                    color_image[i, j] = rgbColorList[0]
    #color_image = np.array([], dtype=np.uint8)
    return color_image
#############################################################################################################################
#def GetSegmentFromMean(param_list, numOfSegments):
#############################################################################################################################
def TestFunction_SimpleParabolic(x):
    """
    calculate the square of input x

    - x: value for the function to be evaluated
    """
    return x ** 2
