import numpy as np
import math
import cv2

from scipy.stats import norm

#############################################################################################################################
def OF1_CalculateRawHistogram(image):
    """
    calculate histogram for image that displays the absolute numbers of gray levels

    - image: input image to calculate the histogram for
    """
    h = np.zeros(256, np.float_)
    for i in np.nditer(image):
        h[i - 1] = h[i - 1] + 1

    return h
#############################################################################################################################
def OF1_CalculateNormalizedHistogram(image):
    """
    calculate histogram for image that displays the relative numbers of gray levels

    - image: input image to calculate the histogram for
    """

    raw = OF1_CalculateRawHistogram(image)
    norm = np.zeros(256, np.float_)

    for i in range(256):
        norm[i] = raw[i] / image.size

    return norm
#############################################################################################################################
def OF1_SumOfGauss(param_list, classNum, g_lvls):
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
#def OF1_CalcErrorEstimation(param_list, classNum, g_lvls, histogram, o):
def OF1_CalcErrorEstimation(param_list, args):
    """
    calculate mean sqare error between histogram and gaussian approximation including penalty in case sum(Pi) != 1

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    """
    #return (sum( \
        #( OF1_SumOfGauss(param_list, classNum, g_lvls) - histogram ) ** 2) / g_lvls.size) + \
        #(abs(sum(param_list[:classNum]) - 1) * o)
    return (sum( \
        ( OF1_SumOfGauss(param_list, args[0], args[1]) - args[2] ) ** 2) / args[1].size) + \
        (abs(sum(param_list[:args[0]]) - 1) * args[3])
#############################################################################################################################
def OF1_CalculateThresholdValues(param_list, classNum):
    """
    calculate threshold values for image segmentation

    - param_list: list of gaussian parameters (P1, P2, ... Pk, my1, my2, ... myk, sigma1, sigma2, ... sigmak - k is the number of classes) 
    - classNum: number of classes
    """
    thresholdValues = [(-1., -1.) for _ in range(classNum-1)] # np.arange(classNum - 1)
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

        if p_roots.size == 1:
            thresholdValues[i] = (np.real(p_roots[0]), -1)
        else:
            r1 = np.real(p_roots[0])
            r2 = np.real(p_roots[1])
            if (r1 == r2) or (r2 < 0.) or (r2 > 255.):
                thresholdValues[i] = (r1, -1)
            elif (r1 < 0) or (r1 > 255):
                thresholdValues[i] = (r2, -1)
            else:
                thresholdValues[i] = (r1, r2)
                #r1 = np.amin(p_roots)
                #r2 = np.amax(p_roots)
                #if i > 0:
                    #if r1 >= thresholdValues[i-1]:
                        #thresholdValues[i] = r1
                    #else:
                        #thresholdValues[i] = r2
                #else:
                    #if (r1 >= my[i]) and (r1 < my[i+1]):
                        #thresholdValues[i] = r1
                    #else:
                        #thresholdValues[i] = r2

    return thresholdValues
#############################################################################################################################
def OF1_DoImageSegmentation(image, thresholdValues, K, rgbColorList):
    """
    examine segmentation on input image based on input threshold values

    - image: image to be segmentated
    - thresholdValues: limit graylevels defining the classes
    """
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    it = np.nditer(image, flags=["multi_index"])

    while not it.finished:
        for t in range(thresholdValues.size-1, -1, -1):
            if it[0] > thresholdValues[t]:
                color_image[it.multi_index] = rgbColorList[t+1]
                break
            else:
                color_image[it.multi_index] = rgbColorList[0]
        it.iternext()

    #for i in range(image.shape[0]):
        #for j in range(image.shape[1]):
            #for k in range(K-2, -1, -1):
                #if image[i, j] > thresholdValues[k]:
                    #color_image[i, j] = rgbColorList[k+1]
                    #break
                #else:
                    #color_image[i, j] = rgbColorList[0]

    return color_image
#############################################################################################################################
def OF2_Calc_InterClusterDistance(centers, inputData):
    numOfSegments = centers.size
    #listOfSegments = [[] for _ in range(numOfSegments)]
    #distSum = 0

    #it = np.nditer(inputData)

    #while not it.finished:
        #distances = [(it[0] - centers[i])**2 for i in range(numOfSegments)]
        #distSum += sum(distances)
        #it.iternext()

    distSum = sum((sum([(d - centers[i])**2 for i in range(numOfSegments)]) for d in np.nditer(inputData)))

    return distSum
#############################################################################################################################
def OF2_GetSegments(centers, inputData):
    numOfSegments = centers.size
    listOfSegments = [[] for _ in range(numOfSegments)]

    it = np.nditer(inputData, flags=["multi_index"])

    while not it.finished:
        distances = np.array([(it[0] - centers[i])**2 for i in range(numOfSegments)])
        listOfSegments[np.argmin(distances)].append(it.multi_index)
        it.iternext()

    return np.array(listOfSegments)
#############################################################################################################################
def OF2_DoImageSegmentation(segments, image, rgbColorList):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for s in range(segments.size):
        for e in range(len(segments[s])):
            color_image[segments[s][e]] = rgbColorList[s]

    return color_image


#############################################################################################################################
def OF0_TestFunction_SimpleParabolic(x):
    """
    calculate the square of input x

    - x: value for the function to be evaluated
    """
    return x ** 2
