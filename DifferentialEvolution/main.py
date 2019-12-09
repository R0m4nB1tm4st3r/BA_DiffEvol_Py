#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import csv
import os
import sys

import diff_evol as de
import objFuncs as obf
import plot
import seg

from PIL import Image
from itertools import repeat
from itertools import product

#############################################################################################################################
#####################################--Functions--###########################################################################
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
#####################################--Globals--#############################################################################
#############################################################################################################################


#############################################################################################################################
#######################################--Main--##############################################################################
#############################################################################################################################
if __name__ == "__main__":
    ############################################################################################################
    ############################################################################################################
    # F: 0.25, 0.5, 0.25, 0.8
    # Cr: 0.8, 0.9, 0.1, 1.0
    F = 0.5
    Cr = 0.9
    o = 1.5
    K = 5
    G = 500
    populationSize_OF2 = 5*K
    numOfImgs = 17
    imgStrings = np.array([ "IMG_01002DOKR5B_c", \
                            "IMG_01002DOKS32_c", \
                            "IMG_01002DOKTAT_c", \
                            "IMG_01002DOKUGK_c", \
                            "IMG_01002DOKVX3_c", \
                            "IMG_01002DOKXUC_c", \
                            "IMG_01002DOKYIL_c", \
                            "IMG_01002DOKZPU_c", \
                            "IMG_01002DOL0UD_c", \
                            "IMG_01002DOL1X4_c", \
                            "IMG_01002DOL2PV_c", \
                            "IMG_01002DOL3IM_c", \
                            "IMG_01002DOL6GN_c", \
                            "IMG_01002DOL55E_c", \
                            "IMG_01002DOL435_c", \
                            "IMG_01002DOLBFF_c", \
                            "IMG_01002DOLH01_c", \
                            #"IMG_1255_DLXFFNICKLT_c", \
                            #"IMG_1324_DLXFFNICKLT_c", \
                            #"IMG_1349_0ZTTM0TUN3H_c", \
                            #"IMG_1555_7D91CX6GNPA_c" \
                            ])

    images = np.array(list((seg.GetImage(imgStrings[i] + ".jpg", cv2.IMREAD_GRAYSCALE) for i in range(numOfImgs))))
    #images = np.array(list((cv2.fastNlMeansDenoising(images, None, 20, 7, 21) for i in range(numOfImgs))))
    graylevels = np.arange(256)

    # black, red, yellow, green, white
    rgbColorList = np.array([   [  0,    0,    0], \
                                [102,    0,    0], \
                                [102,  102,    0], \
                                [ 76,  153,    0], \
                                [255,  255,  255]], \
                                dtype=np.uint8)

    t_min = np.array([])
    t_max = np.array([])
    t_min = np.append(t_min, [list(repeat(0., K)), list(repeat(0., K)), list(repeat(0., K))]) 
    t_max = np.append(t_max, [list(repeat(1., K)), list(repeat(255., K)), list(repeat(7., K))]) 
    #minBounds_OF2 = np.array(list(repeat(0., K)))
    #maxBounds_OF2 = np.array(list(repeat(255., K)))

    de_param_string = "G_" + str(G) + "-" + "K_" + str(K) + "-" + "F_" + str(F) + "-" + "Cr_" + str(Cr)
    ############################################################################################################
    ############################################################################################################
    print("Put in the Test Number to start with:")
    testNumber = int(input())

    currentDate = time.strftime("%d/%m/%Y").replace("/", "_")
    de_test_csv = open("de_test" + currentDate + ".csv", mode = "a")
    csv_writer = csv.writer(de_test_csv, \
        delimiter=';', \
        quoting=csv.QUOTE_MINIMAL)

    if os.stat(de_test_csv.name).st_size == 0:
        csv_writer.writerow(["Test Number", "Image Name", "Threshold Combination", "Number of Classes K", "Number of Iterations G", "Mutation Factor F", "Crossover Rate Cr", "Tesseract Read Result"])

    ############################################################################################################
    ############################################################################################################
    for j in range(numOfImgs):
        print("Optimizing...")
        np.random.seed(123)

        # Initialize needed parameters for DE #
        h = obf.OF1_CalculateNormalizedHistogram(images[j])
        
        test_population = InitializePopulation(3*10*K, 3*K)
        objArgs = (K, graylevels, h, o)
        #testPopulation_OF2 = InitializePopulation(populationSize, K)
        #objArgs_OF2 = (images[j],)

        # Execute DE and do Segmentation of the current image #
        de_handle = de.DE_Handler(F, Cr, G, 3*10*K, test_population, obf.OF1_CalcErrorEstimation, True, t_min, t_max, objArgs)
        #de_OF2 = de.DE_Handler(F, Cr, G, populationSize, testPopulation_OF2, obf.OF2_Calc_InterClusterDistance, True, minBounds_OF2, maxBounds_OF2, objArgs_OF2)
        
        bestParams, bestValueHistory = de_handle.DE_GetBestParameters()
        bestMember = bestParams[0]
        thresholdValues = obf.OF1_CalculateThresholdValues(bestMember, K)
        thresholdCombinations = np.array(list(product(*thresholdValues)))
        #newImage = obf.OF1_DoImageSegmentation(images[j], thresholdValues, K, rgbColorList)
        newImages = np.array([obf.OF1_DoImageSegmentation(images[j], thresholdCombinations[t], K, rgbColorList) \
            for t in range(thresholdCombinations.shape[0]) \
            if np.amin(thresholdCombinations[t]) != -1 \
            ])
        thresholdindices = np.array([i for i in range(thresholdCombinations.shape[0]) if np.amin(thresholdCombinations[i] != -1)])

        timeString = currentDate

        for n in range(newImages.shape[0]):

            segImgFileName = imgStrings[j] + "-" + "SEG_Test-" + str(n) + "-" + de_param_string + ".jpg" #+ "-" + "No_" + str(x) + ".jpg"
            ocrEndResult = ""

            # plot the DE and Segmentation results #
            plotFigure, plotAxes = plot.CreateSubplotGrid(2, 1, False)
            plotFigure.set_dpi(200)
            plotAxes[0].plot(graylevels, h)
            plotAxes[0].plot(graylevels, obf.OF1_SumOfGauss(bestMember, K, graylevels))
            plotAxes[0].legend(("Histogram of the original Image", "Gaussian Approximation of the Histogram", ))
            #plotAxes[0].legend(("Histogram of the original Image", ))
            plotAxes[0].vlines(thresholdCombinations[thresholdindices[n]], 0, np.amax(h), label="Threshold Values")
            #plotAxes[0].vlines(centers[0], 0, np.amax(h), label="Cluster Centers")
            for k in range(thresholdCombinations[thresholdindices[n]].size):
                plotAxes[0].annotate("T" + str(k+1), xy=(thresholdCombinations[thresholdindices[n], k], np.amax(h)), )
            #for k in range(K):
                #plotAxes[0].annotate("c" + str(k+1), xy=(centers[k], np.amax(centers)), )
            plotAxes[0].set_xlabel("Graylevel g")
            plotAxes[0].set_ylabel("n_Pixel_relative")
            plotAxes[0].set_title("Mean Square Error: " + str(bestParams[1]))
            plotAxes[1] = plt.imshow(newImages[n], cmap='gray')
            plotAxes[1].axes.set_title("Result of Image Segmentation")
            plt.tight_layout()
            plt.savefig(imgStrings[j] + "-" + "SEG_Plot-" + str(n) + "-" + de_param_string + ".jpg", dpi=200)
            #plt.show(block=False)
            plt.close(plotFigure)

            newImages[n] = cv2.cvtColor(newImages[n], cv2.COLOR_RGB2BGR)
            #seg.ShowImage(newImages[n])
            seg.SaveImage(newImages[n], segImgFileName)
            seg_image = Image.open(segImgFileName)
            ocrEndResult = seg.Tesseract_ReadTextFromImage(seg_image)
            csv_writer.writerow([j+testNumber, imgStrings[j], str(n), K, G, str(F).replace(".", ","), str(Cr).replace(".", ","), ocrEndResult.encode(sys.stdout.encoding, errors='replace')])

        # plot the Fitness values #
        valHistFigure, valHistAxes = plot.CreateSubplotGrid(1, 1, False)
        valHistAxes.plot(range(1, G+1), bestValueHistory)
        valHistAxes.set_xlabel("Iteration Number")
        valHistAxes.set_ylabel("Mean Square Error")
        valHistAxes.set_title("Mean Square Error History through iterations of DE")# run " + str(x))
        plt.savefig(imgStrings[j] + "-" + "objFuncHist-" + de_param_string + ".jpg", dpi=200) #"-" + "No_" + str(x) + ".jpg", dpi=200)
        #plt.show(block=False)
    ############################################################################################################
    ############################################################################################################
