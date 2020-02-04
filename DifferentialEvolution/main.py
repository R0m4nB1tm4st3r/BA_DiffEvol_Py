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
import img_help as ih

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
    G = 50
    populationSize_OF2 = 10*K
    numOfImgs = 15
    objFunc = 2
    altImgsFlag = True
    denoiseFlag = True
    imgPathString = ""
    resultsPathString = "".join(["ResultsOF", str(objFunc)]) 
    
    if altImgsFlag == False:
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
                            ])

        resultsPathString = "".join([resultsPathString, "\\"]) 
        imgPathString = "SourceImages\\"

    else:
        imgStrings = np.array([ "IMG_3909200", \
                                "IMG_3909201", \
                                "IMG_3909204", \
                                "IMG_3909208", \
                                "IMG_3909209", \
                                "IMG_3909213", \
                                "IMG_3909216", \
                                "IMG_3909217", \
                                "IMG_3909224", \
                                "IMG_3909233", \
                                "IMG_3909236", \
                                "IMG_3909237", \
                                "IMG_3909241", \
                                "IMG_3909244", \
                                "IMG_3909245"])

        resultsPathString = "".join([resultsPathString, "_alt\\"])

        imgPathString = "SourceImages_Alternative\\"

    images = np.array(list((ih.GetImage("".join([imgPathString, imgStrings[i], ".jpg"]), cv2.IMREAD_GRAYSCALE) for i in range(numOfImgs))))
    if denoiseFlag == True:
        images = np.array(list((cv2.fastNlMeansDenoising(images[i], None, 20, 7, 21) for i in range(numOfImgs))))
        #for c in range(numOfImgs):
            #ih.SaveImage(images[c], "".join([imgPathString, imgStrings[c], "-denoised.jpg"]))
            #ih.ShowImage(images[c])

    graylevels = np.arange(256)

    rgbColorPool = (np.array([[  0,    0,    0], [255,  255,  255]], dtype=np.uint8), \
                    np.array([[  0,    0,    0], [102,    0,    0], [255,  255,  255]], dtype=np.uint8), \
                    np.array([[  0,    0,    0], [102,    0,    0], [102,  102,    0], [255,  255,  255]], dtype=np.uint8), \
                    np.array([[  0,    0,    0], [102,    0,    0], [102,  102,    0], [ 76,  153,    0], [255,  255,  255]], dtype=np.uint8))

    # black, red, yellow, green, white
    rgbColorList = rgbColorPool[K-2]

    if objFunc == 1:
        t_min = np.array([])
        t_max = np.array([])
        t_min = np.append(t_min, [list(repeat(0., K)), list(repeat(0., K)), list(repeat(0., K))]) 
        t_max = np.append(t_max, [list(repeat(1., K)), list(repeat(255., K)), list(repeat(7., K))]) 
    elif objFunc == 2:
        minBounds_OF2 = np.array(list(repeat(0., K)))
        maxBounds_OF2 = np.array(list(repeat(255., K)))

    de_param_string = "".join(["K_", str(K), "-", "G_", str(G), "-", "F_", str(F), "-", "Cr_", str(Cr)])
    ############################################################################################################
    ############################################################################################################
    print("Put in the Test Number to start with:")
    testNumber = int(input())

    #currentDate = time.strftime("%Y/%m/%d").replace("/", "_")
    if altImgsFlag == True:
        if denoiseFlag == True:
            de_test_csv = open("".join([resultsPathString, "de_test_OF", str(objFunc), "_alt_denoised", ".csv"]), mode = "a")
        else:
            de_test_csv = open("".join([resultsPathString, "de_test_OF", str(objFunc), "_alt", ".csv"]), mode = "a")
    else:
        if denoiseFlag == True:
            de_test_csv = open("".join([resultsPathString, "de_test_OF", str(objFunc), "_denoised.csv"]), mode = "a")
        else:
            de_test_csv = open("".join([resultsPathString, "de_test_OF", str(objFunc), ".csv"]), mode = "a")

    csv_writer = csv.writer(de_test_csv, \
        delimiter=';', \
        quoting=csv.QUOTE_MINIMAL)

    if os.stat(de_test_csv.name).st_size == 0:
        if objFunc == 1:
            csv_writer.writerow(["Test Number", "Image Name", "Threshold Combination", "Number of Classes K", "Number of Iterations G", "Mutation Factor F", "Crossover Rate Cr", "Tesseract Read Result"])
        elif objFunc == 2:
            csv_writer.writerow(["Test Number", "Image Name", "Number of Classes K", "Number of Iterations G", "Mutation Factor F", "Crossover Rate Cr", "Tesseract Read Result"])
    ############################################################################################################
    ############################################################################################################
    for j in range(numOfImgs):
        print("Optimizing...")
        np.random.seed(123)

        # Initialize needed parameters for DE #
        h = obf.OF1_CalculateNormalizedHistogram(images[j])
        
        if objFunc == 1:
            test_population = InitializePopulation(3*10*K, 3*K)
            objArgs = (K, graylevels, h, o)
            deHandler = de.DE_Handler(F, Cr, G, 3*10*K, test_population, obf.OF1_CalcErrorEstimation, True, t_min, t_max, objArgs)

            bestParams, bestValueHistory = deHandler.DE_GetBestParameters()
            bestMember = bestParams[0]
            thresholdValues = obf.OF1_CalculateThresholdValues(bestMember, K)
            thresholdCombinations = np.array(list(product(*thresholdValues)))
            newImages = np.array([obf.OF1_DoImageSegmentation(images[j], thresholdCombinations[t], K, rgbColorList) \
                for t in range(thresholdCombinations.shape[0]) \
                if np.amin(thresholdCombinations[t]) != -1 \
                ])
            thresholdindices = np.array([i for i in range(thresholdCombinations.shape[0]) if np.amin(thresholdCombinations[i] != -1)])

            #timeString = currentDate

            for n in range(newImages.shape[0]):
                if denoiseFlag == True:
                    segImgFileName = "".join([resultsPathString, imgStrings[j], "_denoised-", de_param_string, "-", "SEG_Test-", str(n), ".jpg"])
                else:
                    segImgFileName = "".join([resultsPathString, imgStrings[j], "-", de_param_string, "-", "SEG_Test-", str(n), ".jpg"])
                
                ocrEndResult = ""

                # plot the DE and Segmentation results #
                plotFigure, plotAxes = plot.CreateSubplotGrid(2, 1, False)
                plotFigure.set_dpi(200)
                plotAxes[0].plot(graylevels, h)
                plotAxes[0].plot(graylevels, obf.OF1_SumOfGauss(bestMember, K, graylevels))
                plotAxes[0].legend(("Histogram of the original Image", "Gaussian Approximation of the Histogram", ))
                plotAxes[0].vlines(thresholdCombinations[thresholdindices[n]], 0, np.amax(h), label="Threshold Values")
                for k in range(thresholdCombinations[thresholdindices[n]].size):
                    plotAxes[0].annotate("T" + str(k+1), xy=(thresholdCombinations[thresholdindices[n], k], np.amax(h)), )
                
                plotAxes[0].set_xlabel("Graylevel g")
                plotAxes[0].set_ylabel("n_Pixel_relative")
                plotAxes[0].set_title("".join(["Mean Square Error: ", str(bestParams[1])]))
                plotAxes[1] = plt.imshow(newImages[n], cmap='gray')
                if denoiseFlag == True:
                    plotAxes[1].axes.set_title("Result of Image Segmentation (with denoised Image)")
                    plotString = "".join([resultsPathString, imgStrings[j], "_denoised-", de_param_string, "-", "SEG_Plot-", str(n), ".jpg"])
                else:
                    plotAxes[1].axes.set_title("Result of Image Segmentation")
                    plotString = "".join([resultsPathString, imgStrings[j], "-", de_param_string, "-", "SEG_Plot-", str(n), ".jpg"])

                plt.tight_layout()
                plt.savefig(plotString, dpi=200)
                
                #plt.show(block=False)
                plt.close(plotFigure)

                newImages[n] = cv2.cvtColor(newImages[n], cv2.COLOR_RGB2BGR)
                ih.SaveImage(newImages[n], segImgFileName)
                seg_image = Image.open(segImgFileName)
                ocrEndResult = ih.Tesseract_ReadTextFromImage(seg_image)

                csv_writer.writerow([j+testNumber, imgStrings[j], str(n), K, G, str(F).replace(".", ","), str(Cr).replace(".", ","), ocrEndResult.encode("utf8")])#.encode(sys.stdout.encoding, errors='replace')])
        elif objFunc == 2:
            testPopulation_OF2 = InitializePopulation(populationSize_OF2, K)
            objArgs_OF2 = (images[j],)
            deHandler = de.DE_Handler(F, Cr, G, populationSize_OF2, testPopulation_OF2, obf.OF2_Calc_InterClusterDistance, True, minBounds_OF2, maxBounds_OF2, objArgs_OF2)
            
            centers, bestValueHistory = deHandler.DE_GetBestParameters()

            segments = obf.OF2_GetSegments(centers[0], images[j])
            newImage = obf.OF2_DoImageSegmentation(segments, images[j], rgbColorList)
        
            #timeString = currentDate
            if denoiseFlag == True:
                segImgFileName = "".join([resultsPathString, imgStrings[j], "_denoised-", de_param_string, "-", "SEG_Test", ".jpg"])
            else:
                segImgFileName = "".join([resultsPathString, imgStrings[j], "-", de_param_string, "-", "SEG_Test", ".jpg"])
            ocrEndResult = ""

            # plot the DE and Segmentation results #
            plotFigure, plotAxes = plot.CreateSubplotGrid(2, 1, False)
            plotFigure.set_dpi(200)
            plotAxes[0].plot(graylevels, h)
            plotAxes[0].legend(("Histogram of the original Image", ))
            plotAxes[0].vlines(centers[0], 0, np.amax(h), label="Cluster Centers")
            for k in range(K):
                plotAxes[0].annotate("c" + str(k+1), xy=(centers[0][k], np.amax(h)), )
            plotAxes[0].set_xlabel("Graylevel g")
            plotAxes[0].set_ylabel("n_Pixel_relative")
            plotAxes[0].set_title("".join(["Cluster Distance Sum: ", str(centers[1])]))
            plotAxes[1] = plt.imshow(newImage, cmap='gray')
            if denoiseFlag == True:
                plotAxes[1].axes.set_title("Result of Image Segmentation (with denoised Image")
                plotString = "".join([resultsPathString, imgStrings[j], "_denoised-", de_param_string, "-", "SEG_Plot", ".jpg"])
            else:
                plotAxes[1].axes.set_title("Result of Image Segmentation")
                plotString = "".join([resultsPathString, imgStrings[j], "-", de_param_string, "-", "SEG_Plot", ".jpg"])

            plt.tight_layout()
            plt.savefig(plotString, dpi=200)
            plt.close(plotFigure)

            newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2BGR)
            ih.SaveImage(newImage, segImgFileName)
            seg_image = Image.open(segImgFileName)
            ocrEndResult = ih.Tesseract_ReadTextFromImage(seg_image)

            csv_writer.writerow([j+testNumber, imgStrings[j], K, G, str(F).replace(".", ","), str(Cr).replace(".", ","), ocrEndResult.encode("utf8")])#.encode(sys.stdout.encoding, errors='replace')])
        

        # plot the Fitness values #
        valHistFigure, valHistAxes = plot.CreateSubplotGrid(1, 1, False)
        valHistAxes.plot(range(1, G+1), bestValueHistory)
        valHistAxes.set_xlabel("Iteration Number")
        valHistAxes.set_ylabel("Fitness Value")
        valHistAxes.set_title("Fitness Value History through iterations of DE")
        if denoiseFlag == True:
            plt.savefig("".join([resultsPathString, imgStrings[j], "_denoised-", de_param_string, "-", "objFuncHist-", ".jpg"]), dpi=200)
        else:
            plt.savefig("".join([resultsPathString, imgStrings[j], "-", de_param_string, "-", "objFuncHist-", ".jpg"]), dpi=200)

        plt.close(valHistFigure)
    ############################################################################################################
    ############################################################################################################
