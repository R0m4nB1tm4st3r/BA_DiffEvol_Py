#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import csv
import os

import diff_evol as de
import plot
import seg

from tesserocr import PyTessBaseAPI, RIL
from PIL import Image
from itertools import repeat
from scipy.stats import norm

#############################################################################################################################
#####################################--Globals--#############################################################################
#############################################################################################################################


#############################################################################################################################
#######################################--Main--##############################################################################
#############################################################################################################################
if __name__ == "__main__":
    # F: 0.25, 0.5, 0.25, 0.8
    # Cr: 0.8, 0.9, 0.1, 1.0
    F = 0.25
    Cr = 0.1
    o = 1.5
    K = 5
    G = 5000

    print("Put in the Test Number to start with:")
    testNumber = int(input())
    imgStrings = np.array(["IMG_1255_DLXFFNICKLT_c", "IMG_1324_DLXFFNICKLT_c", "IMG_1349_0ZTTM0TUN3H_c", "IMG_1555_7D91CX6GNPA_c"])
    images = np.array(list((seg.GetImage(imgStrings[i] + ".jpg", cv2.IMREAD_GRAYSCALE) for i in range(4))))
    #images = np.array(list((cv2.fastNlMeansDenoising(GetImage(imgStrings[i] + ".jpg", cv2.IMREAD_GRAYSCALE), None, 10, 7, 21) for i in range(4))))

    graylevels = np.arange(256)

    # black, red, yellow, green, white
    rgbColorList = np.array([   [  0,    0,    0], \
                                [102,    0,    0], \
                                [102,  102,    0], \
                                [ 76,  153,    0], \
                                [255,  255,  255]], \
                                dtype=np.uint8)


    currentDate = time.strftime("%d/%m/%Y").replace("/", "_")
    de_test_csv = open("de_test" + currentDate + ".csv", mode = "a")
    csv_writer = csv.writer(de_test_csv, \
        delimiter=';', \
        quoting=csv.QUOTE_MINIMAL)

    if os.stat(de_test_csv.name).st_size == 0:
        csv_writer.writerow(["Test Number", "Image Name", "Execution Time", "Number of Classes K", "Number of Iterations G", "Mutation Factor F", "Crossover Rate Cr", "Tesseract Read Result"])

    t_min = np.array([])
    t_max = np.array([])
    t_min = np.append(t_min, [list(repeat(0., K)), list(repeat(0., K)), list(repeat(0., K))]) 
    t_max = np.append(t_max, [list(repeat(1., K)), list(repeat(255., K)), list(repeat(6., K))]) 

    de_param_string = "G_" + str(G) + "-" + "K_" + str(K) + "-" + "F_" + str(F) + "-" + "Cr_" + str(Cr)

    for j in range(4):
        print("Optimizing...")
        np.random.seed(123)

        # Initialize needed parameters for DE #
        h = seg.CalculateNormalizedHistogram(images[j])
        test_population = seg.InitializePopulation(3*10*K, 3*K)
        objArgs = (K, graylevels, h, o)

        # Execute DE and do Segmentation of the current image #
        de_handle = de.DE_Handler(F, Cr, G, 3*10*K, test_population, seg.CalcErrorEstimation, True, t_min, t_max, objArgs)
        bestParams, bestValueHistory = de_handle.DE_GetBestParameters()
        bestMember = bestParams[0]
        thresholdValues = seg.CalculateThresholdValues(bestMember, K)
        newImage = seg.DoImageSegmentation(images[j], thresholdValues, K, rgbColorList)
        newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2BGR)
        currentTime = time.strftime("%H:%M:%S").replace(":", "_")
        timeString = currentDate + "-" + currentTime
        segImgFileName = imgStrings[j] + "-" + "SEG_Test-" + timeString + "-" + de_param_string + ".jpg" #+ "-" + "No_" + str(x) + ".jpg"
        seg.SaveImage(newImage, segImgFileName)
        seg_image = Image.open(segImgFileName)
        ocrEndResult = ""

        # plot the Fitness values #
        valHistFigure, valHistAxes = plot.CreateSubplotGrid(1, 1, False)
        valHistAxes.plot(range(1, G+1), bestValueHistory)
        valHistAxes.set_xlabel("Iteration Number")
        valHistAxes.set_ylabel("Mean Square Error")
        valHistAxes.set_title("Mean Square Error History through iterations of DE")# run " + str(x))
        plt.savefig(imgStrings[j] + "-" + "objFuncHist-" + timeString + "-" + de_param_string + ".jpg", dpi=200) #"-" + "No_" + str(x) + ".jpg", dpi=200)
        #plt.show(block=False)

        # plot the DE and Segmentation results #
        plotFigure, plotAxes = plot.CreateSubplotGrid(2, 1, False)
        plotFigure.set_dpi(200)
        plotAxes[0].plot(graylevels, h)
        plotAxes[0].plot(graylevels, seg.SumOfGauss(bestMember, K, graylevels))
        plotAxes[0].legend(("Histogram of the original Image", "Gaussian Approximation of the Histogram", ))
        plotAxes[0].vlines(thresholdValues, 0, np.amax(h), label="Threshold Values")
        for k in range(thresholdValues.size):
            plotAxes[0].annotate("T" + str(k+1), xy=(thresholdValues[k], np.amax(h)), )
        plotAxes[0].set_xlabel("Graylevel g")
        plotAxes[0].set_ylabel("n_Pixel_relative")
        plotAxes[0].set_title("Mean Square Error: " + str(bestParams[1]))
        plotAxes[1] = plt.imshow(newImage, cmap='gray')
        plotAxes[1].axes.set_title("Result of Image Segmentation")
        plt.tight_layout()
        plt.savefig(imgStrings[j] + "-" + "SEG_Plot-" + timeString + "-" + de_param_string + ".jpg", dpi=200)# + "-" + "No_" + str(x) + ".jpg", dpi=200)
        #plt.show(block=False)

        # try to read text from segmented image with Tesseract #
        ocrEndResult = seg.Tesseract_ReadTextFromImage(seg_image)

        # write the results into csv file #
        csv_writer.writerow([j+testNumber, imgStrings[j], currentTime, K, G, F, Cr, ocrEndResult])
