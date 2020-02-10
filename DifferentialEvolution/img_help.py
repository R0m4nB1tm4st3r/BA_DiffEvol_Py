#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import cv2

from tesserocr import PyTessBaseAPI, RIL, PSM, OEM
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
def Tesseract_ReadTextFromImage(image):

    with PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.TESSERACT_LSTM_COMBINED) as api:
        api.SetImage(image)
        #boxes = api.GetComponentImages(RIL.TEXTLINE, True)

        #print("Found {} textline image components.".format(len(boxes)))
        #for i, (im, box, _, _) in enumerate(boxes):
            #api.SetRectangle(box["x"], box["y"], box["w"], box["h"])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()
            #print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
              #"confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))
            #ocrEndResult += ocrResult
        print("".join(["OCR Result: ", ocrResult]))
        print("".join(["Confidence: ", str(conf)]))

    return ocrResult#ocrEndResult