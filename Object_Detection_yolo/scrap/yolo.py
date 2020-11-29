"""
Author : Shashank Dwivedi
Date : 13-11-2020
Yolo Object Detection Code for Images
"""

#Importing the Neccessary Module required to perform Object Detection
import numpy as np
import argparse
import time
import cv2
import os

#Neccessary information require to perform Object Detection
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="path to Input Image")
ap.add_argument("-y", "--yolo", required = True, help="base path to Yolo directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability required to filter out weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help = "threshold when applying non-maximal suppression")

args = vars(ap.parse_args())


#Loading the COCO Dataset Labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#Randomizing the Colors of the Bounding Boxes of Various Objects in iamages
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS),3), dtype="unit8")

#Initializing the path to COCO wieghts path and configuration file
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

#Loading the Yolo Model
print("[INFO] loading YOLO from disk... ")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#loading the Image
image = cv2.imread(args["image"])
(H, W) = (None, None)


#All Layers name from YOLO Model
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]


# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities     
blob =  cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.timee()


#Printing the time required to perfomr forward Pass    
print("[INFO] YOLO took {:.6f} seconds".format(end - start))    
boxes = []
confidences = []
classIDs = []