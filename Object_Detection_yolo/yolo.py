# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:08:02 2020

@author: shash
"""

import argparse
import numpy as np
import os
import cv2
import time
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="path to Input Image")
ap.add_argument("-y", "--yolo", required = True, help="path to Yolo-Coco Files")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability required to filter out weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help = "threshold when applying non-maximal suppression")


args = vars(ap.parse_args())


def readImage(imagePath):
    img = cv2.imread(imagePath)
    (H, W) = img.shape[:2]
    return img, H, W


def cocoLabels(yoloPath, fileName):
    labelsPath = os.path.sep.join([yoloPath, fileName])
    LABELS = open(labelsPath).read().strip().split("\n")
    return LABELS


def boxColors(length):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(length,3)).astype("uint8")
    return COLORS


def loadModel(weightsPath, configPath):
    print(weightsPath)
    print(configPath)
    print("[INFO]... Loading the Model from Disk.")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net


def forwardPass(net, image):
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob =  cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    return layerOutputs


def makeBoundingBoxes(layerOutputs, image, W, H, LABELS, COLORS, cutoff):
    boxes = []
    confidences = []
    classIDs = []
    
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            
            if confidence>cutoff:
                box = detection[0:4]*np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    
    
    if len(idxs)>0:
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def main():
    img, H, W = readImage(args["image"])
    LABELS = cocoLabels(args["yolo"], "coco.names")
    COLORS = boxColors(len(LABELS))
    net = loadModel(os.path.sep.join([args["yolo"], "yolov3.weights"]), os.path.sep.join([args["yolo"], "yolov3.cfg"]))
    layerOutputs = forwardPass(net, img)
    output = makeBoundingBoxes(layerOutputs, img, W, H, LABELS, COLORS, args["confidence"])
    cv2.imshow("output", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()