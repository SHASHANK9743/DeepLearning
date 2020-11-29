"""
Created on Sun Nov 29 21:08:02 2020

@author: shash
"""

import cv2
import os
import numpy as np
import time
#"videos\Matrix-Trinity.mp4"

def writeVideo(outFrames, H, W):
    writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (W, H))
    for frame in outFrames:
        writer.write(frame)
    writer.release()
    print("Writing successfull")
    
def imageProperties(image):
    (H, W) = image.shape[:2]
    return image, H, W


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
    layerOutputs = net.forward(ln)
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
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    
    if len(idxs)>0:
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image




def driverCode(path):
    cap = cv2.VideoCapture(path)
    net = loadModel(os.path.sep.join(["yolo-coco", "yolov3.weights"]), os.path.sep.join(["yolo-coco", "yolov3.cfg"]))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    outFrames = []
    if (cap.isOpened()== False):
        print("Error opening video  file") 
    i = 0
    totalStart = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image, H, W = imageProperties(frame)
            LABELS = cocoLabels("yolo-coco", "coco.names")
            COLORS = boxColors(len(LABELS))
            start = time.time()
            layerOutputs = forwardPass(net, image)
            end = time.time()
            print("The Frame processing took {} seconds".format(end-start))
            output = makeBoundingBoxes(layerOutputs, image, W, H, LABELS, COLORS, 0.5)
            outFrames.append(output)
            if cv2.waitKey(25) and 0xFF == ord('q'):
                break
        else:
            break
    totalEnd = time.time()
    print("The total time taken for video is {} seconds".format(totalEnd-totalStart))
    cap.release()
    cv2.destroyAllWindows()
    writeVideo(outFrames, H, W)
    
driverCode("videos\GTA San.mp4")