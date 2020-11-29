# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:45:07 2020

@author: shash
"""
import cv2
import numpy as np


def readVidoe():
    out = []
    cap = cv2.VideoCapture("videos\Matrix-Trinity.mp4")
    if (cap.isOpened()== False):
        print("Error opening video  file") 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            out.append(frame)
            if cv2.waitKey(25) and 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return out

def writeVideo(outFrames, H, W):
    writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (W, H))
    for frame in outFrames:
        writer.write(frame)
    writer.release()
    print("Writing successfull")
    
def driver():
    outFrames = readVidoe()
    (H, W) = outFrames[0].shape[:2]
    writeVideo(outFrames, H, W)
    
driver()