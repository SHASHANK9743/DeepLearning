# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:20:21 2020

@author: SHASHANK DWIVEDI

Input class, this class will be called by the server, to parse the recieved information from
the front end to params dictionary.

"""
class Inp():
    def __init__(self):
        self.width = 224
        self.height = 224
        self.channels = 3
        self.arch = []
        self.opt = "Adam"
        self.lr = 1e-3
        
    
    def arch_layer():
        
        conf = [{'layer_type':'Conv2D'}, {'layer_type':'Conv2D'}, {'layer_type':'Conv2D'}, {'layer_type':'Conv2D'}, {'layer_type':'Flatten'}, {'layer_type':'Dense'}]
        pass
