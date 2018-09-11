#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:35:21 2018

@author: leandro
"""

#%%

import os
import tkinter as tk
from tkinter import *
#from tkinter import Scale
from dcgan import DCGAN
import numpy as np

from PIL import Image, ImageTk

model = None

#print("Forcing CPU usage...")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

inputVector = []
#%%

def generateImage():
    global inputVector
    
    if len(inputVector) == 0 :
        inputVector = np.random.normal(0, 1, (1, 100))
        
    
    img = dcgan.generateOne( np.array(inputVector) )#noise)
    img = img[0]
    img = np.asarray( img * 255 , dtype="int" )
    photo = ImageTk.PhotoImage(image=Image.frombytes('RGB', (img.shape[0],img.shape[1]), img.astype('b').tostring()))
    return photo

def updateImage():
    global inputVector
    
    inputVector = []
    for s in sliders:
        inputVector.append(s.get() / 1000)
    
    print( inputVector )
    inputVector = np.array(inputVector).reshape((1,100))
    
    
    photo = generateImage()
    mainImage.configure(image=photo)
    mainImage.image = photo
    
def sliderMoved(e):
    updateImage()
    
root = tk.Tk()
dcgan = DCGAN( 
    outputPath = "foo",
    modelPath = "/home/leandro/Data/yo_128x128/train/dcgan_2018-08-02_17-39-38/models/generator.11000.h5" )

#no son 10 son 100 hay que ponerlos en una grilla

sliders = []
for s in range(0,10):
    slider = Scale(root, from_=0, to=1000)
    slider.bind("<B1-Motion>", sliderMoved)
    slider.pack(side=tk.LEFT)
    sliders.append(slider)

photo = generateImage()
mainImage = Label(root, image=photo)
mainImage.image = photo #esto es necesario por el garbage
mainImage.pack( padx = 10)

#b = Button(root, text="Ok", command=updateImage)
#b.pack()

root.mainloop()