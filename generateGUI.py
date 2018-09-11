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
import numpy as np
import pickle
import tensorflow as tf
import PIL.Image
import scipy

from PIL import Image, ImageTk

generator = None
SIZE_LATENT_SPACE = 2

#print("Forcing CPU usage...")
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

inputVector = []

SNAPSHOT_NUM = "005407"
#PATH_DATA = "/media/macramole/stuff/Data/pgan/"
#PATH_DATA = "/media/macramole/stuff/Data/pgan/1/"
PATH_DATA = "/media/macramole/stuff/Data/pgan/2-latent2/"

#%%

def init():
    global generator
    tf.InteractiveSession()
    
    with open(PATH_DATA + 'network-snapshot-' + SNAPSHOT_NUM + '.pkl', 'rb') as file:
        _, _, generator = pickle.load(file)

def generateFromGAN(latents):
    # Generate latent vectors.
#    latents = np.random.RandomState(1000).randn(1000, *generator.input_shapes[0][1:]) # 1000 random latents
#    latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
    
    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + generator.input_shapes[1][1:])
    
    # Run the generator to produce a set of images.
    images = generator.run(latents, labels)
    
    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    
    return images
    # Save images as PNG.
#    for idx in range(images.shape[0]):
#        PIL.Image.fromarray(images[idx], 'RGB').save(PATH_DATA + '/img%d.png' % idx)

def generateImage():
    global inputVector
    
    if len(inputVector) == 0 :
        inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))
        
    
#    img = dcgan.generateOne( np.array(inputVector) )#noise)
    img = generateFromGAN( np.array(inputVector) )#noise)
    img = img[0]
#    img = np.asarray( img * 255 , dtype="int" )
#    photo = ImageTk.PhotoImage(image=Image.frombytes('RGB', (img.shape[0],img.shape[1]), img.astype('b').tostring()))
    photo = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGB'))
    return photo

def updateImage():
    global inputVector
    
    inputVector = []
    for s in sliders:
        inputVector.append(s.get() / 1000)
    
    print( inputVector )
    inputVector = np.array(inputVector).reshape((1,SIZE_LATENT_SPACE))
    
    
    photo = generateImage()
    mainImage.configure(image=photo)
    mainImage.image = photo
    
def sliderMoved(e):
    updateImage()
    
root = tk.Tk()
init()
#dcgan = DCGAN( 
#    outputPath = "foo",
#    modelPath = "/home/leandro/Data/yo_128x128/train/dcgan_2018-08-02_17-39-38/models/generator.11000.h5" )


#no son 10 son 100 hay que ponerlos en una grilla

sliders = []
for s in range(0,SIZE_LATENT_SPACE):
    slider = Scale(root, from_=0, to=1000)
    slider.bind("<B1-Motion>", sliderMoved)
    slider.pack(side=tk.LEFT)
    sliders.append(slider)

photo = generateImage()
mainImage = Label(root, image=photo)
mainImage.image = photo #esto es necesario por el garbage
mainImage.pack( padx = SIZE_LATENT_SPACE)

#b = Button(root, text="Ok", command=updateImage)
#b.pack()

root.mainloop()