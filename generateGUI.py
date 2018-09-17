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

# SNAPSHOT_NUM = "005407"
# SNAPSHOT_NUM = "000600"
SNAPSHOT_NUM = "000480"
#PATH_DATA = "/media/macramole/stuff/Data/pgan/"
#PATH_DATA = "/media/macramole/stuff/Data/pgan/1/"
# PATH_DATA = "/media/macramole/stuff/Data/pgan/2-latent2/"
# PATH_DATA = "/media/macramole/stuff/Data/pgan/004-pgan-yoFlores-preset-v2-1gpu-fp32-latent2-lod128-norepeat/"
PATH_DATA = "/media/macramole/stuff/Data/pgan/005-pgan-yoFlores-preset-v2-1gpu-fp32-latent2-lod128-norepeat/"

LATENT_RESOLUTION = 512

lastX = 0
lastY = 0

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
    # for s in sliders:
    #     inputVector.append(s.get() / LATENT_RESOLUTION)
    x = mapValue(lastX + sliderX.get(), 0, LATENT_RESOLUTION, -3, 3)
    y = mapValue(lastY + sliderY.get(), 0, LATENT_RESOLUTION, -3, 3)
    inputVector.append(x)
    inputVector.append(y)

    inputVector = np.array(inputVector).reshape((1,SIZE_LATENT_SPACE))
    print( inputVector )


    photo = generateImage()
    mainImage.configure(image=photo)
    mainImage.image = photo

def xyMoved(e):
    global lastX, lastY
    x = 0
    y = 0

    if type(e) is dict:
        x = e["x"]
        y = e["y"]
    else:
        x = e.x
        y = e.y

    lastX = x
    lastY = y
    updateImage()

    canvas.delete("all")
    canvas.create_oval(x-2,y-2,x+2,y+2, fill="green")

def sliderMoved(e):
    updateImage()

def mapValue(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def onAddPoint():
    pointList.insert(END, "%f,%f" % ( inputVector[0][0], inputVector[0][1] ) )
def onListClicked(e):
    strPos = pointList.get(pointList.curselection()).split(",")
    x = mapValue(float(strPos[0]), -3, 3, 0, LATENT_RESOLUTION)
    y = mapValue(float(strPos[1]), -3, 3, 0, LATENT_RESOLUTION)

    pos = { "x" : x, "y" : y }
    xyMoved(pos)

root = tk.Tk()
init()


# sliders = []
# for s in range(0,SIZE_LATENT_SPACE):
#     slider = Scale(root, from_=0, to=LATENT_RESOLUTION)
#     slider.bind("<B1-Motion>", sliderMoved)
#     slider.pack(side=tk.LEFT)
#     sliders.append(slider)

sliderX = tk.Scale(root, from_=1, to=-1, length=512, resolution=0.01)
sliderX.bind("<B1-Motion>", sliderMoved)
sliderX.grid(row=0,column=0)
sliderY = tk.Scale(root, from_=-1, to=1, length=512, resolution=0.01, orient=tk.HORIZONTAL)
sliderY.bind("<B1-Motion>", sliderMoved)
sliderY.grid(row=1,column=1)

canvas = tk.Canvas(root,width=512, height=512, bg="#000000", cursor="cross")
canvas.grid(row=0,column=1)
canvas.bind("<B1-Motion>", xyMoved)
canvas.bind("<Button 1>", xyMoved)

photo = generateImage()
mainImage = Label(root, image=photo)
mainImage.image = photo #esto es necesario por el garbage
# mainImage.pack( padx = SIZE_LATENT_SPACE)
mainImage.grid(row=0,column=2)

tk.Label(root, text="RESULT").grid(row=1,column=2)

pointList = tk.Listbox(root, height = 40)
pointList.grid(row=0,column=3, sticky=S)
pointList.bind("<Double-Button-1>", onListClicked)

btnAddPoint = tk.Button(root, text= "Add point", command=onAddPoint)
btnAddPoint.grid(row=0, column=3, sticky=N)

btnSaveVideo = tk.Button(root, text= "Save video")
btnSaveVideo.grid(row=1, column=3)

#b = Button(root, text="Ok", command=updateImage)
#b.pack()
root.title('PGAN Generator')
root.mainloop()
