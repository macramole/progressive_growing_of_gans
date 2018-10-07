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
from tkinter import filedialog
#from tkinter import Scale
import numpy as np
import pickle
import tensorflow as tf
import PIL.Image
import scipy
import threading
import time
from PIL import Image, ImageTk

from scipy.interpolate import interp1d

generator = None
SIZE_LATENT_SPACE = None # filled on load
OUTPUT_RESOLUTION = None # filled on load

pointsSaved = []
inputVector = []

PATH_LOAD_FILE = "/media/macramole/stuff/Data/pgan/"
PATH_RESULT = "./generateResult/"

lastX = 0
lastY = 0

#%%

def init():
    # global generator
    tf.InteractiveSession()
    onLoadFile()

def generateFromGAN(latents):

    # latents = np.random.RandomState(1000).randn(1000, *generator.input_shapes[0][1:]) # 1000 random latents
    # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + generator.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = generator.run(latents, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    return images

def generateImage():
    global inputVector

    if len(inputVector) == 0 :
        inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))

    img = generateFromGAN( np.array(inputVector) )#noise)
    img = img[0]

    photo = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGB'))
    return photo

def updateImage(newInputVector = None):
    global inputVector

    # inputVector = []
    if not type(newInputVector) is np.ndarray:
        if newInputVector == None:
            x = mapValue(lastX, 0, OUTPUT_RESOLUTION, -3, 3)
            y = mapValue(lastY, 0, OUTPUT_RESOLUTION, -3, 3)

            inputVector[0][sliderX.get()] = x
            inputVector[0][sliderY.get()] = y
        else :
            inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))
    else:
        inputVector = newInputVector

    # inputVector = np.array(inputVector).reshape((1,SIZE_LATENT_SPACE))

    drawAllPointsPair()

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

    # lblCurrentPoint.config(text = str(inputVector[0]))


    # canvas.delete("all")
    # canvas.create_oval(x-2,y-2,x+2,y+2, fill="green")

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
    # pointList.insert(END, "%f,%f" % ( inputVector[0][0], inputVector[0][1] ) )
    pointList.insert(END, "%d" % ( pointList.size()+1 ) )
    pointsSaved.append( inputVector )

def onListClicked(e):
    # strPos = pointList.get(pointList.curselection()).split(",")
    # x = mapValue(float(strPos[0]), -3, 3, 0, OUTPUT_RESOLUTION)
    # y = mapValue(float(strPos[1]), -3, 3, 0, OUTPUT_RESOLUTION)
    #
    # pos = { "x" : x, "y" : y }
    # xyMoved(pos)

    updateImage( pointsSaved[pointList.curselection()[0]] )

def drawAllPointsPair():
    canvas.delete("all")

    for p in range(0, inputVector[0].shape[0] - 1, 2):
        x = mapValue(inputVector[0][p], -3,3, 0, OUTPUT_RESOLUTION)
        y = mapValue(inputVector[0][p+1], -3,3, 0, OUTPUT_RESOLUTION)

        canvas.create_oval(x-2,y-2,x+2,y+2, fill="green")

# def make_frame(t):
#     frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
#     latents = all_latents[frame_idx]
#     labels = np.zeros([latents.shape[0], 0], np.float32)
#     images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
#     grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
#     if image_zoom > 1:
#         grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
#     if grid.shape[2] == 1:
#         grid = grid.repeat(3, 2) # grayscale => RGB
#     return grid
#       ---> esto deberia devolver (width x height x 3) numpy <-- (of 8-bits integers)
#
# # Generate video.
# import moviepy.editor # pip install moviepy
# result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
# moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
# open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

def showFrame(generatedPhotos, index):
    photo = generatedPhotos[index]
    mainImage.configure(image=photo)
    mainImage.image = photo

    if index < 10:
        root.after(800, showFrame, generatedPhotos, index+1)

def onSaveVideo():
    cantInterpolation = 100

    for pointFrom in range(0, pointList.size()):
        pointTo = pointFrom + 1

        #loop
        if pointFrom == pointList.size()-1:
            pointTo = 0

        # strPos = []
        # strPos.append( pointList.get(pointFrom).split(",") )
        # strPos.append( pointList.get(pointTo).split(",") )

        x = np.array([0,1])
        # y = np.array(strPos, dtype="float")
        y = np.vstack((pointsSaved[pointFrom],pointsSaved[pointTo]))
        f = interp1d( x , y, axis = 0  )

        # arrInterpolation = f( np.linspace(0,1,cantInterpolation+1, endpoint = False) )
        arrInterpolation = f( np.linspace(0,1,cantInterpolation+1, endpoint = True) )

        # for latentSample in arrInterpolation:
        for i in range(0,len(arrInterpolation)):
            latentSample = np.array([arrInterpolation[i]])
            generated = generateFromGAN( latentSample )
            generatedImage = Image.fromarray(generated[0], 'RGB')
            generatedImage.save( "%s/%05d.png" % (PATH_RESULT,pointFrom*cantInterpolation+i) )

        # generatedImages = generateFromGAN( arrInterpolation )
        # generatedPhotos = []
        #
        # for g in generatedImages:
        #     photo = ImageTk.PhotoImage(image=Image.fromarray(g, 'RGB'))
        #     generatedPhotos.append(photo)
        #
        # showFrame(generatedPhotos, 0)

def onRandomClick():
    updateImage(False)

def onLoadFile():
    global generator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION

    filePath = filedialog.askopenfilename(initialdir = PATH_LOAD_FILE, title = "Select file")
    with open( filePath, 'rb' ) as file:
        _, _, generator = pickle.load(file)
        SIZE_LATENT_SPACE = int( generator.list_layers()[0][1].shape[1] )
        OUTPUT_RESOLUTION = int( generator.list_layers()[-1][1].shape[2] )

        root.title('PGAN Generator - %s' % filePath )

root = tk.Tk()
init()

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Load file", command=onLoadFile)
menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)

sliderX = tk.Scale(root, from_=0, to=SIZE_LATENT_SPACE-1, length=OUTPUT_RESOLUTION, resolution=1, orient=tk.HORIZONTAL)
# sliderX.bind("<B1-Motion>", sliderMoved)
sliderX.grid(row=1,column=1)
sliderY = tk.Scale(root, from_=0, to=SIZE_LATENT_SPACE-1, length=OUTPUT_RESOLUTION, resolution=1)
sliderY.set(1)
# sliderY.bind("<B1-Motion>", sliderMoved)
sliderY.grid(row=0,column=0)

canvas = tk.Canvas(root,width=OUTPUT_RESOLUTION, height=OUTPUT_RESOLUTION, bg="#000000", cursor="cross")
canvas.grid(row=0,column=1)
canvas.bind("<B1-Motion>", xyMoved)
canvas.bind("<Button 1>", xyMoved)

photo = generateImage()
mainImage = Label(root, image=photo)
mainImage.image = photo #esto es necesario por el garbage
# mainImage.pack( padx = SIZE_LATENT_SPACE)
mainImage.grid(row=0,column=2)

# lblCurrentPoint = tk.Label(root, text="RESULT")
# lblCurrentPoint.grid(row=1,column=2)
btnRandom = tk.Button(root, text="Random", command=onRandomClick)
btnRandom.grid(row=1,column=2)

pointList = tk.Listbox(root, height = 40)
pointList.grid(row=0,column=3, sticky=S)
pointList.bind("<Double-Button-1>", onListClicked)

btnAddPoint = tk.Button(root, text= "Add point", command=onAddPoint)
btnAddPoint.grid(row=0, column=3, sticky=N)

btnSaveVideo = tk.Button(root, text= "Save video", command=onSaveVideo)
btnSaveVideo.grid(row=1, column=3)

root.mainloop()
