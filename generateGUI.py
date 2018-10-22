#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:35:21 2018

@author: leandro
"""

#%%

import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog

#from tkinter import Scale
import numpy as np
import pickle
import tensorflow as tf
import PIL.Image
import scipy
import threading
import time
from PIL import Image, ImageTk
import subprocess

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

selectionRectangle = None
selectionRectangleOriginalCoords = None
pointsMoveOriginalCoords = None
COLOR_POINT = "green"
COLOR_SELECTED = "red"
POINT_RADIUS = 2

canvas = None
pointList = None

recording = False
recordingCurrentFrame = 0

arrayImage = None
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

    img = generateFromGAN( inputVector )#noise)
    img = img[0]

    # photo = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGB'))
    photo = Image.fromarray(img, 'RGB')
    return photo

def updateImage(newInputVector = None):
    global inputVector, recordingCurrentFrame, arrayImage

    # inputVector = []
    if not type(newInputVector) is np.ndarray:
        inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))
        drawAllPointsPair()
    else:
        inputVector = newInputVector

    # inputVector = np.array(inputVector).reshape((1,SIZE_LATENT_SPACE))

    arrayImage = generateImage()
    photo = ImageTk.PhotoImage(image=arrayImage)
    mainImage.configure(image=photo)
    mainImage.image = photo

    if recording:
        arrayImage.save( "%s/%05d.png" % (PATH_RESULT,recordingCurrentFrame) )
        recordingCurrentFrame+=1

def xyClicked(e):
    global pointsMoveOriginalCoords

    pointsMoveOriginalCoords = None
    canvas.itemconfig( "selected", fill=COLOR_POINT )
    canvas.dtag("selected")
def xyMoved(e):
    global selectionRectangle, selectionRectangleOriginalCoords, pointsMoveOriginalCoords

    if selectionRectangle is None:

        if len( canvas.find_withtag("selected") ) == 0:
            selectionRectangle = canvas.create_rectangle(e.x, e.y, e.x + 5, e.y + 5, outline = "white")
            selectionRectangleOriginalCoords = (e.x, e.y)
        else:
            if pointsMoveOriginalCoords is None:
                pointsMoveOriginalCoords = (e.x, e.y)
            else:
                canvas.move("selected", e.x - pointsMoveOriginalCoords[0], e.y - pointsMoveOriginalCoords[1])
                pointsMoveOriginalCoords = (e.x, e.y)
                pointsMoved()

    else:
        x0 = selectionRectangleOriginalCoords[0]
        y0 = selectionRectangleOriginalCoords[1]
        x1 = e.x
        y1 = e.y
        canvas.coords(selectionRectangle, x0, y0, x1, y1)

        canvas.itemconfig( "selected", fill=COLOR_POINT )
        canvas.dtag("selected")
        canvas.addtag_overlapping("selected", x0, y0, x1, y1 )
        canvas.dtag(selectionRectangle, "selected")
        canvas.itemconfig( "selected", fill=COLOR_SELECTED )
def xyMovedFinished(e):
    global selectionRectangle

    canvas.delete(selectionRectangle)
    selectionRectangle = None
    pointsMoveOriginalCoords = None

def pointsMoved():
    global inputVector

    for p in canvas.find_withtag("selected"):
        i = (p - 1) * 2
        coords = canvas.coords(p)

        inputVector[0][i] = mapValue(coords[0] + POINT_RADIUS, 0, OUTPUT_RESOLUTION, -3, 3)
        inputVector[0][i+1] = mapValue(coords[1] + POINT_RADIUS, 0, OUTPUT_RESOLUTION, -3, 3)

    updateImage(inputVector)

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
    pointsSaved.append( np.copy( inputVector ) )

def onRemovePoints():
    global pointsSaved

    pointList.delete(0,tk.END)
    pointsSaved = []

def onListClicked(e):
    updateImage( pointsSaved[pointList.curselection()[0]] )
    drawAllPointsPair()

def drawAllPointsPair():
    needToCreate = True
    if len(canvas.find_all()) > 0:
        needToCreate = False
        # canvas.itemconfig( "selected", fill = COLOR_POINT )
        # canvas.dtag( "selected" )

    for p in range(0, inputVector[0].shape[0] - 1, 2):
        x = mapValue(inputVector[0][p], -3,3, 0, OUTPUT_RESOLUTION)
        y = mapValue(inputVector[0][p+1], -3,3, 0, OUTPUT_RESOLUTION)

        if needToCreate:
            canvas.create_oval(x-POINT_RADIUS,y-POINT_RADIUS,x+POINT_RADIUS,y+POINT_RADIUS, fill=COLOR_POINT)
        else:
            canvas.coords( int(p/2) + 1, x-POINT_RADIUS,y-POINT_RADIUS,x+POINT_RADIUS,y+POINT_RADIUS )


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
    videoFilename = simpledialog.askstring("Input", "Video filename:",
                                parent=root, initialvalue="out")

    if videoFilename is None:
        return

    videoFilename += ".mp4"

    cantInterpolation = sliderTransition.get()
    totalFrames = cantInterpolation * pointList.size()

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

        arrInterpolation = f( np.linspace(0,1,cantInterpolation+1, endpoint = True) )

        for i in range(0,len(arrInterpolation)):
            if pointTo == 0 and i == len(arrInterpolation) - 1 :
                break

            latentSample = np.array([arrInterpolation[i]])
            generated = generateFromGAN( latentSample )
            generatedImage = Image.fromarray(generated[0], 'RGB')
            currentFrame = pointFrom*cantInterpolation+i+1
            generatedImage.save( "%s/%05d.png" % (PATH_RESULT,currentFrame) )

            progressBar['value'] = (currentFrame/totalFrames)*100
            root.update_idletasks()

    subprocess.call(["./imagesToVideo.sh"])
    os.rename("out.mp4", videoFilename)
    subprocess.call(["vlc", videoFilename])
        # generatedImages = generateFromGAN( arrInterpolation )
        # generatedPhotos = []
        #
        # for g in generatedImages:
        #     photo = ImageTk.PhotoImage(image=Image.fromarray(g, 'RGB'))
        #     generatedPhotos.append(photo)
        #
        # showFrame(generatedPhotos, 0)
def onSaveStill():
    global recordingCurrentFrame
    arrayImage.save( "%s/%05d.png" % (PATH_RESULT,recordingCurrentFrame) )
    recordingCurrentFrame += 1

def onRecord():
    global recording
    if not recording:
        recording = True
        btnRecord.config(text="Recording...")
    else :
        recording = False
        btnRecord.config(text="Record")

def onRandomClick():
    pointsMoveOriginalCoords = None
    updateImage()

def onLoadFile():
    global generator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION, pointsSaved

    filePath = filedialog.askopenfilename(initialdir = PATH_LOAD_FILE, title = "Select file")
    with open( filePath, 'rb' ) as file:
        _, _, generator = pickle.load(file)
        SIZE_LATENT_SPACE = int( generator.list_layers()[0][1].shape[1] )
        OUTPUT_RESOLUTION = int( generator.list_layers()[-1][1].shape[2] )

        root.title('PGAN Generator - %s' % filePath )

        if canvas:
            canvas.delete("all")
        if pointList:
            pointList.delete(0,tk.END)
        pointsSaved = []

root = tk.Tk()
init()

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Load file", command=onLoadFile)
menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)

# sliderX = tk.Scale(root, from_=0, to=SIZE_LATENT_SPACE-1, length=OUTPUT_RESOLUTION, resolution=1, orient=tk.HORIZONTAL)
# # sliderX.bind("<B1-Motion>", sliderMoved)
# sliderX.grid(row=1,column=1)
# sliderY = tk.Scale(root, from_=0, to=SIZE_LATENT_SPACE-1, length=OUTPUT_RESOLUTION, resolution=1)
# sliderY.set(1)
# # sliderY.bind("<B1-Motion>", sliderMoved)
# sliderY.grid(row=0,column=0)

canvas = tk.Canvas(root,width=OUTPUT_RESOLUTION, height=OUTPUT_RESOLUTION, bg="#000000", cursor="cross")
canvas.grid(row=0,column=1)
canvas.bind("<B1-Motion>", xyMoved)
canvas.bind("<ButtonRelease-1>", xyMovedFinished)
canvas.bind("<Button 3>", xyClicked)

arrayImage = generateImage()
photo = ImageTk.PhotoImage(image=arrayImage)
drawAllPointsPair()
mainImage = Label(root, image=photo)
mainImage.image = photo #esto es necesario por el garbage
# mainImage.pack( padx = SIZE_LATENT_SPACE)
mainImage.grid(row=0,column=2)

btnRandom = tk.Button(root, text="Random", command=onRandomClick)
btnRandom.grid(row=1,column=2)

buttonsFrame = tk.Frame(root)
buttonsFrame.grid(row=1, column=1)
btnRecord = tk.Button(buttonsFrame, text="Record video", command=onRecord)
btnRecord.pack(side = LEFT)
btnSaveStill = tk.Button(buttonsFrame, text="Save still", command=onSaveStill)
btnSaveStill.pack(side = LEFT)

pointsFrame = tk.Frame(root)
pointsFrame.grid(row=0,column=3)

btnAddPoint = tk.Button(pointsFrame, text= "Add point", command=onAddPoint)
btnAddPoint.pack()

pointList = tk.Listbox(pointsFrame, height = 40)
pointList.pack()
pointList.bind("<Double-Button-1>", onListClicked)

btnRmPoints = tk.Button(pointsFrame, text= "Remove points", command=onRemovePoints)
btnRmPoints.pack()

progressBar = ttk.Progressbar(pointsFrame,orient=HORIZONTAL,length=100,mode='determinate')
progressBar.pack()

btnSaveVideo = tk.Button(root, text= "Save video", command=onSaveVideo)
btnSaveVideo.grid(row=1, column=3)

sliderTransition = tk.Scale(root, from_=0, to=1000, length=OUTPUT_RESOLUTION, resolution=1)
sliderTransition.set(300)
sliderTransition.grid(row=0,column=4)

lblTransition = tk.Label(root, text='Transition')
lblTransition.grid(row=1,column=4)

root.mainloop()
