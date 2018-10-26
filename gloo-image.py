#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
from glumpy import app, gloo, gl
import numpy as np
import tensorflow as tf
import PIL.Image
import pickle
from scipy.interpolate import interp1d


vertex = """
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
"""

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
"""

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

def initNeuralActivities():
    global generator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION, arrInterpolation
    
    tf.InteractiveSession()
    with open( "/media/macramole/stuff/Data/pgan/011-pgan-cuadrosPiel-preset-v2-1gpu-fp32-latent512-lod512-norepeat/network-snapshot-000840.pkl", 'rb' ) as file:
        _, _, generator = pickle.load(file)
        SIZE_LATENT_SPACE = int( generator.list_layers()[0][1].shape[1] )
        OUTPUT_RESOLUTION = int( generator.list_layers()[-1][1].shape[2] )
    
    x = np.array([0,1])
    y = np.vstack((np.random.normal(0, 1, (1, SIZE_LATENT_SPACE)),np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))))
    f = interp1d( x , y, axis = 0  )

    cantInterpolation = 500
    arrInterpolation = f( np.linspace(0,1,cantInterpolation+1, endpoint = True) )

def updateImage():
    global quad, currentImage
#    arrImage = generateFromGAN( np.random.normal(0, 1, (1, SIZE_LATENT_SPACE)) )
    arrImage = generateFromGAN( np.array([arrInterpolation[currentImage]]) )
    arrImage = np.ascontiguousarray(arrImage[0])    
    quad['texture'] = np.ascontiguousarray(arrImage)
    currentImage += 1

initNeuralActivities()
currentImage = 0

window = app.Window(width=512, height=512, aspect=1)

@window.event
def on_draw(dt):
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
    updateImage()
    
quad = gloo.Program(vertex, fragment, count=4)
quad['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
quad['texcoord'] = [( 0, 1), ( 0, 0), ( 1, 1), ( 1, 0)]
updateImage()

app.run()
