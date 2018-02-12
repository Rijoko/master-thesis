'''
Created on 04.02.2018

@author: Rijoko
'''

import tensorflow as tf

def loadImage(fileName):
    imageString = tf.read_file(fileName)
    imageDecoded = tf.image.decode_png(imageString)
    #imageResized = tf.image.resize_images(imageDecoded, size=[270, 480])
    return imageDecoded

def saveImage(image, currentdir, filename):
    image = tf.image.encode_png(image)
    filename = tf.constant(currentdir + filename + ".png")
    return tf.write_file(filename, image)

def loadImages(fileNames):
    images = tf.map_fn(loadImage, fileNames, dtype=tf.uint8)
    return images