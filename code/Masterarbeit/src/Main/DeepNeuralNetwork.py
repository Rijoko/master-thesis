'''
Created on 04.02.2018

@author: Rijoko
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import Main.ImageSerialization as ImageSerialization
import Main.Settings as Settings


def loadImageDataSet(filePattern, fromNum, toNum, batchSize):
    ds = tf.data.Dataset.list_files(filePattern)
    ds = ds.skip(fromNum)
    ds = ds.take(toNum - fromNum + 1)
    ds = ds.batch(batchSize)
    ds = tf.data.Dataset.zip((ds, ds.map(ImageSerialization.loadImages)))
    return ds
    
def showImage(image):
    plt.imshow(image)
    plt.show()

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

FromImageNum = 0
ToImageNum = 7231
BatchSize = 100

# Load images into pipeline
origImageDS = loadImageDataSet(Settings.DirPathBig + "big_*.png", FromImageNum, ToImageNum, BatchSize)

nextElement = origImageDS.make_one_shot_iterator().get_next()

fileNames, originalImages = nextElement

originalImagesCasted = tf.cast(originalImages, dtype=tf.float32)

# Modifiy the image

# Conv Layer
W_conv1 = weightVariable([5, 5, 4, 8])
b_conv1 = biasVariable([8])
h_conv1 = tf.nn.relu(conv2d(originalImagesCasted, W_conv1) + b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 480 * 270 * 4 * 8])

# Fully Connected Layer
W_fc1 = weightVariable([480 * 270 * 4 * 8, 16])
b_fc1 = biasVariable([16])

h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


identityImages = (tf.cast(originalImagesCasted, dtype = tf.uint8), originalImages, fileNames)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for counter in range(FromImageNum, ToImageNum + 1, BatchSize):
        print("Iteration " + (counter).__str__())
            
        try:
            modified, originals, fileNames = session.run(identityImages)
            #print(session.run(h_fc1_drop, feed_dict={ keep_prob : 0.95 }))
            showImage(modified[0])
            showImage(originals[0])
        except tf.errors.OutOfRangeError:
            break
        '''
        print("Save images...")
        for i, image in enumerate(modified):
            #showImage(image)
            filename = "big_SS_"+str(counter + i).zfill(4)
            session.run(ImageSerialization.saveImage(image, Settings.DirPathBigSS, filename))
        print("Done saving.")
        '''