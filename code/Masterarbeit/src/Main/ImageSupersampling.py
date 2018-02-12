'''
Created on 27.01.2018

@author: Rijoko
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import Main.ImageSerialization as ImageSerialization
import Main.Settings as Settings

def showImage(image):
    plt.imshow(image)
    plt.show()

FromImageNum = 7100
ToImageNum = 7231
BatchSize = 100

# Load images into pipeline
fileNameList = tf.data.Dataset.list_files(Settings.DirPathBig + "big_*.png")
fileNameList = fileNameList.skip(FromImageNum)
fileNameList = fileNameList.take(ToImageNum-FromImageNum + 1)

dataSet = tf.data.Dataset.zip((fileNameList, fileNameList.map(ImageSerialization.loadImage)))

batchedDataSet = dataSet.batch(BatchSize)
batchedDataSetIterator = batchedDataSet.make_one_shot_iterator()
nextElement = batchedDataSetIterator.get_next()

fileNames, originalImages = nextElement
originalImagesCasted = tf.cast(originalImages, dtype=tf.float32)

# Modifiy the image
subsampling = tf.nn.avg_pool(originalImagesCasted, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

finalImages = (tf.cast(subsampling, dtype=tf.uint8), originalImages, fileNames)

print("Start converting...")
with tf.Session() as session:
    for counter in range(FromImageNum, ToImageNum + 1, BatchSize):
        print("Iteration " + (counter).__str__())
            
        try:
            modified, originals, fileNames = session.run(finalImages)
            print (fileNames)
        except tf.errors.OutOfRangeError:
            break
        
        print("Save images...")
        for i, image in enumerate(modified):
            #showImage(image)
            filename = "big_SS_"+str(counter + i).zfill(4)
            session.run(ImageSerialization.saveImage(image, Settings.DirPathBigSS, filename))
        print("Done saving.")
        
print("Done converting.")