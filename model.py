import cv2
import csv
import numpy as np
import os

def getDrivingLogEntries(dataPath, skipHeader=False):
    """
    Returns the entries from driving log file in `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    entries = []
    with open(dataPath + '/driving_log_rel.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for row in reader:
            entries.append(row)
    return entries


def getImageDirectories(dataPath):
    """
    Returns the paths to images and measurements for all data recording folders in `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    recFolders = [x[0] for x in os.walk(dataPath)]
    recordings = list(filter(lambda recording: os.path.isfile(recording + '/driving_log_rel.csv'), recFolders))
    centerCam = []
    leftCam = []
    rightCam = []
    steerAngleMeas = []
    for recording in recordings:
        entries = getDrivingLogEntries(recording)
        center = []
        left = []
        right = []
        measurements = []
        for entry in entries:
            measurements.append(float(entry[3]))
            center.append(recording + entry[0].strip())
            left.append(recording + entry[1].strip())
            right.append(recording + entry[2].strip())
        centerCam.extend(center)
        leftCam.extend(left)
        rightCam.extend(right)
        steerAngleMeas.extend(measurements)

    return (centerCam, leftCam, rightCam, steerAngleMeas)

def correctMeasurements(center, left, right, steerMeasurements, correction):
    """
    Corrects the steering measurements for camera images 'left' and 'right' 
    using the correction factor `correction` and combines images & measurements
    Returns ([imageDir], [measurements])
    """
    imageDir = []
    imageDir.extend(center)
    imageDir.extend(left)
    imageDir.extend(right)
    measurements = []
    measurements.extend(steerMeasurements)
    measurements.extend([steering_center + correction for steering_center in steerMeasurements])
    measurements.extend([steering_center - correction for steering_center in steerMeasurements])
    return (imageDir, measurements)

import sklearn

def generator(samples, batch_size=32):
    """
    Generate the required images and measurements for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Get image directories & correct measurements for left and right camera images
centerDir, leftDir, rightDir, measurements = getImageDirectories('data')
imageDirectories, measurements = correctMeasurements(centerDir, leftDir, rightDir, measurements, 0.2)
print('Total Images: {}'.format( len(imageDirectories)))

# Split training & validation samples
from sklearn.model_selection import train_test_split
samples = list(zip(imageDirectories, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# Create generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create model (based on nVidia model)
model = nVidiaModel()

# Compile & train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

