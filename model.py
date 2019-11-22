import math
import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.utils import shuffle

# ---------------------------------------------------------------------------------------------------------------------------
#
# Pre-processing
# 
# ---------------------------------------------------------------------------------------------------------------------------

# Read driving log
lines = []
#with open('driving_log.csv') as csvfile:
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for (i, line) in enumerate(reader):
        if i == 0:
            print ("Reading << driving_log.csv >> with header: ", line)
        else:
            lines.append(line)

# Split train/validation straight from lines read
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# Verify GPU mode
if not os.path.exists('/opt/carnd_p3/data/IMG/'):
#if not os.path.exists('./IMG/'):
    print("You are running in CPU mode only, switch to GPU.")
    exit()
            
images = []
measurements = []

# Index of driving_log line stuff 
CENTER_IMAGE = 0
LEFT_IMAGE   = 1
RIGHT_IMAGE  = 2
STEERING     = 3
THROTTLE     = 4
BRAKE        = 5
SPEED        = 6

# Generator for images
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                #name = './IMG/' + batch_sample[0].split('/')[-1]
                name = '/opt/carnd_p3/data/IMG/' + batch_sample[ CENTER_IMAGE ].split('/')[-1]
    
                # BGR 2 RGB
                #center_image = cv2.imread(name)
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[ STEERING ])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size = 32

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320  # Non-Trimmed image format

# Process driving_log and load images
def load_lines():
    for line in lines:
        # Source of center image
        source_path = line[ CENTER_IMAGE ]

        # Get filename without path - equivalent to << source_path_split('/')[-1] >>
        filename = os.path.basename(source_path) 

        # Get current path for training
        current_path = '/opt/carnd_p3/data/IMG/' + filename

        # Read center image
        image = cv2.imread(current_path)

        # Append to current list of mages
        images.append(images)

        # Get steering angle from file
        measurement = float(line[ STEERING ])

        # Append to our list of measurements
        measurements.append(measurement)

        print("Read image ", source_path)

    # Generate training data
    X_train = np.array(images)
    y_train = np.array(measurements)
    print("Generated training data")

# ---------------------------------------------------------------------------------------------------------------------------
#
# Model
# 
# ---------------------------------------------------------------------------------------------------------------------------

# Regression basic images and NN
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

# Generate model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,
#        input_shape  = (ch, row, col),
#        output_shape = (ch, row, col)))
        input_shape  = (row, col, ch),
        output_shape = (row, col, ch)))
#model.add(Flatten(input_shape = (160, 320, 3)))
#model.add(Flatten())
model.add(Convolution2D(filters = 6, kernel_size = (5, 5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(filters = 16, kernel_size = (5, 5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
print("Generated model")

model.compile(loss = 'mse', optimizer = 'adam')
print("Compiled model.")
    
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7, verbose = 1)
#print("Fit model to data.")
model.fit_generator(train_generator, 
            steps_per_epoch  = math.ceil(len(train_samples) / batch_size), 
            validation_data  = validation_generator, 
            validation_steps = math.ceil(len(validation_samples) / batch_size), 
            epochs = 5, verbose = 1)
print("Fit model to data using generator.")

model.save('model.h5')
print("Saved << model.h5 >>")