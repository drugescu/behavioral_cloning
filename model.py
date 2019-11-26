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
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[ STEERING ])
                
                # Trim image upper side
                FROM_TOP = 60
                FROM_BOTTOM = 20
                images.append(center_image[0 + FROM_TOP : 160 - FROM_BOTTOM, 0 : 320, 0 : 3])

                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Second variant of the generator - load images with left and right
# Generator for images
# NVIDIA
FROM_TOP = 60
FROM_BOTTOM = 20

new_size_x, new_size_y = 200, 66
old_size_x, old_size_y = 320, 160

#first = 0

def generator_all(samples, batch_size = 32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            
            for (k, batch_sample) in enumerate(batch_samples):
                #name = './IMG/' + batch_sample[0].split('/')[-1]
                #name = "./IMG/center_2016_12_01_13_30_48_287.jpg"
                name = '/opt/carnd_p3/data/IMG/' + batch_sample[ CENTER_IMAGE ].split('/')[-1]
                name_left  = '/opt/carnd_p3/data/IMG/' + batch_sample[  LEFT_IMAGE ].split('/')[-1]
                name_right = '/opt/carnd_p3/data/IMG/' + batch_sample[ RIGHT_IMAGE ].split('/')[-1]
                #print("Generator: Processing batch_sample ", k, "...")
    
                # BGR 2 RGB
                center_image = cv2.cvtColor(cv2.imread(name      ), cv2.COLOR_BGR2RGB)
                left_image   = cv2.cvtColor(cv2.imread(name_left ), cv2.COLOR_BGR2RGB)
                right_image  = cv2.cvtColor(cv2.imread(name_right), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[ STEERING ])
                
                # Adjust angles
                correction = 0.25 # this is a parameter to tune
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                
                # Trim image upper side
                #images.append(center_image[80 : 160, 0 : 320, 0 : 3])
                icenter = cv2.resize(center_image[0 + FROM_TOP : old_size_y - FROM_BOTTOM, 0 : old_size_x, 0 : 3], (new_size_x, new_size_y), interpolation = cv2.INTER_AREA)
                images.append(icenter)
                angles.append(center_angle)
                #if (first == 0):
                #    cv2.imwrite('./IMG/center_cropped_new.jpg', images[-1]) 
                #    print("cropped")
                
                # Append left and right as well - angle remains the same, should make fr a more robust training solution
                ileft = cv2.resize(  left_image[0 + FROM_TOP : old_size_y - FROM_BOTTOM, 0 : old_size_x, 0 : 3], (new_size_x, new_size_y), interpolation = cv2.INTER_AREA)
                images.append(ileft)
                angles.append(steering_left)
                
                iright = cv2.resize( right_image[0 + FROM_TOP : old_size_y - FROM_BOTTOM, 0 : old_size_x, 0 : 3], (new_size_x, new_size_y), interpolation = cv2.INTER_AREA)
                images.append(iright)
                angles.append(steering_right)
                
                # Now append images flipped to account for bias
                images.append(cv2.flip(ileft, 1))
                angles.append(-steering_left)
                images.append(cv2.flip(iright, 1))
                angles.append(-steering_right)
                images.append(cv2.flip(icenter, 1))
                angles.append(-center_angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size = 32

# Compile and train the model using the generator function
train_generator      = generator_all(train_samples, batch_size = batch_size)
validation_generator = generator_all(validation_samples, batch_size = batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format
#ch, row, col = 3, 160, 320  # Non-Trimmed image format
ch, row, col = 3, new_size_y, new_size_x  # Trimmed image format - nVidia

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
    print("Generated training data.")

# ---------------------------------------------------------------------------------------------------------------------------
#
# Model
# 
# ---------------------------------------------------------------------------------------------------------------------------

# Regression basic images and NN
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

# Generate model LeNet
def generate_le_net():
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 1.0,
            input_shape  = (row, col, ch),
            output_shape = (row, col, ch)))
            #(160, 320, 3)))
    model.add(Convolution2D(filters = 6, kernel_size = (5, 5), activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(filters = 16, kernel_size = (5, 5), activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    print("Generated model - LeNet, normalized, ADAM, MSE.")

    model.compile(loss = 'mse', optimizer = 'adam')
    print("Compiled model.")
    
    return model

# Generate model nVidia - http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def generate_nvidia():
    model = Sequential()
    #model.add(Cropping2D(cropping = ((50, 20), (0, 0)), input_shape = (row, col, ch)))
    
    print("row, col, ch = ", (row, col, ch))
    
    # Lambda normalization layer
    model.add(Lambda(lambda x: x/127.5 - 1.0,
            input_shape  = (row, col, ch),
            output_shape = (row, col, ch), name = 'normalization'))
            #(66, 200, 3)))

    # nVidia model - Conv layers
    # Filters, kernel, padding, strides
    model.add(Convolution2D(filters = 24, kernel_size = (5, 5), padding = 'valid', strides = (2,2), name = 'conv2d_1'))
    model.add(ELU(name = 'elu_1'))
    model.add(Convolution2D(filters = 36, kernel_size = (5, 5), padding = 'valid', strides = (2,2), name = 'conv2d_2'))
    model.add(ELU(name = 'elu_2'))
    model.add(Convolution2D(filters = 48, kernel_size = (5, 5), padding = 'valid', strides = (2,2), name = 'conv2d_3'))
    model.add(ELU(name = 'elu_3'))
    model.add(Convolution2D(filters = 64, kernel_size = (3, 3), padding = 'valid', strides = (1,1), name = 'conv2d_4'))
    model.add(ELU(name = 'elu_4'))
    model.add(Convolution2D(filters = 64, kernel_size = (3, 3), padding = 'valid', strides = (1,1), name = 'conv2d_5'))
    model.add(ELU(name = 'elu_5'))
    
    # Flatten and pass to FC layers
    model.add(Flatten()) # 1161 neurons
    model.add(Dense(100, name = 'fc_1'))
    model.add(Dropout(0.3, name = 'dropout_1'))
    model.add(ELU(name = 'elu_6'))
    model.add(Dense(50, name = 'fc_2'))
    model.add(Dropout(0.3, name = 'dropout_2'))
    model.add(ELU(name = 'elu_7'))
    model.add(Dense(10, name = 'fc_3'))
    model.add(Dropout(0.3, name = 'dropout_3'))
    model.add(ELU(name = 'elu_8'))
    model.add(Dense(1, name = 'output', activation = 'tanh'))
    print("Generated model.")

    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
    print("Compiled model - nVidia, ADAM, MSE, nomalized, metrics = accuracy, dropouts = 0.3, (5 conv, 3 fc).")
    
    return model

# Choose the model here
model = generate_nvidia()
    
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7, verbose = 1)
#print("Fit model to data.")
model.fit_generator(train_generator, 
            steps_per_epoch  = math.ceil(len(train_samples) / batch_size), 
            validation_data  = validation_generator, 
            validation_steps = math.ceil(len(validation_samples) / batch_size), 
            epochs = 10, verbose = 1)
print("Fit model to data using generator.")

model.save('model.h5')
print("Saved << model.h5 >>.")