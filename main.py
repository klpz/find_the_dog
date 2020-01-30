import sys
from keras import Sequential, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Conv1D, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
import os
from matplotlib.image import imread
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from keras.preprocessing.image import ImageDataGenerator
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


def loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


"""
im = cv2.imread('./IMG_0431.jpg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()
"""

# define location of dataset
dataset_home = './'
subdirs = ['train_dataset/', 'test_dataset/']

"""

for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dir_dogs/', 'dir_cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)


seed(1)
val_ratio = 0.25

for file in listdir(subdirs[0]):
    src = subdirs[0] + file
    print(src)
    dst_dir = subdirs[0]
    if random() < val_ratio:
        dst_dir = subdirs[1]
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'dir_cats/' + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dir_dogs/' + file
        copyfile(src, dst)
"""

model = define_model()
data_generator = ImageDataGenerator(rescale=1.0/255.0)

train_set = data_generator.flow_from_directory(subdirs[0], class_mode='binary', batch_size=64, target_size=(200, 200))
test_set = data_generator.flow_from_directory(subdirs[1], class_mode='binary', batch_size=64, target_size=(200, 200))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit_generator(train_set, steps_per_epoch=len(train_set), validation_data=test_set,
                              validation_steps=len(test_set), epochs=50, verbose=0)


_, acc = model.evaluate_generator(test_set, steps=len(test_set), verbose=0)
print('> %.3f' % (acc * 100.0))


summarize_diagnostics(history)
loss_plot(history)

y_predicted = model.predict(test_set)
print(y_predicted)