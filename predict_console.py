import os
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow import keras
from keras import layers

import datetime

import nibabel as nib

from scipy import ndimage
import threading

import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()

# кол-во потоков
num_threads = 4

if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("logs"):
    os.makedirs("logs")

now = datetime.datetime.now()
time = str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + str(now.second)
log = open("logs/log_" + time + ".csv", "w+")

class Scan:
    def __init__(self, number, file, data):
        self.number = number
        self.file = file
        self.data = data


class ProcThread(threading.Thread):
    def __init__(self, index, data):
        threading.Thread.__init__(self)
        self.index = index
        self.data = data
        self.result = None

    def run(self):
        self.result = [process_scan(scan, self.data.index(scan) + 1, len(self.data), self.index) for scan in self.data]

    def join(self):
        threading.Thread.join(self)
        return self.result


def getFiles(paths):
    threads = []
    files = []
    parts = int(len(paths) / num_threads)
    print(parts)
    scans = []
    for i in range(len(paths)):
        scan = Scan(i + 1, paths[i], None)
        scans.append(scan)
    for i in range(num_threads):
        if i != num_threads-1:
            thread = ProcThread(i + 1, scans[parts * i:parts * i + parts])
        else:
            thread = ProcThread(i + 1, scans[parts * i:])
        thread.start()
        threads.append(thread)
    for t in threads:
        results = t.join()
        for r in results:
            files.append(r)
    return files


def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def convert(path):
    r_path = "temp"+'\\'+path.rpartition('\\')[2]
    dicom2nifti.dicom_series_to_nifti(path, r_path, reorient_nifti=True)
    return r_path+".nii"

def process_scan(scan, index, size, thread):
    study = scan.file.rpartition('\\')[2]
    print("Study: " + str(index) + "/" + str(size) + " - " + study + " - " + str(thread))
    nifti_path = convert(scan.file)
    volume = read_nifti_file(nifti_path)
    #scan.file = study[:study.index(".")]
    scan.file = study
    volume = normalize(volume)
    volume = resize_volume(volume)
    scan.data = volume
    os.remove(nifti_path)
    return scan


# abnormal_paths = [
#     os.path.join(os.getcwd(), "D:\\niftidataset\\datasets", x)
#     for x in os.listdir("D:\\niftidataset\\datasets")
# ]
#
# normal_paths = [
#     os.path.join(os.getcwd(), "D:\\niftidataset\\datasets_converted", x)
#     for x in os.listdir("D:\\niftidataset\\datasets_converted")
# ]

paths = [
    os.path.join(os.getcwd(), "D:\\niftidataset\\exported", x)
    for x in os.listdir("D:\\niftidataset\\exported")
]

print("CT scans: " + str(len(paths)))

scans_object = getFiles(paths[:10])

import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

print("Result: ")
log.write("index,study,n_chance,ab_chance\n")
for i in range(len(scans_object)):
    model.load_weights("3d_image_classification.h5")
    prediction = model.predict(np.expand_dims(scans_object[i].data, axis=0))[0]
    scores = [1 - prediction[0], prediction[0]]
    print(str(i + 1) + " - " + scans_object[i].file + " - ")
    log.write(str(i + 1) + "," + scans_object[i].file + ",")
    log.write(str(1-prediction[0])+","+str(prediction[0])+"\n")
    print(str(1 - prediction[0]) + " - " + str(prediction[0]))
log.close()
