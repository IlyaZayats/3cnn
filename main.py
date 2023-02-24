
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers

import nibabel as nib

import random
from scipy import ndimage

# import matplotlib.pyplot as plt
import threading

class ProcThread (threading.Thread):
    def __init__(self, index, data):
        threading.Thread.__init__(self)
        self.index = index
        self.data = data
        self.result = None
    def run(self):
        #print(str(len(self.data)) + "\n")
        self.result = np.array([process_scan(path, self.data.index(path)+1, len(self.data), self.index) for path in self.data])
    def join(self):
        threading.Thread.join(self)
        return self.result

def getFiles(paths):
    threads = []
    files = []
    parts = int(len(paths)/5)
    print(parts)
    for i in range(5):
        if i != 4:
            thread = ProcThread(i + 1, paths[parts * i:parts * i + parts])
        else:
            thread = ProcThread(i + 1, paths[parts * i:])
        thread.start()
        threads.append(thread)
    for t in threads:
        results = t.join()
        for r in results:
            files.append(r)
    return files

def read_nifti_file(filepath):
    # Читаем файл
    scan = nib.load(filepath)
    # Получаем необработаннные данные
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    # Нормализуем объем
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    # Устанавливаем желаемую глубину
    desired_depth = 64
    desired_width = 128
    desired_height = 256
    # Получаем текущую глубину
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Вычисляем коэффициент глубины
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Поворачиваем
    img = ndimage.rotate(img, 90, reshape=False)
    # Меняем размер вдоль оси Z
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    i = 255
    # Первые 85
    while i > 84:
        img = np.delete(img, i, 1)
        i -= 1
    return img


def process_scan(path, index, size, thread):
    study = path.rpartition('\\')[2]
    print("Study: " + str(index) + "/" + str(size) + " - " + str(study[:study.index(".")]) + " - " + str(thread))
    # Читаем файл
    volume = read_nifti_file(path)
    # Нормализуем
    volume = normalize(volume)
    # Меняем ширину, высоту и глубину
    volume = resize_volume(volume)
    return volume


"""
Let's read the paths of the CT scans from the class directories.
"""
# normal = input("Input path to normal scans dir: ")
# abnormal = input("Input path to abnormal scans dir: ")
# # Folder "CT-0" consist of CT scans having normal lung tissue,
# # no CT-signs of viral pneumonia.
# normal_scan_paths = [
#     os.path.join(os.getcwd(), normal, x)
#     for x in os.listdir(normal)
# ]
# # Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# # involvement of lung parenchyma.
# abnormal_scan_paths = [
#     os.path.join(os.getcwd(), abnormal, x)
#     for x in os.listdir(abnormal)
# ]

# normalp = ''
# for x in normal_scan_paths:
#     if x.rpartition('/')[2] == '.DS_Store':
#         normalp = x
#         break
#
# abnormalp = ''
# for x in abnormal_scan_paths:
#     if x.rpartition('/')[2] == '.DS_Store':
#         abnormalp = x
#         break
#
# if (not normalp==''):
#     normal_scan_paths.remove(normalp)
# if (not abnormalp==''):
#     abnormal_scan_paths.remove(abnormalp)


# Создаём списки с путями к файлам
# Датасет с аневризмами
abnormal_paths = [
    os.path.join(os.getcwd(), "D:\\3dcnn_datasets\\nifti_ab_front", x)
    for x in os.listdir("D:\\3dcnn_datasets\\nifti_ab_front")
]
# Датасет без аневризмов
normal_paths = [
    os.path.join(os.getcwd(), "D:\\3dcnn_datasets\\nifti_n", x)
    for x in os.listdir("D:\\3dcnn_datasets\\nifti_n")
]

# Перемешиваем списки
np.random.shuffle(abnormal_paths)
np.random.shuffle(normal_paths)

# Берем по 100 первых путей из датасетов
# normal_scan_paths = normal_paths[:10]
# abnormal_scan_paths = abnormal_paths[:10]

normal_scan_paths = normal_paths[:80]
abnormal_scan_paths = abnormal_paths[:80]

print("CT normal scans: " + str(len(normal_scan_paths)))
print("CT abnormal scans: " + str(len(abnormal_scan_paths)))

print("Processing abnormal scans: ")
abnormal_scans = np.array(getFiles(abnormal_scan_paths))
print(len(abnormal_scans))
print("\n")

print("Processing normal scans: ")
normal_scans = np.array(getFiles(normal_scan_paths))
print(len(normal_scans))
print("\n")
# Читаем и обрабатываем файлы.
# Каждый скан изменен по высоте, ширине, и глубина перемасштабирована

# abnormal_scans = np.array([process_scan(path, abnormal_scan_paths.index(path ) +1, len(abnormal_scan_paths))
#                            for path in abnormal_scan_paths])


# normal_scans = np.array([process_scan(path, normal_scan_paths.index(path ) +1, len(normal_scan_paths)
#                                       ) for path in normal_scan_paths])
# print("\n")

# Для слоев из датасета с аневризмами указываем 1
# Для слоев без - 0
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Разделяем данные в соотнощшении 70 к 30 для обучения и валидации

x_train = np.concatenate((abnormal_scans[:56], normal_scans[:56]), axis=0)
y_train = np.concatenate((abnormal_labels[:56], normal_labels[:56]), axis=0)
x_val = np.concatenate((abnormal_scans[56:], normal_scans[56:]), axis=0)
y_val = np.concatenate((abnormal_labels[56:], normal_labels[56:]), axis=0)

# x_train = np.concatenate((abnormal_scans[:7], normal_scans[:7]), axis=0)
# y_train = np.concatenate((abnormal_labels[:7], normal_labels[:7]), axis=0)
# x_val = np.concatenate((abnormal_scans[7:], normal_scans[7:]), axis=0)
# y_val = np.concatenate((abnormal_labels[7:], normal_labels[7:]), axis=0)

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

"""
## Data augmentation
The CT scans also augmented by rotating at random angles during training. Since
the data is stored in rank-3 tensors of shape `(samples, height, width, depth)`,
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on
the data. The new shape is thus `(samples, height, width, depth, 1)`. There are
different kinds of preprocessing and augmentation techniques out there,
this example shows a few simple ones to get started.
"""
@tf.function
def rotate(volume):

    def scipy_rotate(volume):
        # определяем несколько углов поворота
        angles = [-20, -10, -5, 5, 10, 20]
        # берем случаный
        angle = random.choice(angles)
        # поворачиваем объем
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    # тренировочный подсет
    # поворачиваем объем
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    # валидационный подсет
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


"""
While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.
"""

# Определим загрузчики данных.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Аугментируем данные в процессе обучения.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Не аугментируем, только меняем размер.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

"""
Visualize an augmented CT scan.
"""



# data = train_dataset.take(1)
# images, labels = list(data)[0]
# images = images.numpy()
# image = images[0]
# print("Dimension of the CT scan is:", image.shape)
# plt.imshow(np.squeeze(image[:, :, 3]), cmap="gray")


"""
Since a CT scan has many slices, let's visualize a montage of the slices.
"""


# def plot_slices(num_rows, num_columns, width, height, data):
#     """Plot a montage of 20 CT slices"""
#     data = np.rot90(np.array(data))
#     data = np.transpose(data)
#     data = np.reshape(data, (num_rows, num_columns, width, height))
#     rows_data, columns_data = data.shape[0], data.shape[1]
#     heights = [slc[0].shape[0] for slc in data]
#     widths = [slc.shape[1] for slc in data[0]]
#     fig_width = 12.0
#     fig_height = fig_width * sum(heights) / sum(widths)
#     f, axarr = plt.subplots(
#         rows_data,
#         columns_data,
#         figsize=(fig_width, fig_height),
#         gridspec_kw={"height_ratios": heights},
#     )
#     for i in range(rows_data):
#         for j in range(columns_data):
#             axarr[i, j].imshow(data[i][j], cmap="gray")
#             axarr[i, j].axis("off")
#     plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#     plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
# plot_slices(4, 10, 128, 128, image[:, :, :40])

"""
## Define a 3D convolutional neural network
To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).
"""
def get_model(width=128, height=85, depth=64):

    # Определяем входные данные.
    inputs = keras.Input((width, height, depth, 1))

    # Определяем сверточные слои с фукнцией макс пулинга
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

    # Определяем модель.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Создаем модель.
model = get_model(width=128, height=85, depth=64)
model.summary()

# Компилируем модель.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Определяем обратный вызов.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Обучаем модель, валидируя в конце каждой итерации(epoch)
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

"""
It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in the results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.
"""

"""
## Visualizing model performance
Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.
"""

# fig, ax = plt.subplots(1, 2, figsize=(20, 3))
# ax = ax.ravel()
#
# for i, metric in enumerate(["acc", "loss"]):
#     ax[i].plot(model.history.history[metric])
#     ax[i].plot(model.history.history["val_" + metric])
#     ax[i].set_title("Model {}".format(metric))
#     ax[i].set_xlabel("epochs")
#     ax[i].set_ylabel(metric)
#     ax[i].legend(["train", "val"])

"""
## Make predictions on a single CT scan
"""

# Load best weights.
# for i in range(len(x_val)):
#     model.load_weights("3d_image_classification.h5")
#     prediction = model.predict(np.expand_dims(x_val[i], axis=0))[0]
#     scores = [1 - prediction[0], prediction[0]]
#
#     class_names = ["normal", "abnormal"]
#     print("Scan", i, ":\n")
#     for score, name in zip(scores, class_names):
#         print(
#             "This model is %.2f percent confident that CT scan is %s"
#             % ((100 * score), name)
#         )
