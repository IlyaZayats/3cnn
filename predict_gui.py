import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QFileDialog, QLabel, QGridLayout, \
    QGroupBox, QListWidget, QListWidgetItem, QProgressBar, QListView, QLineEdit, QTableWidget, QHeaderView, \
    QTableWidgetItem
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from tensorflow import keras
from keras import layers

import random

import datetime

import nibabel as nib

from scipy import ndimage
import threading

import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()

mutex = QMutex()

class Scan:
    def __init__(self, number, file, data):
        self.number = number
        self.file = file
        self.data = data

class Worker(QObject):
    progress_worker = pyqtSignal()
    completed_worker = pyqtSignal(list)

    @pyqtSlot(int, list)
    def do_work_worker(self, index, scans):
        result = [self.process_scan(scan, scans.index(scan) + 1, len(scans), index) for scan in scans]
        mutex.lock()
        self.completed_worker.emit(result)
        mutex.unlock()

    def read_nifti_file(self, filepath):
        scan = nib.load(filepath)
        scan = scan.get_fdata()
        return scan

    def normalize(self, volume):
        min = -1000
        max = 400
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume

    def resize_volume(self, img):
        desired_depth = 64
        desired_width = 128
        desired_height = 256
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

        img_sectors = []
        i = 255
        temp = img
        #Первые 85
        while i > 84:
            temp = np.delete(temp,i,1)
            i -= 1
        img_sectors.append(temp)
        #Средние 85
        i = 255
        temp = img
        while i > 169:
            temp = np.delete(temp, i, 1)
            i -= 1
        i=84
        while i>-1:
            temp = np.delete(temp, 0, 1)
            i -= 1
        img_sectors.append(temp)
        temp = img
        #последние 85
        i=170
        while i > -1:
            temp = np.delete(temp, i, 1)
            i -= 1
        img_sectors.append(temp)

        # for i in range(3):
        #     print(len(img_sectors[i]))
        #     print(len(img_sectors[i][0]))
        #     print(len(img_sectors[i][0][0]))
        # print(img_sectors.ndim)
        return img_sectors

    def convert(self, path):
        r_path = "temp" + '\\' + path.rpartition('\\')[2]
        dicom2nifti.dicom_series_to_nifti(path, r_path, reorient_nifti=True)
        return r_path + ".nii"

    def process_scan(self, scan, index, size, thread):
        study = scan.file.rpartition('\\')[2]
        mutex.lock()
        print("Study: " + str(index) + "/" + str(size) + " - " + study + " - " + str(thread))
        mutex.unlock()
        nifti_path = self.convert(scan.file)
        volume = self.read_nifti_file(nifti_path)
        # scan.file = study[:study.index(".")]
        scan.file = study
        volume = self.normalize(volume)
        volume_sectors = self.resize_volume(volume)
        scan.data = volume_sectors
        os.remove(nifti_path)
        mutex.lock()
        self.progress_worker.emit()
        mutex.unlock()
        return scan

class Master(QObject):
    work_to_worker = pyqtSignal(int, list)
    progress_master = pyqtSignal()
    completed_master = pyqtSignal(list)

    predict_update = pyqtSignal(int, list)
    predict_finished = pyqtSignal()

    results = []
    paths = []
    parts_worker = []
    index_worker = 0

    def get_model(self, width=128, height=85, depth=64):
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

    @pyqtSlot(list)
    def predict(self, files):
        model = self.get_model(width=128, height=85, depth=64)
        model.summary()
        print("Result: ")
        for k in range(len(files)):
            temp = []
            for i in range(3):
                model.load_weights("front.h5")
                #print(len(files[k].data[i][0]))
                prediction = model.predict(np.expand_dims(files[k].data[i], axis=0))[0]
                temp.append(prediction[0])
            self.predict_update.emit(k, [files[k].file, temp[0], temp[1], temp[2]])
            self.progress_master.emit()
        self.predict_finished.emit()

    @pyqtSlot(list)
    def do_work_master(self, paths):
        self.results = []
        num_threads=4
        if len(paths)<num_threads:
            num_threads=len(paths)
        self.workers = []
        self.workers_threads = []
        for i in range(num_threads):
            self.workers.append(Worker())
            self.workers_threads.append(QThread())
        self.paths = paths
        parts = int(len(self.paths) / num_threads)
        scans = []
        for i in range(len(self.paths)):
            scan = Scan(i + 1, self.paths[i], None)
            scans.append(scan)
        for i in range(num_threads):

            self.workers[i].progress_worker.connect(self.update_progress_worker)
            self.workers[i].completed_worker.connect(self.completed_worker)
            self.work_to_worker.connect(self.workers[i].do_work_worker)
            self.workers[i].moveToThread(self.workers_threads[i])
            self.index_worker = i+1
            if i != num_threads-1:
                self.parts_worker = scans[parts * i:parts * i + parts]
            else:
                self.parts_worker = scans[parts * i:]
            self.workers_threads[i].start()
            self.work_to_worker.emit(self.index_worker, self.parts_worker)
            self.work_to_worker.disconnect()

    def update_progress_worker(self):
        self.progress_master.emit()

    def completed_worker(self, incoming_result):
        for item in incoming_result:
            self.results.append(item)
        if len(self.results) == len(self.paths):
            for thread in self.workers_threads:
                thread.exit()
            self.completed_master.emit(self.results)

class MainWindow(QMainWindow):

    work_to_master = pyqtSignal(list)
    start_predict = pyqtSignal(list)

    def __init__(self):
        super().__init__()

        if not os.path.exists("temp"):
            os.makedirs("temp")
        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.it = 1
        self.log = None
        self.setWindowTitle("Intracranial aneurysm predict")
        self.paths = []
        self.selected = set()

        self.mainWidget = QWidget()
        self.mainLayout = QGridLayout(self.mainWidget)

        selectWidget = QWidget()
        #self.mainLayout.addWidget(selectWidget, 0, 0, 1, 2)
        selectLayout = QGridLayout(selectWidget)
        self.selectAllButton = QPushButton("Выбрать всё", selectWidget)
        self.selectedLabel = QLabel("Выбрано: "+str(0), selectWidget)
        #self.selectedAmount = QLabel(str(0), selectWidget)
        selectLayout.addWidget(self.selectedLabel, 0, 0, 1, 1)
        #selectLayout.addWidget(self.selectedAmount, 0, 1, 1, 1)
        selectLayout.addWidget(self.selectAllButton, 0, 2, 1, 1)

        self.selectAllButton.clicked.connect(self.selectAllButtonClicked)

        gb = QGroupBox("Список исследований: ")
        self.mainLayout.addWidget(gb, 0, 0, 1, 2)
        glw = QGridLayout(gb)
        self.listWidget = QListWidget()
        self.listWidget.setFlow(QListView.Flow.TopToBottom)
        self.dirTitle = QLabel("", self.mainWidget)
        glw.addWidget(selectWidget, 0,0,1,3)
        glw.addWidget(self.listWidget, 1, 0, 1, 3)
        self.progressMsg = QLabel("")
        self.progressBar = QProgressBar(self.mainWidget)
        self.progressBar.setMinimum(0)
        self.mainLayout.addWidget(self.progressMsg, 1, 0, 1, 2)
        self.mainLayout.addWidget(self.progressBar, 2, 0, 1, 2)
        self.mainLayout.addWidget(self.dirTitle, 3, 0, 1, 1)

        self.selectButton = QPushButton("Выбрать корневую папку", self.mainWidget)
        self.mainLayout.addWidget(self.selectButton, 4, 0, 1, 1)
        self.confirmButton = QPushButton("Подтвердить", self.mainWidget)
        self.mainLayout.addWidget(self.confirmButton, 4, 1, 1, 1)

        self.selectAllButton.setDisabled(True)

        self.errorLabel = QLabel("", self.mainWidget)
        self.errorLabel.setStyleSheet("color : red; font : bold")
        #self.mainLayout.addWidget(self.errorLabel, 5, 0, 1, 2)
        glw.addWidget(self.errorLabel,2,0,1,3)
        self.selectButton.setCheckable(True)
        self.confirmButton.setDisabled(True)
        self.selectButton.clicked.connect(self.selectButtonClicked)
        self.confirmButton.clicked.connect(self.confirmeButtonClicked)

        self.master = Master()
        self.master_thread = QThread()

        self.master.progress_master.connect(self.update_progress_master)
        self.master.completed_master.connect(self.completed_master)

        self.master.predict_update.connect(self.predict_update)
        self.master.predict_finished.connect(self.predict_finished)
        self.start_predict.connect(self.master.predict)

        self.work_to_master.connect(self.master.do_work_master)

        self.master.moveToThread(self.master_thread)

        self.setCentralWidget(self.mainWidget)

        self.dicomDir = QLineEdit("")
        self.dicomDir.setVisible(False)
        self.dicomDir.textChanged.connect(self.dirSelected)

        self.selectedCount = QProgressBar()
        self.selectedCount.setVisible(False)
        self.selectedCount.setMinimum(0)
        self.selectedCount.setValue(0)

        self.selectedCount.valueChanged.connect(self.selectedValueChanged)

        self.listWidget.itemClicked.connect(self.listItemClicked)

        self.connectedSlots = True
        self.emptyError = False

        self.outputTable = QTableWidget(10, 4)
        self.outputTable.setShowGrid(True)
        table_headers = ["Исследование", "Лобная", "Височная", "Затылочная"]
        for i in range(len(table_headers)):
            item = QTableWidgetItem(table_headers[i])
            self.outputTable.setHorizontalHeaderItem(i, item)
        for i in range(10):
            for j in range(4):
                item = QTableWidgetItem("")
                #item.setFlags(Qt.ItemFlag.NoItemFlags)
                self.outputTable.setItem(i, j, item)
        self.outputTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.outputTable.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mainLayout.addWidget(self.outputTable, 0, 2, 5, 3)


    def selectAllButtonClicked(self):
        i = self.listWidget.count() - 1
        font = QFont(self.listWidget.item(0).font())
        font.setWeight(600)
        self.selected = set()
        while i > -1:
            self.listWidget.item(i).setFont(font)
            i -= 1
        for path in self.paths:
            self.selected.add(path.rpartition('\\')[2])
        self.selectedCount.setValue(len(self.selected))


    def selectButtonClicked(self):
        dir = QFileDialog.getExistingDirectory(self.centralWidget(), "Select dir", "C:\\Users", QFileDialog.Option.ShowDirsOnly)
        self.dirTitle.setText(dir)
        if dir != "":
            self.emptyError = False
            self.confirmButton.setEnabled(True)
            self.selectAllButton.setEnabled(True)
            self.errorLabel.setText("")
        else:
            self.emptyError = True
            self.confirmButton.setDisabled(True)
            self.selectAllButton.setDisabled(True)
            self.errorLabel.setText("Директория не выбрана!")
        self.dicomDir.insert(str(dir))
    def confirmeButtonClicked(self):
        if len(self.selected) != 0:
            if self.it > 1:
                print("it:" + str(self.it))
            self.selectButton.setDisabled(True)
            self.selectAllButton.setDisabled(True)
            now = datetime.datetime.now()
            time = str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + str(
                now.second)
            self.log = open("logs/log_" + time + ".csv", "w+")

            print("CT scans: " + str(len(list(self.selected))))
            self.progressBar.setValue(0)
            self.progressBar.setMaximum(len(list(self.selected)))
            self.confirmButton.setDisabled(True)
            self.selectAllButton.setDisabled(True)
            self.master_thread.start()
            self.errorLabel.setText("")
            proceed_paths = []
            for path in self.paths:
                if path.rpartition('\\')[2] in self.selected:
                    proceed_paths.append(path)
            i = self.listWidget.count()-1
            while(i>-1):
                self.listWidget.takeItem(i)
                i -= 1
            for study in self.selected:
                item = QListWidgetItem(study, self.listWidget)
                font = item.font()
                font.setWeight(600)
                item.setFont(font)
            self.connectedSlots = False
            self.listWidget.itemClicked.disconnect()
            self.selectedCount.valueChanged.disconnect()
            self.progressMsg.setText("Обработка исследований...")
            self.work_to_master.emit(proceed_paths)


        else:
            self.errorLabel.setText("Выберите хотя бы одно исследование!")
        # self.start_master.emit()

    # def start(self):
    #


    def selectedValueChanged(self):
        #self.selectedAmount.setText(str(self.selectedCount.value()))
        self.selectedLabel.setText("Выбрано: " + str(self.selectedCount.value()))

    def dirSelected(self):
        if not self.connectedSlots:
            self.selectedCount.valueChanged.connect(self.selectedValueChanged)
            self.listWidget.itemClicked.connect(self.listItemClicked)
            self.dicomDir.textChanged.disconnect()
            self.dicomDir.setText("")
            self.dicomDir.textChanged.connect(self.dirSelected)
            self.connectedSlots=True
            i = self.outputTable.rowCount() - 1
            while i > 9:
                for j in range(4):
                    self.outputTable.takeItem(i, j)
                i -= 1
            while i > -1:
                for j in range(4):
                    self.outputTable.item(i, j).setText("")
                i -= 1
        if self.listWidget.count() != 0:
            i = self.listWidget.count() - 1
            while i > -1:
                self.listWidget.takeItem(i)
                i -= 1
        self.paths = []
        self.selected = set()
        self.selectedCount.setValue(0)
        print(self.dicomDir.text())
        if not self.emptyError:
            for path in os.listdir(self.dirTitle.text()):
                study = path.rpartition('\\')[2]
                if study[0] != '.':
                    self.paths.append(os.path.join(os.getcwd(), self.dirTitle.text(), study))
                    item = QListWidgetItem(path.rpartition('\\')[2], self.listWidget)
            self.selectedCount.setMaximum(len(self.paths))



    def update_progress_master(self):
        self.progressBar.setValue(self.progressBar.value()+1)

    def completed_master(self, result):
        # i = self.listWidget.count()-1
        # while i > -1:
        #     self.listWidget.takeItem(i)
        #     i -= 1

        self.log.write("index,study,front,mid,back\n")

        self.progressBar.setValue(0)
        self.progressBar.setMaximum(len(result))
        self.progressMsg.setText("Получение прогнозов...")
        self.start_predict.emit(result)
    #index, file_name, front, mid, back
    def predict_update(self, index, output):
        for i in range(4):
            item = QTableWidgetItem(str(output[i]))
            # item.setFlags(Qt.ItemFlag.NoItemFlags)
            # item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            # item.setFlags(Qt.ItemFlag.ItemIsSelectable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.outputTable.setItem(index, i, item)
        #item = QListWidgetItem(file_name + " - " + str(prediction), self.listWidget)
        self.log.write(str(index+1) + "," + output[0] + "," + str(output[1]) + "," + str(output[2]) + str(output[3]) + "\n")
        print(str(index+1) + "," + output[0] + "," + str(output[1]) + "," + str(output[2]) + str(output[3]))

    def listItemClicked(self, item):
        font = QFont(item.font())
        if item.font().weight() == 400:
            self.selected.add(item.text())
            font.setWeight(600)
            item.setFont(font)
            self.selectedCount.setValue(self.selectedCount.value()+1)
        else:
            self.selected.remove(item.text())
            font.setWeight(400)
            item.setFont(font)
            self.selectedCount.setValue(self.selectedCount.value()-1)

    def predict_finished(self):
        self.progressMsg.setText("Готово")
        self.log.close()
        self.it += 1
        self.master_thread.exit()
        self.selectButton.setEnabled(True)

app = QApplication(sys.argv)

window = MainWindow()
window.setMinimumSize(1000, 600)
window.show()

app.exec()
