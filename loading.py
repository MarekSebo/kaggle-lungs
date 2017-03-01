import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image as pilimg
from numpy import random
import cv2
import logging
import pandas as pd


# cesta k obrazkom
url = os.getcwd()



class DataClass(object):
    """
    POUZITIE
    -vytvor instanciu meno = DataClass(args)
    -vypytaj si novy batch: meno.next_batch()
    BACKEND
    -data sa nacitavaju v chunkoch zadanej velkosti
    -batche sa vzdy beru zaradom z aktualneho chunku
    -ked sa minie chunk (dojde sa na koniec), nacita sa novy chunk
    -ked sa minu chunky, premiesaju sa data a zacne sa znova
    """
    def __init__(self, path, labels_path, batch_size, chunk_size, h, w, augm=False, data_use="train"):
        self.data = None
        self.labels = None

        self.path = path
        self.labels_path = labels_path
        self.data_use = data_use
        self.augm = augm

        self.h = h
        self.w = w

        self.dirnames_and_labels = self.load_dirnames_and_labels()

        self.total_data_size = len(self.dirnames_and_labels)

        self.batch_size = batch_size
        self.batch_cursor = 0              # pozicia batchu vramci chunku

        self.chunk_size = chunk_size        # (chunk_size // batch_size) * batch_size
        self.chunk_cursor = 0           # pozicia chunku vramci datasetu
        assert self.batch_size <= self.chunk_size
        self.next_chunk()

    @staticmethod
    def load_labels(self, csv_name='stage1_labels.csv', path=''):
        # labels from csv to 1-D pandas df
        labels_df = pd.read_csv(csv_name, index_col=0)
        return labels_df

    ## load folder name
    @staticmethod
    def load_dirnames(path):
        dirnames = []
        for dir in os.listdir(path):
            dirnames.append(dir)
            dirnames.sort()
        return dirnames

    def load_dirnames_and_labels(self):
        self.dirnames = self.load_dirnames(self.path)
        self.labels = self.load_labels(self.labels_path)

        # sanity checks
        assert len(self.dirnames) == len(self.labels)
        for dirname, label in self.dirnames, self.labels.index:
            if label not in dirname:
                logging.error("Some of the labels and directory names dont match. Check filenames.")
                exit(1)
                # TODO: do the dirname-label pairs match?

        # zip
        labels_np = np.array(self.labels) # shape (n, 1)
        self.dirnames_and_labels = zip(self.dirnames, labels_np)


    def shuffle(self):
        if self.data_use == "train":
            random.shuffle(self.dirnames_and_labels)

    def load_chunk(self):
        # input: zipped pairs dirname, label
        # loads: data chunk_size x no_imgs x img_width x img_height

        chunk_imgs = []
        chunk_labels = []
        assert self.chunk_size <= 1
        # TODO: solve the problem with variable image counts in dirs and alter the code

        # even though cv2 works with BGR colourspace, it changes the channels automatically, so no need to change to BGR manually
        for dirname, label in self.dirnames_and_labels[self.chunk_cursor:self.chunk_cursor + self.chunk_size]:
            # load labels

            for img_filename in os.listdir(os.path.join(self.path, dirname)):
                dir_imgs = []

                img = cv2.imread(os.path.join(self.path, img_filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.astype(float) / 255

                if img.shape != (self.h, self.w, 3):
                    logging.warning("Some images had different size and were resized to {} x {}.".format(self.w, self.h))
                    img = cv2.resize(img, (self.w, self.h))

                #if self.data_use == 'train' and self.augm:
                    # place for augmentations
                dir_imgs.append(img)

            chunk_imgs.append(dir_imgs)
            chunk_labels.append(label)

        self.chunk_cursor += self.chunk_size
        self.current_chunk_size = len(chunk_imgs)

        # docasne riesenie
        if self.chunk_cursor + self.chunk_size > self.total_data_size:
            print('last chunk of the epoch')
            self.chunk_cursor = 0
            self.shuffle()

        return np.array(chunk_imgs), np.array(chunk_labels)


    def next_chunk(self):
        self.data, self.labels = self.load_chunk()

    def next_batch(self):
        data = self.data[self.batch_cursor:self.batch_cursor + self.batch_size]
        labels = self.labels[self.batch_cursor:self.batch_cursor + self.batch_size]

        self.batch_cursor += self.batch_size
        if self.batch_cursor + self.batch_size > self.current_chunk_size:
            self.batch_cursor = 0
            self.next_chunk()

        if len(labels) < self.batch_size:
            self.next_batch()


        return data, labels