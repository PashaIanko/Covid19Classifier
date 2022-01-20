# Load filenames
import os
import numpy as np
from DataProperties import DataProperties
import cv2
import tensorflow as tf

def load_filenames(data_path, max_files = None):
    p = os.listdir(data_path)
    if max_files is not None:
        p = p[: min(max_files, len(p))]
    p = [data_path + file_path for file_path in p]
    return p

def get_filenames(
    covid_path, pneumonia_path, normal_path
):
    return (
        load_filenames(covid_path),
        load_filenames(pneumonia_path),
        load_filenames(normal_path)
    )

def get_labels(
    covid_fnames,
    pn_fnames,
    normal_fnames
):
    return (
        np.full(len(covid_fnames), fill_value = DataProperties.covid_class),
        np.full(len(pn_fnames), fill_value = DataProperties.pneumonia_class),
        np.full(len(normal_fnames), fill_value = DataProperties.healthy_class)
    )

def getXY(covid_fnames, pn_fnames, normal_fnames,
          covid_labels, pn_labels, normal_labels):
    X = [
         *covid_fnames, *pn_fnames, *normal_fnames
    ]
    Y = [
         *covid_labels, *pn_labels, *normal_labels
    ]
    return X, Y

def load_image(full_path):
    # print(f'Loading, {full_path}')
    img = cv2.imread(full_path, cv2.IMREAD_COLOR)
    # print(type(img))
    return img

def get_dataset(
    path,
    batch_size,
    image_size,
    shuffle,
    seed,
    subset,
    validation_split = None
):  

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels = 'inferred',
        label_mode = 'int',
        color_mode = 'rgb',
        batch_size = batch_size,
        image_size = image_size,
        shuffle = shuffle,
        seed = seed,
        validation_split = validation_split,
        subset = subset
    )

    #ds = ds.repeat()
    return {
        'class names': ds.class_names,
        'data': ds.cache().prefetch(
            buffer_size = tf.data.experimental.AUTOTUNE
        )
    }
    