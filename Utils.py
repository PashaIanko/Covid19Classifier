# Load filenames
import os
import numpy as np
from DataProperties import DataProperties
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

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

def plot_history(history, metrics_name, plot_validation, figsize = (12, 8)):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(
        history[metrics_name], 
        label = metrics_name,
        marker = 'o',
        markersize = 11,
        markerfacecolor = 'white'
    )
    
    if plot_validation:
        plt.plot(
            history['val_' + metrics_name], 
            label = metrics_name + ' (validation)',
            marker = 'o',
            markersize = 11,
            markerfacecolor = 'white'
        )
    
    plt.xlabel('Epoch')
    plt.ylabel(metrics_name)
    plt.ylim([0.01, 1])
    plt.legend(loc = 'lower right')
    plt.grid()

def visualize(batch, labels, n_subplots):
    for i in range(n_subplots): #(batch_size):
        ax = plt.subplot(
            int(np.sqrt(n_subplots)), 
            int(np.sqrt(n_subplots)), 
            i + 1
        )
        plt.imshow(batch[i])
        plt.title(str(labels[i]))
        plt.axis("off")

    