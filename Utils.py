# Load filenames
import os
import numpy as np
from DataProperties import DataProperties
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import convolve2d


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



def plot_confusion_matrix(Y_true, Y_pred, class_indices):

    #fig, axes = plt.subplots(1, 1, figsize = (5, 5))
    
    #axes.set_ylabel('True', fontdict={'size': '16'})
    #axes.set_xlabel('Predicted', fontdict={'size': '16'})
    #axes.tick_params(axis='both', labelsize=17)

    cm = confusion_matrix(
        Y_true, 
        Y_pred, 
        normalize = 'true'
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels = [k for k in class_indices.keys()] #translate_labels(class_indices),
    )
    disp.plot(
        cmap = 'Oranges',
        xticks_rotation = 'vertical',
    )

    #plt.title(f'Confusion matrix for {model_name}', fontsize = 18)
    plt.show()


def visualize_convolutions(
    image, 
    kernels, 
    label,
    n_color_channels
):
    rgb_components = [
        image[:, :, channel] for channel in range(n_color_channels)
    ]
    
    assert len(rgb_components) == len(kernels)

    convolved = [
        convolve2d(
            rgb_components[i], 
            kernels[i],
            mode = 'same'
        ) 
        for i in range(n_color_channels)
    ]

    _, axes = plt.subplots(1, 4, figsize = (15, 15))

    axes[0].imshow(image)
    axes[0].set_title('Source image')
    
    for i in range(len(convolved)):
        axes[i + 1].imshow(convolved[i])
        axes[i + 1].set_title(f'Color channel {i + 1}')
    return convolved

def visualize_kernels(kernels):
    n_subplots = len(kernels)
    _, axes = plt.subplots(1, n_subplots)
    for i, kernel in enumerate(kernels):
        ax = axes[i]
        ax.imshow(kernel)
        ax.set_title(f'Color channel {i + 1}')

def fit_(model, train_flow, train_steps, val_flow, val_steps, epochs, callbacks):
    history = model.fit(
        train_flow,
        steps_per_epoch = train_steps,
        
        validation_data = val_flow,
        validation_steps = val_steps,

        epochs = epochs,
        callbacks = callbacks
    )
    return history

def visualize_kernel_work(model, n_layer, n_kernel, image, label, n_color_channels):
    
    conv_layer = model.layers[n_layer]
    kernels = conv_layer.get_weights()[0]

    color_kernels = [
        kernels[:, :, color_ch, n_kernel]
        for color_ch in range(n_color_channels)
    ]

    kern_shape = kernels.shape

    print(
        f'''We have:
        {kern_shape[0]} by {kern_shape[1]} kernel, 
        of {kern_shape[2]} color channels,
        total: {kern_shape[3]} kernels'''
    )

    visualize_kernels(color_kernels)

    _ = visualize_convolutions(
        image,
        color_kernels,
        label = label,
        n_color_channels = n_color_channels
    )


def collect_metrics(models_dict, data_flow, data_steps):
    res_dict = {k: {} for k in models_dict.keys()}

    for name, model in models_dict.items():
        data_flow.reset()
        eval_res = model.model.evaluate(
           data_flow,
           steps = data_steps
        )

        res_dict[name]['test_loss^(-1)'] = eval_res[0]  # loss
        res_dict[name]['test_accuracy'] = eval_res[1]  # accuracy


        data_flow.reset()
        metrics = model.evaluate_metrics(
            data_flow,
            data_steps
        )

        res_dict[name]['F1'] = metrics['F1']
        res_dict[name]['precision'] = metrics['Precision']
        res_dict[name]['recall'] = metrics['Recall']

        # number of trainable parameters
        trainable_params = np.sum(
            [np.prod(v.get_shape()) for v in model.model.trainable_weights]
        )
        res_dict[name]['tr_params'] = int(trainable_params)
        # nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        # totalParams = trainableParams + nonTrainableParams
        
    return res_dict

def calc_files(directory):
    total_files = 0

    for base, _, files in os.walk(directory):
        # print('Searching in : ',base)
        for _ in files:
            total_files += 1
    return total_files


    