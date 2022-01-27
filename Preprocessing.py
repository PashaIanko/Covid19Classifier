import tensorflow as tf
from skimage.exposure import equalize_adapthist
from cv2 import bilateralFilter
from PreprocessingParameters import PreprocessingParameters

def preprocess(image):

    image = image / 255.
    image = equalize_adapthist(image, clip_limit = 0.015)
    
    image = bilateralFilter(
        image,
        d = 2, #PreprocessingParameters.d,  # kernel
        sigmaColor = 12, # PreprocessingParameters.sigma_color,  # Gauss blur component
        sigmaSpace = 2# PreprocessingParameters.sigma_space  # Range component
    )

    return image