class DatasetParameters:
    batch_size = 32
    seed = 123

    validation_split = 0.3

    shuffle_train = True
    shuffle_validation = False
    shuffle_test = False

    epochs = 500

    width_shift_range = 0.09
    height_shift_range = 0.09
    
    horizontal_flip = True
    vertical_flip = True
    
    rotation_range = 0.2  # 0.1
    zoom_range = [0.9, 1.1]
    samplewise_center = False
    featurewise_center = False