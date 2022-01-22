class DatasetParameters:
    batch_size = 32
    seed = 123

    validation_split = 0.3

    shuffle_train = True
    shuffle_validation = False
    shuffle_test = False

    epochs = 500

    initial_subset = 200  # Use only 200 images to check if everything works well
