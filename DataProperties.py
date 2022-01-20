class DataProperties:
    #train_data_path = '/content/drive/MyDrive/UNIPD/HDA/Project/TrainData/'
    #test_data_path = # '/content/drive/MyDrive/UNIPD/HDA/Project/TestData/'
    
    train_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/TrainData/'
    test_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/TestData/'
    
    train_covid_path = train_data_path + 'covid/'
    train_pneumonia_path = train_data_path + 'pneumonia/'
    train_healthy_path = train_data_path + 'normal/'

    test_covid_path = test_data_path + 'covid/'
    test_pneumonia_path = test_data_path + 'pneumonia/'
    test_healthy_path = test_data_path + 'normal/'

    healthy_class = 0
    covid_class = 1
    pneumonia_class = 2
    n_classes = 3
    classes = ['covid', 'pneumonia', 'normal']

    strategy = 'check'  # 'normal'
