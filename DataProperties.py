class DataProperties:

    covid_class = 0
    pneumonia_class = 1
    healthy_class = 2
    n_classes = 3
    classes = ['covid', 'pneumonia', 'normal']

    def __init__(self, environment):

        self.envorinment = environment #'pc'  # 'colab'
        
        self.train_data_path = None
        self.test_data_path = None

        self.assign_data_paths()

        self.train_covid_path = self.train_data_path + 'covid/'
        self.train_pneumonia_path = self.train_data_path + 'pneumonia/'
        self.train_healthy_path = self.train_data_path + 'normal/'

        self.val_covid_path = self.val_data_path + 'covid/'
        self.val_pneumonia_path = self.val_data_path + 'pneumonia/'
        self.val_healthy_path = self.val_data_path + 'normal/'

        self.test_covid_path = self.test_data_path + 'covid/'
        self.test_pneumonia_path = self.test_data_path + 'pneumonia/'
        self.test_healthy_path = self.test_data_path + 'normal/'
    
    def assign_data_paths(self):
        if self.envorinment == 'pc':
            self.checkpoint_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Checkpoints/'
            self.train_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/TrainData/'
            self.test_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/TestData/'
            self.val_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/ValidationData/'
        elif self.envorinment == 'colab':
            self.checkpoint_path = '/content/drive/MyDrive/UNIPD/HDA/Project/Checkpoints/'
            self.train_data_path = '/content/drive/MyDrive/UNIPD/HDA/Project/TrainData/'
            self.test_data_path = '/content/drive/MyDrive/UNIPD/HDA/Project/TestData/'
            self.val_data_path =  '/content/drive/MyDrive/UNIPD/HDA/Project/ValidationData/'
