from datetime import date
from os.path import isdir
from os import mkdir

class DataProperties:

    covid_class = 0
    pneumonia_class = 1
    healthy_class = 2
    n_classes = 3
    classes = ['covid', 'pneumonia', 'normal']

    def __init__(self, environment, n_trial):

        self.envorinment = environment #'pc'  # 'colab'
        self.n_trial = n_trial
        
        self.train_data_path = None
        self.test_data_path = None

        self.assign_data_paths()
        self.update_save_path()

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
            #self.models_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/SavedModels/'
            self.checkpoint_path = f'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Checkpoints/{str(date.today())}/'
            self.train_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/TrainData/'
            self.test_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/TestData/'
            self.val_data_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/Data/ValidationData/'
        elif self.envorinment == 'colab':
            #self.models_path = '/content/drive/MyDrive/UNIPD/HDA/Project/SavedModels/'
            self.checkpoint_path = '/content/drive/MyDrive/UNIPD/HDA/Project/Checkpoints/'
            self.train_data_path = '/content/drive/MyDrive/UNIPD/HDA/Project/TrainData/'
            self.test_data_path = '/content/drive/MyDrive/UNIPD/HDA/Project/TestData/'
            self.val_data_path =  '/content/drive/MyDrive/UNIPD/HDA/Project/ValidationData/'
    
    def update_save_path(self):
        today_date = str(date.today())
        if self.envorinment == 'pc':
            self.core_path = 'C:/Users/79137/Pasha/2. UNIPD/HDA/Project/SavedModels/'
        elif self.envorinment == 'colab':
            self.core_path = '/content/drive/MyDrive/UNIPD/HDA/Project/SavedModels/'

        today_path = f'{self.core_path}/{today_date}/'
        models_path = f'{self.core_path}/{today_date}/trial-{str(self.n_trial)}/'
        histories_path = f'{models_path}Histories/'

        paths = [self.core_path, today_path, models_path, histories_path]
        for p in paths:
            if not (isdir(p)):
                mkdir(p)

        self.models_path = models_path
        self.histories_path = histories_path