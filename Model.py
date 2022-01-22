from abc import ABC, abstractclassmethod

class Model:
    def __init__(
        self, 
        **kwargs
    ):
        self.name = None
        self.model = None
        self.optimizer = kwargs['optimizer']
        self.loss = kwargs['loss']
        self.metrics = kwargs['metrics']
        self.checkpoint_callback = kwargs['checkpoint']

        self.init_name()

    @abstractclassmethod
    def init_name(self):
        pass

    @abstractclassmethod
    def construct_model(self):
        pass





