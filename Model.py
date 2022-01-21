from abc import ABC, abstractclassmethod

class Model:
    def __init__(self):
        self.name = None
        self.model = None

        self.init_name()

    @abstractclassmethod
    def init_name(self):
        pass

    @abstractclassmethod
    def construct_model(self):
        pass

    @abstractclassmethod
    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )





