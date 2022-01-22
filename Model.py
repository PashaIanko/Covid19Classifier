from abc import abstractclassmethod
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np

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

    def flow_predict(self, flow, steps):
        flow.reset()
        y_pred = self.model.predict(flow, steps)
        return np.argmax(y_pred, axis = 1)

    def compile_model(self):
        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = self.metrics
        )

    def evaluate_metrics(self, flow, steps):
        res_dict = {}
        Y_pred = self.flow_predict(flow, steps)

        params = {
            'y_true': flow.classes,
            'y_pred': Y_pred,
            'average': 'macro'
        }

        res_dict['F1'] = f1_score(
            **params
        )

        res_dict['Precision'] = precision_score(
            **params
        )

        res_dict['Recall'] = recall_score(
            **params
        )

        res_dict['Accuracy'] = accuracy_score(
            y_true = flow.classes,
            y_pred = Y_pred
        )

        return res_dict







