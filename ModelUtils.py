class ModelUtils:
    def __init__(
        self, 
        model_params_dict,
        checkpoint_params_dict,
        train_params_dict
    ):
        self.model_params_dict = model_params_dict
        self.checkpoint_params_dict = checkpoint_params_dict
        self.train_params_dict = train_params_dict